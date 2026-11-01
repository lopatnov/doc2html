"""Tests for tools/fetch_random_doc.py: the two catalog lookups, the
fallback/circuit-breaker between them, and the download size cap. No network:
HTTP is replaced by tiny fake openers.

Run: python -m unittest discover -s tests -p "test_*.py"
"""
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import fetch_random_doc as frd

RDF_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:dcterms="http://purl.org/dc/terms/"
         xmlns:pgterms="http://www.gutenberg.org/2009/pgterms/">
  <pgterms:ebook rdf:about="ebooks/{book_id}">
    <dcterms:title>La Quimera</dcterms:title>
    <dcterms:rights>{rights}</dcterms:rights>
    <dcterms:creator>
      <pgterms:agent rdf:about="2009/agents/1">
        <pgterms:name>Pardo Bazán, Emilia, condesa de</pgterms:name>
      </pgterms:agent>
    </dcterms:creator>
    {files}
  </pgterms:ebook>
</rdf:RDF>
"""


def rdf_file(url, mime):
    return f"""
    <dcterms:hasFormat>
      <pgterms:file rdf:about="{url}">
        <dcterms:format>
          <rdf:Description rdf:nodeID="n{abs(hash(url)) % 10000}">
            <rdf:value rdf:datatype="http://purl.org/dc/terms/IMT">{mime}</rdf:value>
          </rdf:Description>
        </dcterms:format>
      </pgterms:file>
    </dcterms:hasFormat>"""


def make_rdf(book_id=49756, rights="Public domain in the USA.", files=None):
    if files is None:
        files = [
            ("https://www.gutenberg.org/ebooks/{id}.epub.noimages", "application/epub+zip"),
            ("https://www.gutenberg.org/ebooks/{id}.epub.images", "application/epub+zip"),
            ("https://www.gutenberg.org/ebooks/{id}.epub3.images", "application/epub+zip"),
            ("https://www.gutenberg.org/ebooks/{id}.txt.utf-8", "text/plain; charset=utf-8"),
        ]
    body = "".join(rdf_file(url.format(id=book_id), mime) for url, mime in files)
    return RDF_TEMPLATE.format(book_id=book_id, rights=rights, files=body).encode("utf-8")


class FakeResponse:
    def __init__(self, data):
        self._data = data
        self._pos = 0

    def read(self, n=-1):
        chunk = self._data[self._pos:] if n < 0 else self._data[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    """Maps a URL to bytes, or to an exception instance to raise."""

    def __init__(self, routes):
        self.routes = routes
        self.requested = []

    def open(self, req, timeout=None):
        self.requested.append(req.full_url)
        result = self.routes[req.full_url]
        if isinstance(result, Exception):
            raise result
        return FakeResponse(result)


def http_error(url, code):
    return urllib.error.HTTPError(url, code, "x", {}, None)


class ParseRdfTests(unittest.TestCase):
    def test_public_domain_record(self):
        candidate = frd.parse_gutenberg_rdf(make_rdf(), 49756)
        self.assertEqual(candidate["title"], "La Quimera")
        self.assertEqual(candidate["authors"], ["Pardo Bazán, Emilia, condesa de"])
        self.assertIn("Public domain in the USA.", candidate["license_basis"])

    def test_most_complete_file_wins_per_mime_type(self):
        formats = frd.parse_gutenberg_rdf(make_rdf(), 49756)["formats"]
        self.assertEqual(formats["application/epub+zip"], "https://www.gutenberg.org/ebooks/49756.epub3.images")
        self.assertEqual(formats["text/plain; charset=utf-8"], "https://www.gutenberg.org/ebooks/49756.txt.utf-8")

    def test_copyrighted_record_is_skipped(self):
        raw = make_rdf(rights="Copyrighted. Read the copyright notice inside this book for details.")
        self.assertIsNone(frd.parse_gutenberg_rdf(raw, 1))

    def test_record_without_rights_is_skipped(self):
        self.assertIsNone(frd.parse_gutenberg_rdf(make_rdf(rights=""), 1))

    def test_unexpected_document_is_an_error_not_a_skip(self):
        raw = b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>'
        with self.assertRaises(ValueError):
            frd.parse_gutenberg_rdf(raw, 1)
        with self.assertRaises(Exception):
            frd.parse_gutenberg_rdf(b"<html>not xml", 1)

    def test_picked_format_prefers_epub(self):
        formats = frd.parse_gutenberg_rdf(make_rdf(), 49756)["formats"]
        mime_prefix, _key, url = frd.gutenberg_http.pick_format(formats)
        self.assertEqual(mime_prefix, "application/epub+zip")
        self.assertTrue(url.endswith(".epub3.images"))


class LookupTests(unittest.TestCase):
    def test_rdf_lookup_reads_the_catalog_record(self):
        url = frd.GUTENBERG_RDF_URL.format(book_id=49756)
        opener = FakeOpener({url: make_rdf()})
        with mock.patch.object(frd, "_DOWNLOAD_OPENER", opener):
            candidate = frd.lookup_gutenberg_rdf(49756)
        self.assertEqual(candidate["title"], "La Quimera")
        self.assertEqual(opener.requested, [url])

    def test_rdf_404_means_no_such_book(self):
        url = frd.GUTENBERG_RDF_URL.format(book_id=7)
        with mock.patch.object(frd, "_DOWNLOAD_OPENER", FakeOpener({url: http_error(url, 404)})):
            self.assertIsNone(frd.lookup_gutenberg_rdf(7))

    def test_rdf_server_error_is_raised(self):
        url = frd.GUTENBERG_RDF_URL.format(book_id=7)
        with mock.patch.object(frd, "_DOWNLOAD_OPENER", FakeOpener({url: http_error(url, 503)})):
            with self.assertRaises(urllib.error.HTTPError):
                frd.lookup_gutenberg_rdf(7)

    def test_gutendex_lookup_requires_copyright_false(self):
        url = frd.GUTENDEX_BOOK_URL.format(book_id=5)
        for copyright_value, expected in ((False, True), (True, False), (None, False)):
            body = json.dumps({"title": "T", "authors": [{"name": "A"}], "formats": {}, "copyright": copyright_value})
            with mock.patch.object(frd, "_API_OPENER", FakeOpener({url: body.encode()})):
                self.assertEqual(frd.lookup_gutendex(5) is not None, expected, copyright_value)


class FallbackTests(unittest.TestCase):
    def sources(self, mode="auto", gutendex=None, rdf=None):
        calls = []

        def make(name, fn):
            def lookup(book_id):
                calls.append((name, book_id))
                return fn(book_id)
            return lookup

        sources = frd.LookupSources(
            mode,
            gutendex=make("gutendex", gutendex or (lambda _id: {"source": "g"})),
            rdf=make("rdf", rdf or (lambda _id: {"source": "r"})),
        )
        return sources, calls

    @staticmethod
    def down(_book_id):
        raise TimeoutError("The read operation timed out")

    def test_gutendex_answer_is_used_without_asking_the_catalog(self):
        sources, calls = self.sources()
        self.assertEqual(frd.find_candidate(1, 1, sources), {"source": "g"})
        self.assertEqual(calls, [("gutendex", 1)])

    def test_a_skip_answer_is_final(self):
        sources, calls = self.sources(gutendex=lambda _id: None)
        self.assertIsNone(frd.find_candidate(1, 1, sources))
        self.assertEqual(calls, [("gutendex", 1)])

    def test_gutendex_failure_falls_back_to_the_catalog_for_the_same_id(self):
        sources, calls = self.sources(gutendex=self.down)
        self.assertEqual(frd.find_candidate(9, 1, sources), {"source": "r"})
        self.assertEqual(calls, [("gutendex", 9), ("rdf", 9)])

    def test_gutendex_is_skipped_after_repeated_failures(self):
        sources, calls = self.sources(gutendex=self.down)
        for book_id in range(1, 6):
            frd.find_candidate(book_id, book_id, sources)
        gutendex_calls = [c for c in calls if c[0] == "gutendex"]
        self.assertEqual(len(gutendex_calls), frd.GUTENDEX_MAX_CONSECUTIVE_FAILURES)
        self.assertEqual([c for c in calls if c[0] == "rdf"][-1], ("rdf", 5))

    def test_a_gutendex_success_resets_the_failure_count(self):
        outcomes = iter([TimeoutError, TimeoutError, "ok", TimeoutError, TimeoutError, TimeoutError])

        def flaky(_book_id):
            outcome = next(outcomes)
            if outcome is TimeoutError:
                raise TimeoutError("timed out")
            return {"source": "g"}

        sources, calls = self.sources(gutendex=flaky)
        for book_id in range(1, 5):
            frd.find_candidate(book_id, book_id, sources)
        self.assertFalse(sources.gutendex_skipped)

    def test_pinned_modes_never_use_the_other_catalog(self):
        sources, calls = self.sources("gutendex", gutendex=self.down)
        self.assertIsNone(frd.find_candidate(1, 1, sources))
        self.assertEqual(calls, [("gutendex", 1)])

        sources, calls = self.sources("rdf")
        self.assertEqual(frd.find_candidate(1, 1, sources), {"source": "r"})
        self.assertEqual(calls, [("rdf", 1)])

    def test_both_down_means_skip(self):
        sources, _calls = self.sources(gutendex=self.down, rdf=self.down)
        self.assertIsNone(frd.find_candidate(1, 1, sources))


class TryFetchBookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def candidate(url="https://www.gutenberg.org/ebooks/5.epub3.images"):
        return {
            "source": "test", "title": "T", "authors": ["A"], "license_basis": "test basis",
            "formats": {"application/epub+zip": url},
        }

    def run_with(self, candidate, routes):
        sources = frd.LookupSources(gutendex=lambda _id: candidate)
        with mock.patch.object(frd, "_DOWNLOAD_OPENER", FakeOpener(routes)):
            return frd.try_fetch_book(5, 1, self.out_dir, sources)

    def test_downloads_the_file_and_writes_metadata(self):
        url = "https://www.gutenberg.org/ebooks/5.epub3.images"
        self.assertTrue(self.run_with(self.candidate(url), {url: b"epub-bytes"}))
        self.assertEqual((self.out_dir / "gutenberg_5.epub").read_bytes(), b"epub-bytes")
        metadata = json.loads((self.out_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["gutenberg_id"], 5)
        self.assertEqual(metadata["license_basis"], "test basis")
        self.assertEqual(metadata["local_file"], "gutenberg_5.epub")

    def test_file_over_the_publish_limit_is_skipped_without_leaving_a_file(self):
        url = "https://www.gutenberg.org/ebooks/5.epub3.images"
        with mock.patch.object(frd, "MAX_PUBLISHABLE_FILE_BYTES", 1024):
            self.assertFalse(self.run_with(self.candidate(url), {url: b"x" * 2048}))
        self.assertEqual(list(self.out_dir.iterdir()), [])

    def test_download_from_a_foreign_host_is_refused(self):
        url = "https://evil.example.com/book.epub"
        self.assertFalse(self.run_with(self.candidate(url), {}))
        self.assertEqual(list(self.out_dir.iterdir()), [])

    def test_book_without_a_usable_format_is_skipped(self):
        candidate = self.candidate()
        candidate["formats"] = {"image/jpeg": "https://www.gutenberg.org/cover.jpg"}
        self.assertFalse(self.run_with(candidate, {}))


if __name__ == "__main__":
    unittest.main()
