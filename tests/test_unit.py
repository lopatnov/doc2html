"""Fast unit tests with no real books and no OCR model: pure functions plus
one end-to-end conversion of a tiny PDF built on the fly with PyMuPDF.

Unlike tests/regression_check.py (which needs the maintainer's local books
and is skipped almost entirely in CI), everything here runs everywhere.

Run: python -m unittest discover -s tests -p "test_*.py"
"""
import contextlib
import io
import json
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import unquote

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import doc2html


def make_block(*lines):
    """A PyMuPDF-style text block; each line is a str or a list of
    (text, flags) spans."""
    block_lines = []
    for line in lines:
        spans = [{"text": line, "flags": 0}] if isinstance(line, str) else [
            {"text": text, "flags": flags} for text, flags in line
        ]
        block_lines.append({"spans": spans})
    return {"type": 0, "lines": block_lines}


class HyphenWrapTests(unittest.TestCase):
    def test_wrapped_word_is_rejoined(self):
        self.assertEqual(doc2html.join_wrapped_lines(["inter-", "national trade"]), "international trade")
        self.assertEqual(doc2html.join_wrapped_lines(["бо-", "лее того"]), "более того")
        self.assertEqual(doc2html.join_wrapped_lines(["año-", "ranza ñu"]), "añoranza ñu")

    def test_trailing_blanks_around_the_hyphen_are_ignored(self):
        self.assertEqual(doc2html.join_wrapped_lines(["inter- ", " national"]), "international")

    def test_next_line_in_uppercase_is_not_a_continuation(self):
        self.assertEqual(doc2html.join_wrapped_lines(["Нью-", "Йорк"]), "Нью- Йорк")

    def test_hyphen_not_after_a_letter_is_kept(self):
        self.assertEqual(doc2html.join_wrapped_lines(["2020-", "и дальше"]), "2020- и дальше")
        self.assertEqual(doc2html.join_wrapped_lines(["a -", "b"]), "a - b")

    def test_suspended_hyphen_inside_a_line_is_untouched(self):
        for text in ("two- and three-dimensional models", "научно- и технический прогресс", "pre- and post-processing"):
            self.assertEqual(doc2html.clean_text(text), text)

    def test_block_text_unwraps_across_lines_only(self):
        block = make_block("A two- and three-dimensional inter-", "national model")
        self.assertEqual(doc2html.extract_block_text(block), "A two- and three-dimensional international model")

    def test_runs_unwrap_at_the_same_boundaries(self):
        italic = 2
        block = make_block(
            [("see the ", 0), ("inter-", italic)],
            [("national", italic), (" trade", 0)],
        )
        runs = doc2html.extract_block_runs(block)
        self.assertEqual("".join(r[0] for r in runs), "see the international trade")
        self.assertEqual(doc2html.extract_block_text(block), "see the international trade")

    def test_ocr_rows_are_reflowed_with_unwrap(self):
        rows = [
            {"x0": 0, "x1": 100, "y0": 0, "y1": 10, "text": "interna-"},
            {"x0": 0, "x1": 100, "y0": 11, "y1": 21, "text": "tional trade is up"},
        ]
        paragraphs = doc2html.reflow_rows_into_paragraphs(rows)
        self.assertEqual([p["text"] for p in paragraphs], ["international trade is up"])

    def test_table_cell_wrapped_over_lines(self):
        rows = doc2html.clean_table_rows([["Name", "Description"], ["a", "inter-\nnational trade"]])
        self.assertEqual(rows[1], ["a", "international trade"])

    def test_list_item_wrapped_over_lines(self):
        block = make_block("► first inter-", "national item", "► second item")
        self.assertEqual(
            doc2html.split_block_into_list_items(block),
            [("first international item", False), ("second item", False)],
        )


class LinkSafetyTests(unittest.TestCase):
    def test_dangerous_schemes_are_rejected(self):
        for href in (
            "javascript:alert(1)",
            "JavaScript:alert(1)",
            "  javascript:alert(1)",
            "java\tscript:alert(1)",
            "java\nscript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "vbscript:msgbox(1)",
            "file:///etc/passwd",
        ):
            self.assertIsNone(doc2html.safe_href(href), href)

    def test_ordinary_targets_are_kept(self):
        for href in (
            "https://example.com/a?b=c#d",
            "http://example.com",
            "HTTPS://EXAMPLE.COM",
            "mailto:someone@example.com",
            "tel:+380501234567",
            "#page-12",
            "../other.pdf",
            "//example.com/x",
        ):
            self.assertEqual(doc2html.safe_href(href), href)

    def test_html_drops_the_link_but_keeps_the_text(self):
        runs = [
            {"text": "bad", "bold": False, "italic": False, "superscript": False, "href": "javascript:alert(1)"},
            {"text": "good", "bold": False, "italic": False, "superscript": False, "href": "https://example.com"},
        ]
        html = doc2html.render_runs_html(runs)
        self.assertNotIn("javascript", html)
        self.assertIn("bad", html)
        self.assertIn('<a href="https://example.com">good</a>', html)

    def test_markdown_drops_the_link_but_keeps_the_text(self):
        runs = [
            {"text": "bad", "bold": False, "italic": False, "superscript": False, "href": "javascript:alert(1)"},
            {"text": "good", "bold": False, "italic": False, "superscript": False, "href": "https://example.com/a b(c)"},
        ]
        md = doc2html.render_runs_markdown(runs)
        self.assertNotIn("javascript", md)
        self.assertEqual(md, "bad[good](https://example.com/a%20b%28c%29)")

    def test_bare_url_with_balanced_parentheses_stays_one_link(self):
        md = doc2html.linkify_markdown("see https://en.wikipedia.org/wiki/Foo_(bar) now")
        self.assertIn("](https://en.wikipedia.org/wiki/Foo_%28bar%29)", md)
        html = doc2html.linkify_html("see https://en.wikipedia.org/wiki/Foo_(bar) now")
        self.assertIn('href="https://en.wikipedia.org/wiki/Foo_(bar)"', html)

    def test_bare_url_drops_trailing_punctuation_and_unpaired_parenthesis(self):
        html = doc2html.linkify_html("(see https://example.com/a), then https://example.com/b.")
        self.assertIn('<a href="https://example.com/a">https://example.com/a</a>),', html)
        self.assertIn('<a href="https://example.com/b">https://example.com/b</a>.', html)


class ImagePathTests(unittest.TestCase):
    def test_url_path(self):
        self.assertEqual(doc2html.url_path("images/book/image_0001_0001.png"), "images/book/image_0001_0001.png")
        self.assertEqual(doc2html.url_path("images/My #Book (1)/a.png"), "images/My%20%23Book%20%281%29/a.png")
        self.assertEqual(doc2html.url_path("images/100%/a.png"), "images/100%25/a.png")
        self.assertEqual(doc2html.url_path("images/Веб-дизайн/a.png"), "images/Веб-дизайн/a.png")
        self.assertEqual(doc2html.url_path("images/a b/x.png"), "images/a%C2%A0b/x.png")

    def test_renderers_use_the_encoded_path(self):
        block = {"type": "img", "src": 'images/My "Book" #2/a.png', "alt": ""}
        self.assertIn('src="images/My%20%22Book%22%20%232/a.png"', doc2html.render_block_html(block))
        self.assertIn("(images/My%20%22Book%22%20%232/a.png)", doc2html.render_block_markdown(block))

    def test_markdown_title_is_escaped(self):
        doc = doc2html.build_markdown_document("<noframes> *x*", ["page"])
        self.assertTrue(doc.startswith("# \\<noframes\\> \\*x\\*\n"))


class OcrLanguageTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(self.restore_defaults)

    @staticmethod
    def restore_defaults():
        doc2html.set_ocr_languages(doc2html.DEFAULT_OCR_LANGUAGES)
        doc2html._ocr_reader = None

    @staticmethod
    def fake_easyocr(reader_cls):
        return mock.patch.dict(sys.modules, {"easyocr": types.SimpleNamespace(Reader=reader_cls)})

    def test_parse_accepts_easyocr_codes(self):
        self.assertEqual(doc2html.parse_ocr_languages("es,en"), ["es", "en"])
        self.assertEqual(doc2html.parse_ocr_languages(" ES , en,es "), ["es", "en"])
        self.assertEqual(doc2html.parse_ocr_languages("ch_sim,en"), ["ch_sim", "en"])
        self.assertEqual(doc2html.parse_ocr_languages("uk"), ["uk"])

    def test_parse_rejects_things_that_are_not_codes(self):
        for raw in ("english", "e", "ru;en", "ru en", "", " , ", "ru,en,русский"):
            with self.assertRaises(doc2html.argparse.ArgumentTypeError, msg=raw):
                doc2html.parse_ocr_languages(raw)

    def test_command_line(self):
        self.assertEqual(doc2html.parse_args(["book.pdf"]).ocr_lang, ["ru", "en"])
        self.assertEqual(doc2html.parse_args(["book.pdf", "--ocr-lang", "de,en"]).ocr_lang, ["de", "en"])
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
            doc2html.parse_args(["book.pdf", "--ocr-lang", "english"])
        self.assertIn("--ocr-lang", err.getvalue())

    def test_reader_is_created_for_the_chosen_languages_and_cached(self):
        created = []

        class FakeReader:
            def __init__(self, languages, gpu):
                created.append((list(languages), gpu))

        with self.fake_easyocr(FakeReader), contextlib.redirect_stderr(io.StringIO()):
            doc2html.set_ocr_languages(["es", "en"])
            first = doc2html.get_ocr_reader(False)
            self.assertIs(doc2html.get_ocr_reader(False), first)
            doc2html.set_ocr_languages(["es", "en"])
            self.assertIs(doc2html.get_ocr_reader(False), first)
            doc2html.set_ocr_languages(["de", "en"])
            self.assertIsNot(doc2html.get_ocr_reader(False), first)
        self.assertEqual(created, [(["es", "en"], False), (["de", "en"], False)])

    def test_easyocr_rejecting_the_languages_is_a_clear_exit(self):
        class RejectingReader:
            def __init__(self, languages, gpu):
                raise ValueError("Russian is only compatible with English, try again.")

        with self.fake_easyocr(RejectingReader), contextlib.redirect_stderr(io.StringIO()):
            doc2html.set_ocr_languages(["ru", "es"])
            with self.assertRaises(SystemExit) as ctx:
                doc2html.get_ocr_reader(False)
        message = str(ctx.exception)
        self.assertIn("ru,es", message)
        self.assertIn("only compatible with English", message)
        self.assertIsNone(doc2html._ocr_reader)

    def test_resume_notice_when_the_languages_change(self):
        identity = {"input_path": "x", "input_size": 1, "input_mtime": 1.0}
        settings = {"save_images": True, "generate_alt": False, "ocr_languages": ["ru", "en"]}
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "state"
            quiet = io.StringIO()
            with contextlib.redirect_stderr(quiet):
                doc2html.load_or_init_state(state_dir, identity, settings, False, 1, 1)
                doc2html.load_or_init_state(state_dir, identity, settings, False, 1, 1)
            self.assertEqual(quiet.getvalue(), "")

            noisy = io.StringIO()
            with contextlib.redirect_stderr(noisy):
                doc2html.load_or_init_state(state_dir, identity, {**settings, "ocr_languages": ["es", "en"]}, False, 1, 1)
            self.assertIn("--ocr-lang", noisy.getvalue())

    def test_state_from_before_the_option_counts_as_the_default_languages(self):
        identity = {"input_path": "x", "input_size": 1, "input_mtime": 1.0}
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "state"
            (state_dir / "pages").mkdir(parents=True)
            legacy = {"identity": identity, "last_settings": {"save_images": True, "generate_alt": False}}
            (state_dir / "meta.json").write_text(json.dumps(legacy), encoding="utf-8")

            same = io.StringIO()
            with contextlib.redirect_stderr(same):
                doc2html.load_or_init_state(
                    state_dir, identity, {"save_images": True, "generate_alt": False, "ocr_languages": ["ru", "en"]},
                    False, 1, 1,
                )
            self.assertEqual(same.getvalue(), "")

            (state_dir / "meta.json").write_text(json.dumps(legacy), encoding="utf-8")
            other = io.StringIO()
            with contextlib.redirect_stderr(other):
                doc2html.load_or_init_state(
                    state_dir, identity, {"save_images": True, "generate_alt": False, "ocr_languages": ["es", "en"]},
                    False, 1, 1,
                )
            self.assertIn("--ocr-lang", other.getvalue())

    def test_convert_document_records_the_chosen_languages(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "tiny.pdf"
            doc = pymupdf.open()
            doc.new_page().insert_text((72, 100), "Hello there, this is a tiny page.", fontsize=11)
            doc.save(str(pdf_path))
            doc.close()

            out_dir = Path(tmp) / "out"
            doc2html.set_ocr_languages(["es", "en"])
            with contextlib.redirect_stderr(io.StringIO()):
                doc2html.convert_document(
                    pdf_path, out_dir, generate_alt=False, use_gpu=False, clean_state=False,
                )
            self.assertEqual(doc2html._ocr_languages, ["es", "en"])
            meta = json.loads((out_dir / ".tiny.doc2html-state" / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["last_settings"]["ocr_languages"], ["es", "en"])


class EndToEndTests(unittest.TestCase):
    """A tiny PDF with a link, a picture and a wrapped word, named so that its
    image folder needs URL-encoding."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        pdf_path = root / "My #Book (1).pdf"

        doc = pymupdf.open()
        page = doc.new_page()
        before = "Body text before the link "
        page.insert_text((72, 100), before, fontsize=11)
        link_x = 72 + pymupdf.get_text_length(before, fontsize=11)
        page.insert_text((link_x, 100), "CLICKME", fontsize=11, fontname="hebo")
        page.insert_link({
            "kind": pymupdf.LINK_URI,
            "from": pymupdf.Rect(link_x - 1, 88, link_x + 60, 104),
            "uri": "javascript:alert(document.domain)",
        })
        page.insert_text((72, 130), "A two- and three-dimensional inter-", fontsize=11)
        page.insert_text((72, 143), "national model of the whole thing", fontsize=11)
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 80, 80), False)
        pixmap.set_rect(pixmap.irect, (200, 30, 30))
        page.insert_image(pymupdf.Rect(72, 200, 172, 300), pixmap=pixmap)
        doc.save(str(pdf_path))
        doc.close()

        cls.outputs = {}
        for fmt in ("html", "md"):
            out_dir = root / f"out_{fmt}"
            out_file = doc2html.convert_document(
                pdf_path, out_dir, generate_alt=False, out_format=fmt, use_gpu=False,
            )
            cls.outputs[fmt] = (out_dir, out_file.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_javascript_link_is_not_emitted(self):
        for fmt, (_out_dir, text) in self.outputs.items():
            self.assertNotIn("javascript:", text, fmt)
            self.assertIn("CLICKME", text, fmt)

    def test_wrapped_word_joined_and_suspended_hyphen_kept(self):
        for fmt, (_out_dir, text) in self.outputs.items():
            self.assertIn("two- and three-dimensional international model", text, fmt)

    def test_image_url_resolves_to_the_saved_file(self):
        out_dir, html = self.outputs["html"]
        src = re.search(r'<img src="([^"]+)"', html).group(1)
        self.assertNotIn(" ", src)
        self.assertNotIn("#", src)
        self.assertTrue((out_dir / unquote(src)).is_file(), src)

        out_dir, md = self.outputs["md"]
        dest = re.search(r"!\[[^\]]*\]\(([^)]+)\)", md).group(1)
        self.assertNotIn(" ", dest)
        self.assertTrue((out_dir / unquote(dest)).is_file(), dest)


if __name__ == "__main__":
    unittest.main()
