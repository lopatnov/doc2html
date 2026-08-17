"""Shared safe-HTTP helpers for the tools/*.py scripts that talk to
Gutendex/Project Gutenberg (tools/fetch_random_doc.py and
tools/find_book.py). Not meant to be run directly.

Both scripts fetch a download URL out of a third-party API response and
then follow it - the allowlisted-redirect handler, capped reads, and
atomic writes here exist so that data is treated as untrusted (SSRF via
a compromised/malicious API response, or a runner/disk exhausted by an
unbounded body) rather than as a URL safe to hand straight to urlopen().
"""
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "doc2html-gutenberg-tools/1.0 (+https://github.com/lopatnov/doc2html)"
MAX_RESPONSE_BYTES = 200 * 1024 * 1024  # generous cap for even a large scanned-page epub/pdf

# Ordered by how well doc2html.py (via PyMuPDF) handles them.
PREFERRED_MIME_PREFIXES = ["application/epub+zip", "application/pdf", "text/html"]
EXTENSION_BY_MIME_PREFIX = {
    "application/epub+zip": ".epub",
    "application/pdf": ".pdf",
    "text/html": ".html",
}


def make_allowlisted_redirect_handler(allowed_hosts):
    """A urllib redirect handler that re-validates *every* hop against
    allowed_hosts (https + exact hostname) instead of trusting urlopen()'s
    default behavior of blindly following redirects. An earlier version of
    this refused all redirects outright, which broke the normal case too:
    Gutendex's REST API (Django-style) 301-redirects /books/{id} ->
    /books/{id}/ on every request, same host - confirmed by running the
    real workflow, where all attempts failed with "HTTP 301" before this
    fix existed. Allowlisting the host (rather than just accepting "same
    as request") still blocks a same-host response from redirecting
    off-host to an attacker-controlled server (SSRF).
    """

    class _AllowlistedRedirectHandler(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            parsed = urllib.parse.urlparse(newurl)
            # Same three conditions as is_allowed_download_url - the redirect
            # hop is exactly the untrusted path this handler exists to
            # constrain, so it must not be weaker than the direct-URL check.
            if parsed.scheme == "https" and parsed.hostname in allowed_hosts and parsed.port is None:
                return super().redirect_request(req, fp, code, msg, headers, newurl)
            raise urllib.error.HTTPError(newurl, code, f"refusing to follow redirect to {newurl!r}", headers, fp)

    return _AllowlistedRedirectHandler


def build_opener(allowed_hosts):
    return urllib.request.build_opener(make_allowlisted_redirect_handler(allowed_hosts))


def read_capped(resp, max_bytes=MAX_RESPONSE_BYTES):
    """Read at most max_bytes+1 from resp - the +1 lets the caller tell
    "exactly at the cap" apart from "went over" without reading unbounded
    data first. Guards against a malicious/broken response streaming an
    unbounded body at us (disk/memory exhaustion)."""
    chunks = []
    total = 0
    while total <= max_bytes:
        chunk = resp.read(min(1024 * 1024, max_bytes + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    if total > max_bytes:
        raise ValueError(f"response exceeded {max_bytes} byte cap")
    return b"".join(chunks)


def fetch_json(opener, url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(read_capped(resp))


def is_allowed_download_url(url, allowed_hosts):
    parsed = urllib.parse.urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.hostname in allowed_hosts
        and parsed.port is None  # reject an explicit non-default port, e.g. gutenberg.org:8081
    )


def normalize_download_url(url, allowed_hosts):
    """Upgrade http -> https for a recognized host (Gutendex sometimes
    hands back http:// links). Never downgrades; anything else is
    returned unchanged and left for is_allowed_download_url to reject.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "http" and parsed.hostname in allowed_hosts:
        return urllib.parse.urlunparse(parsed._replace(scheme="https"))
    return url


def atomic_download(opener, url, dest, timeout=60):
    """Write to a temp file and rename into place only once the full body
    is on disk, so a failed attempt (network error, disk full mid-write)
    never leaves a partial/empty file at dest for a caller to pick up and
    use/publish by accident.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with opener.open(req, timeout=timeout) as resp:
        data = read_capped(resp)
    # A fixed ".{name}.part" temp name would collide if two downloads to the
    # same dest ever ran concurrently - mkstemp gives each call its own file.
    fd, temp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".part", dir=str(dest.parent))
    os.close(fd)
    temp_dest = Path(temp_name)
    try:
        with open(temp_dest, "wb") as f:
            f.write(data)
        temp_dest.replace(dest)
    finally:
        temp_dest.unlink(missing_ok=True)


def pick_format(formats, preferred_prefixes=None):
    for mime_prefix in preferred_prefixes if preferred_prefixes is not None else PREFERRED_MIME_PREFIXES:
        for key, url in formats.items():
            if key.startswith(mime_prefix):
                return mime_prefix, key, url
    return None, None, None
