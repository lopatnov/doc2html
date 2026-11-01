# doc2html

Converts a document — PDF, EPUB, XPS, MOBI, FB2, CBZ, and anything else
that [PyMuPDF](https://pymupdf.readthedocs.io/) can open — into readable
HTML, Markdown, or plain text (UTF-8), keeping the images.

The script is called `doc2html.py` (it used to be `pdf2html.py`; the
rename was done with `git mv`, so history is preserved). PDF remains the
primary and best-tested format — callout detection via vector graphics
and the workaround for broken font encodings rely on PDF-specific
PyMuPDF features. The pipeline is the same for other formats (it is
format-agnostic by design), but has so far been specifically tested only
on PDF and on a raster image (as a one-page document).

It addresses the typical problems of "naive" converters:

- **Encodings.** Text is extracted through PyMuPDF, normalized (`NFKC`),
  and cleaned of "garbage" Private Use Area characters. If a document's
  font ToUnicode table is broken (the text layer is unreadable in
  principle — glyphs map to control characters rather than letters), the
  script detects this automatically and recognizes that page through
  offline OCR (EasyOCR, ru+en) instead of the broken text layer. The
  result is always valid UTF-8.
- **Images.** Embedded images are rendered as-is (taking
  orientation/transformations into account — otherwise an image can come
  out flipped or mirrored), saved into an `images/<book name>/`
  subfolder (a separate one per book — several books can be converted
  into the same `-o` without the risk of image file names colliding at
  identical page numbers), and inserted at their real position in the
  text (next to the caption/paragraph they belong to) rather than dumped
  in bulk at the end of the page.
- **Running headers/footers.** Short text in the top/bottom ~9% of the
  page (page number, chapter title) is recognized as a running
  header/footer and moved into a separate small `.page-meta` line
  instead of landing in the middle of a paragraph — and if a
  chapter/section title repeats unchanged across several consecutive
  pages, it is shown only once (the page number is always shown).
- **Headings.** Section headings are recognized and marked up
  (`<h2>`/`<h3>` in HTML, `##`/`###` in Markdown) — on a clean text layer
  by font size relative to the page's body text, in OCR by line height
  relative to the median. This works regardless of which of the two
  extraction paths was used for a given page.
- **Multi-column layout (HTML).** If the geometry of the PDF blocks
  shows that a page really is split into two columns (see below), they
  are rendered side by side in HTML (`<div class="cols">`) instead of as
  one continuous flow — on mobile screens the columns stack vertically.
  Markdown/txt always stay linear text (column after column), since for
  them reading order matters more than visual resemblance to the page.
- **Tables.** Real tables (`page.find_tables()`) are rendered as
  `<table>` / a pipe table in Markdown instead of falling apart into
  unrelated one-cell paragraphs. Noisy pseudo-columns that PyMuPDF
  sometimes adds on its own are merged back into the original ones.
- **Code blocks/listings.** Recognized by a monospace font or by
  proximity to a "Лістинг/Листинг N.N" caption (Ukrainian/Russian for
  "Listing N.N") and rendered as `<pre><code>` / ` ``` ` in Markdown —
  preserving indentation and line breaks (not as one run-together line).
- **Lists.** Items with a marker (►•‣▪●○◦ or "N."/"N)") become real
  `<ul>/<ol><li>` (`- `/`N. ` in Markdown) instead of separate
  paragraphs mixed in with ordinary text.
- **Italic/bold inside a paragraph** is preserved (`<em>`/`<strong>`,
  `*`/`**` in Markdown) — only for pages with an intact text layer; on
  OCR pages this information is physically unavailable.
- **Links.** Real PDF links become working ones: external ones become
  `<a href>`/`mailto:`, internal ones become an anchor to the right page
  of the output (`#page-N`, including the table of contents/index). Bare
  URLs printed as plain text with no link object also become clickable —
  this works on any page, including OCR ones.
- **Superscripts and footnotes.** Real superscript characters
  (exponents, footnote markers) become `<sup>`. A footnote that sits at
  the edge of the page is recognized as part of the page text rather
  than a running footer if its number matches a superscript marker in
  the body.
- **Callouts/notes.** Besides a frame drawn in the PDF itself, a text
  label ("ПРИМЕЧАНИЕ"/"ВНИМАНИЕ"/"СОВЕТ" — Russian for
  "NOTE"/"WARNING"/"TIP") is also recognized without any frame — this
  works on any page, not only in OCR.
- **Multiple output formats.** `--out-format html|md|txt`. Each page is
  recognized and extracted **once**, into an intermediate representation
  (JSON in the state folder); rendering to a specific format is done
  from it separately, so switching `--out-format` on already finished
  pages does not require repeating OCR.
- **Page range, streaming output, resuming an interrupted
  conversion.** The result is reassembled after every processed page —
  the file can be opened while the script is still running. Re-running
  with the same (or different) `-o`/file parameters automatically skips
  pages that are already done.

## Installation

Requires conda (Miniconda/Anaconda) and PowerShell.

```powershell
.\install.ps1
```

If PowerShell blocks running scripts ("cannot be loaded because
running scripts is disabled"), allow them once for your user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

This creates a conda environment named **`doc2html`** and installs the
base dependencies from `requirements.txt` (PyMuPDF, Pillow, torch,
easyocr) — enough for conversion without captions (the default) or with
`--generate-alt lmstudio`. The EasyOCR model (ru+en, ~100 MB) is
downloaded once, on first use.

The offline caption backend `--generate-alt blip` requires the
`transformers` package — it is **not installed by default** (it is a
separate dependency needed only for BLIP; nothing else depends on it).
There are two ways to install it:

```powershell
# Right at install time
.\install.ps1 -WithBlip

# Or later, into the already created environment
conda run -n doc2html python -m pip install -r requirements-blip.txt
```

The BLIP model weights themselves (~1 GB) are downloaded once, on the
first real call of `--generate-alt blip`.

## Activating the environment

The script must be **dot-sourced** (run with a leading dot and a space),
otherwise the activation stays inside the script and does not affect
the current PowerShell session:

```powershell
. .\activate.ps1
```

If `conda activate` complains about "CommandNotFoundError" or asks you
to run `conda init` first, PowerShell has not been initialized for
conda yet. Run `conda init powershell` once and open a new PowerShell
session — after that `. .\activate.ps1` will work.

## Usage

```powershell
python doc2html.py input\example.pdf -o output
```

### Parameters

| Parameter                            | Default                    | Description                                                                                |
| ------------------------------------ | -------------------------- | ------------------------------------------------------------------------------------------ |
| `input_file`                         | —                          | Path to the source file: PDF, EPUB, XPS, MOBI, FB2, CBZ, image, etc. (required)            |
| `-o`, `--output-dir`                 | `output`                   | Output folder                                                                              |
| `--out-format {html,md,txt}`         | `html`                     | Output format                                                                              |
| `--save-images` / `--no-save-images` | on                         | Save images to `<output>/images/<book name>`                                               |
| `--generate-alt {off,blip,lmstudio}` | `off`                      | Image descriptions: don't generate / offline BLIP / local LM Studio server                 |
| `--start-page N`                     | 1                          | First page to convert (1-indexed)                                                          |
| `--end-page N`                       | last                       | Last page, inclusive                                                                       |
| `--restart`                          | off                        | Ignore saved state (within the requested range) and start over                             |
| `--clean-state` / `--no-clean-state` | **on**                     | Delete the state folder once the whole document is fully done                              |
| `--gpu` / `--no-gpu`                 | auto-detect                | Use the GPU (CUDA) for OCR/BLIP                                                            |
| `--lmstudio-url`                     | `http://localhost:1234/v1` | LM Studio server address                                                                   |
| `--lmstudio-model`                   | `lfm2.5-vl-450m`           | Vision model name in LM Studio (used with `--generate-alt lmstudio`)                       |

For a one-off manual check of a single specific page, use a separate
scratch folder via `-o` (not the real `output/`!), the same N in
`--start-page`/`--end-page`, and flags that skip saved state and image
description generation for speed:

```powershell
python doc2html.py input\example.pdf -o scratch\page-check --start-page 42 --end-page 42 --restart --no-clean-state --generate-alt off
```

### How the image-saving flags behave

By default (`--save-images`, `--generate-alt off`): the image is saved
(`<img>` in HTML, `![]()` in Markdown), but the alt/caption is empty —
the fastest mode; no model is launched or downloaded.

- `--save-images` (on) + `--generate-alt blip|lmstudio`: the image is
  saved with a meaningful description.
- `--save-images` (on) + `--generate-alt off` (default): the image is
  saved but has no description.
- `--no-save-images` + `--generate-alt blip|lmstudio`: the image is not
  saved; `[Изображение: <description>]` is inserted into the text in
  its place (the program's output placeholder is in Russian — "Image").
- `--no-save-images` + `--generate-alt off`: the image is not saved;
  the placeholder `[Изображение]` ("Image") is inserted instead.

In **txt**, images are always described in text (`[Изображение: ...]`
or `[Изображение]`); the files are still saved to disk if
`--save-images` is on — there is simply nowhere to embed them in plain
text.

### Page range, multiple formats, streaming output, and resuming

```powershell
# Only the first 50 pages
python doc2html.py input\book.pdf -o output --end-page 50

# Continue from page 51 - the finished pages 1-50 are not touched, the
# new ones are just appended
python doc2html.py input\book.pdf -o output --start-page 51

# The same range, but as Markdown - the pages are already extracted, so
# there is no repeated OCR and it finishes in seconds
python doc2html.py input\book.pdf -o output --end-page 50 --out-format md
```

- After every processed page the resulting file is rewritten in full
  (atomically) — you can open and refresh it while the script is
  running.
- If the script is interrupted (Ctrl+C, a crash, a reboot) and started
  again with the same `-o` folder, it finds the already finished pages
  by itself and continues from where it stopped — there is no need to
  convert again from scratch.
- The page range in a repeated run may differ from the first one — this
  way you can feed a document to the script in pieces by hand.
- `--out-format` can be changed freely between runs — rendering comes
  from the already extracted pages instead of recognizing the document
  again. The output file name depends on the format
  (`book.html`/`book.md`/`book.txt`), so different formats don't
  overwrite each other.
- If you change `--save-images`/`--generate-alt` between runs, the
  script will not silently carry on with the new settings: it prints a
  warning (these flags change the content of pages not yet processed)
  and asks for either `--restart` or a new `-o` folder if you need to
  recompute already finished pages with the new settings.
- `--restart` recomputes **only the requested range** — pages outside
  `--start-page`/`--end-page` are not touched.

#### The state folder

Intermediate page fragments live in a hidden folder next to the result:
`output\.<file name>.doc2html-state\pages\0001.json`, `0002.json`, ...
Each file is the page's extracted content in a format independent of
`--out-format` (a list of paragraphs/images/running headers/callouts);
the images themselves are not duplicated there — they exist only in
`output\images\<book name>\`, and the fragment merely references the
file by name.

By default the state folder is **deleted automatically**
(`--clean-state`), but only once the **whole** document (not just the
range requested in this run) is fully done — if anything outside the
range is not yet converted, the script prints a message about it and
leaves the folder alone. To disable auto-deletion, use
`--no-clean-state` (for example, if you plan to re-render the pages in
a different `--out-format` later and don't want to lose the cache
prematurely). You can delete the state folder manually at any time —
you won't lose images, only the ability to resume/finish converting
without recomputation.

### GPU and LM Studio

`--gpu`/`--no-gpu` enables CUDA acceleration for EasyOCR and BLIP via
`torch.cuda` — it works only with NVIDIA cards. **If the GPU is not
NVIDIA (for example, Intel Arc/Iris Xe with OpenVINO), this flag will
not help** — a CPU build of PyTorch won't see CUDA regardless of the
flag.

In that case, you can use a local **LM Studio** server with a vision
model for image captions — many LM Studio runtimes can accelerate on
GPUs from different vendors, not only NVIDIA:

1. In LM Studio, load a vision model (one that accepts an image as
   input) and enable the Local Server (`http://localhost:1234` by
   default). By default the script expects
   **`LiquidAI/LFM2.5-VL-450M-GGUF`** (model id in LM Studio:
   `lfm2.5-vl-450m`) — a compact (450M) and fast vision model that is
   worth keeping loaded specifically for this task. You can use any
   other one (`Qwen2-VL`, `InternVL`, etc.) via `--lmstudio-model`.
2. Run the conversion with this backend:

```powershell
python doc2html.py input\book.pdf -o output --generate-alt lmstudio
```

If the server is unavailable or the model is not found, the script
stops immediately with a clear error (listing the models available in
LM Studio), without wasting time converting text for nothing.

**OCR is deliberately not moved to an LLM/LM Studio.** For verbatim
recognition of a document's text, offline OCR (EasyOCR) is the right
tool: a language model risks silently paraphrasing or "filling in"
text, which is unacceptable for digitization. LM Studio is used only
where an approximate, creative description is appropriate and useful —
for image captions.

### Example

```powershell
. .\activate.ps1
python doc2html.py "input\Jakob Nielsen's Web Design Book.pdf" -o output
```

Result: `output\Jakob Nielsen's Web Design Book.html` and, if image saving
is on, `output\images\Jakob Nielsen's Web Design Book\*.png`.

## How the layout is reconstructed in OCR

When a document's text layer is broken, the page text is recognized
entirely through OCR (EasyOCR) rather than taken from the source file.
It then goes through a custom layout reconstruction (not EasyOCR's
built-in grouping), written specifically for the typical problems of
such recognition:

- **Tables of contents and indexes.** The text detector often sees an
  entry like "Title ....... 27" as TWO separate blocks (a sparse dotted
  leader is not recognized as text) — but both blocks occupy almost the
  same line height. Lines are merged by the percentage of vertical
  overlap of the boxes (not by X position), so the title and the number
  are correctly joined into one "Title 27" entry, no matter how wide the
  dotted line between them is.
- **Hyphenated word wraps** ("exam-\nple" → "example") are fixed
  automatically.
- **Sidebars/callouts in the margin** (framed text beside the main
  column, at the same page height) are found not by the position of the
  OCR text (in some books the side callouts are deliberately aligned to
  the same line grid as the main column, so they can't be told apart by
  height), but by the document's own vector graphics — the callout's
  frame/fill. Such an area is painted over before the main OCR pass (as
  images are) and recognized separately, as a whole, and the result is
  laid out as a separate block (`<aside class="callout">` in HTML, a `>`
  quote in Markdown, an indent in txt) after the main text of the page —
  instead of being split up and wedged line by line into the middle of
  the main column's sentences. If a particular document's layout does
  not visually set callouts off at all (neither a frame nor a fill),
  this mechanism won't find them, and they can still end up in the main
  flow.
- Images are painted over before OCR so that interface text/captions
  inside screenshots don't get into the main text as "noise"; a frame
  around the image itself (if any) is not mistaken for a separate
  callout.
- **Running headers/footers** (short text in the top/bottom ~9% of the
  page) are taken out of the general flow into `.page-meta`; if a margin
  paragraph has actually glued a page number and a section title
  together into one line ("8 Contents"), they are separated first, so
  that deduplication of repeats (see above) works regardless of which
  page number happened to be next to the title this time.
- Occasionally (for example, ISBN in the imprint) a short text with a
  high share of digits may still be mistakenly given the `.ocr-numbers`
  class (pale gray, italic) — the heuristic goes only by the share of
  digits in the text, not by meaning.
- **Genuine two columns** (not a callout, but simply text set in two
  columns on one page) are detected from the geometry of the PDF
  blocks: if the blocks cleanly group into left and right groups of
  equal height (not merely with different indents — real parallelism is
  checked separately, otherwise a title page/imprint with paragraphs of
  differing widths would be mistaken for two columns), a real empty gap
  (the "gutter") is searched for between them, and OCR lines whose
  internal break falls exactly into that gap are split along it — even
  if the gap itself is narrow on the page (a few points). This is the
  same gap used for the side-by-side rendering in HTML (see above);
  nested sub-items inside one column (for example, sub-topics of one
  table-of-contents entry) get a visual indent (`margin-left`)
  proportional to their own offset from the start of the column — this
  is not a real `<ul>`/`<li>`, just a visual approximation of nesting,
  which is usually enough to keep a list from turning into a solid wall
  of text.
- Like any heuristic, this is not perfect: occasionally a short insert
  (for example, a parenthetical reference to a figure on the adjacent
  page) that sits close in height to both columns at once on the source
  page may end up slightly out of place in the text — but, unlike the
  previous behavior, words inside a sentence are no longer broken in
  the middle.

## Notes

- Image descriptions via BLIP are generated in English; via LM Studio —
  in whatever language the loaded model chooses (this depends on the
  prompt and the model itself).
- The first run with `--generate-alt blip|lmstudio`/OCR will be
  noticeably slower because of downloading the model weights; on
  subsequent runs the models are taken from the local cache.
- For large documents (hundreds of pages), OCR and caption generation
  are the slowest stages on CPU; use `--generate-alt off` (the default)
  if you don't need descriptions, or `--generate-alt lmstudio` with a
  GPU.
