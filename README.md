# KIFU-extract

Converts the go diagrams of a scanned problem book (built for *Get Strong at
Attacking*) into one SGF file per problem.

Each SGF holds the problem position (`AB`/`AW`, side to play, marked stones as
`TR`, the problem text as a comment) and one variation per answer diagram
("Correct Answer", "Failure", "Variation 2", ...) with the numbered stones as
the move sequence and the book's explanation as the comment.

## Requirements

- Python 3.9+ and `pip install -r requirements.txt`
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) on the `PATH`
  (on Windows the default `C:\Program Files\Tesseract-OCR` install is found
  automatically; otherwise set `TESSERACT_CMD` to the executable)

## Usage

```bash
python pdf_to_sgf.py Get_Strong_at_Attacking.pdf [output_dir] [first_page-last_page]
```

Output goes to `output_sgf/` by default. The whole book takes a couple of
minutes. The script prints a line for anything it could not read cleanly, e.g.
a move the diagram does not show, or a diagram whose caption names the wrong
problem.

## How it works

1. Each page is rendered at 360 dpi (the scan's native resolution). Board
   diagrams are found from their long grid lines, deskewed, and an evenly
   spaced 19-line lattice is fitted to them. Partial diagrams are placed on the
   19x19 board by looking at which sides are real board edges.
2. Every intersection is classified as empty, black (solid disc) or white
   (circle outline). Numbers printed on stones are read with Tesseract, and
   triangle markers are recognised by shape.
3. Captions come from the PDF's text layer: "Problem N. Black to play" above a
   problem diagram, "Problem N: Correct Answer" etc. below an answer diagram.
4. Each answer diagram is compared with its problem position: numbered stones
   become moves (colours follow the numbering parity), unnumbered new stones
   become setup stones, and notes such as "8: connects at 1" are applied.
