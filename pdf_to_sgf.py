"""Convert the go diagrams of a scanned problem book (e.g. "Get Strong at
Attacking") into one SGF file per problem.

Pipeline per page:
  1. render the page and find every 19x19 board diagram,
  2. locate the 19 grid lines of each board and read every intersection
     (empty / black / white, plus the number or triangle printed on a stone),
  3. pair each board with its caption from the PDF text layer
     ("Problem 12. Black to play" above problem boards,
      "Problem 12: Correct Answer" below answer boards).
Then every answer diagram is diffed against its problem diagram: numbered
stones become the move sequence, unnumbered extra stones become setup stones.
"""
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import fitz
import numpy as np
import pytesseract

DPI = 360  # native resolution of the scanned foreground layer
SCALE = DPI / 72.0
INK = 60   # darkness above the paper background that counts as ink

if os.name == "nt" and not os.environ.get("TESSERACT_CMD"):
    default_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default_tesseract):
        pytesseract.pytesseract.tesseract_cmd = default_tesseract
if os.environ.get("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]

PROBLEM_RE = re.compile(r"^Problem\s+(\d+)\s*[.,:]?\s*(Black|White)\s+to\s+play", re.IGNORECASE)
# Some captions under answer diagrams omit the problem number ("Failure 2").
BARE_RE = re.compile(r"^(Correct Answer|Failure|Variation|Continuation|Wrong Answer)(\s+\d+)?$")
ANSWER_RE = re.compile(r"^Problem\s+(\d+)\s*(?:[:;]\s*|\.\s*(?=(?:Correct|Failure|Variation|Continuation|Wrong)))([A-Z].*)$")


# ---------------------------------------------------------------------------
# Board reading
# ---------------------------------------------------------------------------

def darkness(gray):
    """Ink darkness relative to the local paper colour (0 = paper)."""
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    paper = cv2.dilate(small, np.ones((25, 25), np.uint8))  # brightest nearby pixel
    paper = cv2.GaussianBlur(paper, (0, 0), 5)
    paper = cv2.resize(paper, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(paper.astype(np.int16) - gray.astype(np.int16), 0, 255).astype(np.uint8)


def line_masks(ink, length, bridge=1):
    """Long horizontal / vertical runs of ink. `bridge` closes gaps of up to
    that many pixels along the line first (faint scans break thin lines up)."""
    h = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((1, bridge), np.uint8)) if bridge > 1 else ink
    v = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((bridge, 1), np.uint8)) if bridge > 1 else ink
    h = cv2.morphologyEx(h, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    v = cv2.morphologyEx(v, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8))
    return h, v


def projection_peaks(proj, threshold):
    idx = np.where(proj > threshold)[0]
    if not len(idx):
        return []
    return [g.mean() for g in np.split(idx, np.where(np.diff(idx) > 3)[0] + 1)]


def fit_lattice(proj, step, lo, hi):
    """Positions of the evenly spaced grid lines along one axis of a board.
    `proj` is the fraction of each pixel row/column covered by line pixels;
    only lines within lo..hi (the board's own extent) count."""
    best = None
    for offset in np.arange(0, step, 0.5):
        pos = np.arange(offset, len(proj) - 1, step)
        score = np.interp(pos, np.arange(len(proj)), proj).sum()
        if best is None or score > best[0]:
            best = (score, pos)
    pos = best[1]
    present = [p for p in pos if lo - 8 <= p <= hi + 8 and proj[max(0, int(p) - 2):int(p) + 3].max() > 0.2]
    if len(present) < 2:
        return None
    pos = np.arange(present[0], present[-1] + step / 2, step)
    # Refine offset and spacing on the measured lines. The printed grid is
    # uniform, so fit one straight lattice: snapping lines one by one can lock
    # onto a row of stones instead.
    peaks = np.array(projection_peaks(proj, 0.2))
    k = np.arange(len(pos))
    for _ in range(2):
        if not len(peaks):
            break
        nearest = peaks[np.abs(peaks[None, :] - pos[:, None]).argmin(axis=1)]
        ok = np.abs(nearest - pos) < step * 0.25
        if ok.sum() < 3:
            break
        slope, icpt = np.polyfit(k[ok], nearest[ok], 1)
        fitted = icpt + slope * k
        # drop outliers (e.g. a stone row next to a faint line) and refit
        ok &= np.abs(nearest - fitted) < 3
        if ok.sum() >= 3:
            slope, icpt = np.polyfit(k[ok], nearest[ok], 1)
        pos = icpt + slope * k
    return pos


def has_stubs(lines_mask, at, direction, step):
    """True if the crossing lines run on past the outermost line at `at`
    (the diagram is cut off there) rather than stopping (a board edge)."""
    lo, hi = (at - 0.25 * step, at - 0.08 * step) if direction < 0 else (at + 0.08 * step, at + 0.25 * step)
    lo, hi = int(max(0, lo)), int(min(lines_mask.shape[0], hi))
    if hi <= lo:
        return False
    band = lines_mask[lo:hi]
    return (band > 0).all(axis=0).sum() > 3 * 3  # several crossing lines, a few px wide each


def deskewed_crop(dark, contour):
    """Crop a board out of the page, rotated so that its grid is axis-aligned."""
    x, y, w, h = cv2.boundingRect(contour)
    m = int(0.05 * max(w, h))
    x0, y0 = max(0, x - m), max(0, y - m)
    crop = dark[y0:y + h + m, x0:x + w + m]
    offset = (x - x0, y - y0)
    center = (crop.shape[1] / 2, crop.shape[0] / 2)

    def rotated(angle):
        if angle == 0:
            return crop
        rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(crop, rot, (crop.shape[1], crop.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)

    def sharpness(img):  # straight grid lines give a peaky row/column profile
        ink = (img > INK).astype(np.float32)
        return (ink.sum(axis=1) ** 2).sum() + (ink.sum(axis=0) ** 2).sum()

    best = max(np.arange(-1.5, 1.51, 0.25), key=lambda a: sharpness(rotated(a)))
    return rotated(best), offset


def board_grid(img, offset, w, hgt, bridge):
    """Fit the grid of one (deskewed) board crop, or return None. The board
    occupies w x hgt pixels at `offset` in the crop (which has some margin)."""
    ink = (img > INK).astype(np.uint8) * 255
    h, v = line_masks(ink, int(DPI * 0.2), bridge)
    # Thicken the lines across their direction: the scan is not perfectly
    # straight, so one grid line wanders over a few pixel rows.
    hb = cv2.dilate(h, np.ones((7, 1), np.uint8))
    vb = cv2.dilate(v, np.ones((1, 7), np.uint8))
    hproj, vproj = (hb > 0).sum(axis=1) / w, (vb > 0).sum(axis=0) / hgt

    diffs = []
    for proj in (hproj, vproj):
        diffs += list(np.diff(projection_peaks(proj, 0.3)))
    diffs = [d for d in diffs if DPI * 0.08 < d < DPI * 0.2]
    if len(diffs) < 4:
        return None
    step = float(np.median(diffs))
    xs = fit_lattice(vproj, step, offset[0], offset[0] + w)
    ys = fit_lattice(hproj, step, offset[1], offset[1] + hgt)
    if xs is None or ys is None or not (5 <= len(xs) <= 19 and 5 <= len(ys) <= 19):
        return None

    # Which part of the 19x19 board does a partial diagram show?
    col0 = row0 = 0
    if len(xs) < 19:
        if has_stubs(hb.T, xs[0], -1, step) and not has_stubs(hb.T, xs[-1], 1, step):
            col0 = 19 - len(xs)
    if len(ys) < 19:
        if has_stubs(vb, ys[0], -1, step) and not has_stubs(vb, ys[-1], 1, step):
            row0 = 19 - len(ys)
    return {"img": img, "xs": xs, "ys": ys, "step": step, "col0": col0, "row0": row0}


def find_boards(dark):
    ink = (dark > INK).astype(np.uint8) * 255
    h, v = line_masks(ink, int(DPI * 0.2))
    grid = cv2.dilate(h | v, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boards = []
    for c in contours:
        x, y, w, hgt = cv2.boundingRect(c)
        if w < DPI * 0.6 or hgt < DPI * 0.6:  # narrow strips of a board are used too
            continue
        img, offset = deskewed_crop(dark, c)
        # Some diagrams are printed with broken (dotted) grid lines; bridging the
        # gaps only as a fallback keeps nearby caption text from turning into lines.
        board = board_grid(img, offset, w, hgt, 1) or board_grid(img, offset, w, hgt, 9)
        if board is None:
            if w > DPI and hgt > DPI:
                print(f"  warning: no go board grid found in board-like area at {x},{y}; skipped")
            continue
        board["box"] = (x, y, x + w, y + hgt)
        boards.append(board)
    return boards


def label_glyphs(label, radius):
    """Split a label mask into glyphs (lists of component boxes, left to right),
    dropping specks and leftover slivers of the stone's outline."""
    n, cc, stats, _ = cv2.connectedComponentsWithStats(label, connectivity=8)
    c = (label.shape[1] - 1) / 2.0
    yy, xx = np.mgrid[:label.shape[0], :label.shape[1]]
    reach = np.hypot(xx - c, yy - c)
    boxes = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 10 or reach[cc == i].min() > 0.55 * radius:
            continue
        x, y, w, h = stats[i, :4]
        boxes.append([x, y, x + w, y + h, [i]])
    boxes.sort()
    glyphs = []
    for b in boxes:  # merge pieces of one broken digit (overlapping columns)
        if glyphs and min(glyphs[-1][2], b[2]) - max(glyphs[-1][0], b[0]) > 0.5 * min(b[2] - b[0], glyphs[-1][2] - glyphs[-1][0]):
            g = glyphs[-1]
            glyphs[-1] = [min(g[0], b[0]), min(g[1], b[1]), max(g[2], b[2]), max(g[3], b[3]), g[4] + b[4]]
        else:
            glyphs.append(b)
    return cc, glyphs


def is_triangle(cc, glyphs):
    if len(glyphs) != 1:
        return False
    x0, y0, x1, y1, ids = glyphs[0]
    h = y1 - y0
    pts = cv2.findNonZero(np.isin(cc, ids).astype(np.uint8))
    hull = cv2.convexHull(pts)
    corners = cv2.approxPolyDP(hull, 0.1 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    if len(corners) != 3 or (x1 - x0) < 0.9 * h:
        return False
    # The marker is an outline triangle with its apex at the top (unlike a "7").
    apex_up = sorted(corners[:, 1] - y0)
    hollow = len(pts) < 0.9 * cv2.contourArea(hull)
    return apex_up[0] < 0.3 * h and apex_up[1] > 0.7 * h and hollow


# Tesseract is fussy about glyph size: it rejects some perfectly clean digits
# at one height and reads them at another, so labels are tried at several.
# (glyph height, page segmentation mode, blur)
OCR_SETTINGS = [(56, 8, 0), (32, 7, 0), (40, 7, 1.0), (56, 13, 0)]


def tesseract_digits(img, psm, height=56, blur=0):
    f = float(height) / max(1, img.shape[0])
    img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    else:
        img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]
    img = cv2.copyMakeBorder(img, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255)
    text = pytesseract.image_to_string(img, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789")
    return text.strip()


def ocr_digits(cc, glyphs):
    """OCR the glyphs of a stone label. Returns int or None."""
    if not glyphs or len(glyphs) > 3:
        return None
    ids = [i for g in glyphs for i in g[4]]
    x0, y0 = min(g[0] for g in glyphs), min(g[1] for g in glyphs)
    x1, y1 = max(g[2] for g in glyphs), max(g[3] for g in glyphs)
    img = np.where(np.isin(cc, ids), 0, 255).astype(np.uint8)
    for height, psm, blur in OCR_SETTINGS:
        whole = tesseract_digits(img[y0:y1, x0:x1], psm, height, blur)
        if whole.isdigit() and len(whole) == len(glyphs):
            return int(whole)
    # Read glyph by glyph: two-digit labels are cramped and tesseract tends to
    # drop one of the digits.
    digits = ""
    for g in glyphs:
        d = tesseract_digits(img[g[1]:g[3], g[0]:g[2]], 10)
        if len(d) != 1:
            return None
        digits += d
    return int(digits)


CIRCLE = np.linspace(0, 2 * np.pi, 64, endpoint=False)


def stone_outline(ink, cx, cy, step, color):
    """Locate a stone's printed circle near (cx, cy): returns (cx, cy, radius).
    The radius is fixed: letting it vary makes touching neighbours win."""
    r = (0.48 if color == "B" else 0.46) * step

    def on(x, y, rr):  # share of ink on a circle of radius rr around (x, y)
        return ink[np.round(y + rr * np.sin(CIRCLE)).astype(int), np.round(x + rr * np.cos(CIRCLE)).astype(int)].mean()

    def score(x, y):
        if color == "B":  # solid disc: ink inside the edge, paper outside
            return on(x, y, r - 2) - on(x, y, r + 3)
        return on(x, y, r) - 0.5 * (on(x, y, r - 3) + on(x, y, r + 3))  # thin ring

    x, y = max(((cx + dx, cy + dy) for dy in range(-5, 6) for dx in range(-5, 6)), key=lambda p: score(*p))
    return x, y, r


def read_board(board):
    """Read every intersection. Returns {(col, row): {color, num, marked, labelled}}."""
    dark, xs, ys, step = board["img"], board["xs"], board["ys"], board["step"]
    R = int(round(step * 0.7))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    dist = np.hypot(xx, yy)
    disc = dist < 0.42 * step
    # Inside a black stone's rim there is solid ink, apart from the printed
    # number: tall single digits reach into the top and bottom of this ring,
    # wide two-digit numbers into its sides, so test both separately.
    annulus = (dist > 0.30 * step) & (dist < 0.42 * step)
    arcs = (annulus & (np.abs(yy) > 0.7 * dist), annulus & (np.abs(xx) > 0.7 * dist))
    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)

    def ring(lo, hi):
        radii = np.linspace(lo, hi, 9) * step
        return (np.round(np.outer(np.sin(angles), radii)).astype(int),
                np.round(np.outer(np.cos(angles), radii)).astype(int))

    def ring_fraction(x, y, offsets):  # share of directions crossing an outline
        return ink[y + offsets[0], x + offsets[1]].any(axis=1).mean()

    # A wide band finds outlines even when the stone sits a few pixels off
    # the intersection; the narrow one confirms it around the located stone.
    wide_ring, narrow_ring = ring(0.30, 0.62), ring(0.38, 0.54)

    ink = cv2.copyMakeBorder(dark, 2 * R, 2 * R, 2 * R, 2 * R, cv2.BORDER_CONSTANT, value=0) > INK
    stones = {}
    for j in range(len(ys)):
        for i in range(len(xs)):
            cx, cy = int(round(xs[i])) + 2 * R, int(round(ys[j])) + 2 * R
            patch = ink[cy - R:cy + R + 1, cx - R:cx + R + 1]
            if patch[disc].mean() < 0.5 and ring_fraction(cx, cy, wide_ring) < 0.6:
                continue
            # Stones are not always printed exactly on the intersection, so
            # decide the colour only once the stone itself has been located.
            sx, sy, r = stone_outline(ink, cx, cy, step, "B")
            patch = ink[sy - R:sy + R + 1, sx - R:sx + R + 1]
            if max(patch[a].mean() for a in arcs) > 0.7:
                color = "B"
            else:
                color = "W"
                sx, sy, r = stone_outline(ink, cx, cy, step, "W")
                # Scanned outlines can have gaps, while an empty point boxed in
                # by four stones only reaches about 0.35 here.
                if ring_fraction(sx, sy, narrow_ring) < 0.5:
                    continue
                patch = ink[sy - R:sy + R + 1, sx - R:sx + R + 1]
            stone = {"color": color, "num": None, "marked": False, "labelled": False}

            # Label pixels: paper inside a black stone, ink inside a white one.
            label = (~patch if color == "B" else patch) & (dist < r - 3)
            label = label.astype(np.uint8) * 255
            cc, glyphs = label_glyphs(label, r)
            area = sum((np.isin(cc, g[4])).sum() for g in glyphs)
            tall = any(g[3] - g[1] > 0.6 * r for g in glyphs)  # not just a speck
            if area > 0.03 * np.pi * r * r and tall:
                if is_triangle(cc, glyphs):
                    stone["marked"] = True
                else:
                    stone["labelled"] = True
                    stone["num"] = ocr_digits(cc, glyphs)
            stones[(i + board["col0"], j + board["row0"])] = stone
    return stones


# ---------------------------------------------------------------------------
# Page layout: boards + captions
# ---------------------------------------------------------------------------

def clean_text(s):
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" \ufffd ", " \u2014 ").replace("\ufffd", "'")
    return s


def page_words(page):
    return [(x0, y0, x1, y1, clean_text(w)) for x0, y0, x1, y1, w, *_ in page.get_text("words")]


def lines_in(words, x0, x1, y0, y1):
    """Rebuild text lines from the words whose centre lies in a rectangle.
    (The OCR layer often joins lines of neighbouring columns into one.)"""
    sel = sorted((w for w in words if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1),
                 key=lambda w: (w[1] + w[3]) / 2)
    rows = []
    for w in sel:
        yc = (w[1] + w[3]) / 2
        if rows and yc - rows[-1][0] < 4:
            rows[-1][1].append(w)
        else:
            rows.append((yc, [w]))
    lines = []
    for _, ws in rows:
        ws.sort()
        lines.append({"text": clean_text(" ".join(w[4] for w in ws)),
                      "bbox": (min(w[0] for w in ws), min(w[1] for w in ws),
                               max(w[2] for w in ws), max(w[3] for w in ws))})
    return lines


def is_caption(text):
    return bool(PROBLEM_RE.match(text) or ANSWER_RE.match(text) or BARE_RE.match(text))


def paragraph(words, x0, x1, top, max_gap=8):
    """The paragraph of text starting just below `top` in the column x0..x1."""
    out, bottom = [], top
    for l in lines_in(words, x0, x1, top, float("inf")):
        if l["bbox"][1] - bottom > max_gap or is_caption(l["text"]):
            break
        out.append(l["text"])
        bottom = l["bbox"][3]
    return re.sub(r"(\w)- (\w)", r"\1\2", " ".join(out))


def norm(text):
    return re.sub(r"\W", "", text).lower()


def text_columns(boxes, page_width):
    """Horizontal extent of the text column belonging to each board: the page
    half it is in, or, when three diagrams share a row, the space up to the
    midpoints between it and its neighbours."""
    cols = []
    for b in boxes:
        row = sorted((o for o in boxes if abs(o[1] - b[1]) < 60), key=lambda o: o[0])
        if len(row) >= 3:
            k = row.index(b)
            left = (row[k - 1][2] + b[0]) / 2 if k > 0 else 0
            right = (b[2] + row[k + 1][0]) / 2 if k + 1 < len(row) else page_width
        else:
            mid, quarter = page_width / 2, (b[2] - b[0]) / 4
            if b[0] < mid - quarter and b[2] > mid + quarter:  # centred on the page
                left, right = 0, page_width
            else:
                left, right = (0, mid) if (b[0] + b[2]) / 2 < mid else (mid, page_width)
        cols.append((left, right))
    return cols


def ocr_caption(gray, x0, y0, x1, y1):
    """Fallback when the PDF text layer has no caption near a board."""
    h, w = gray.shape
    crop = gray[max(0, int(y0 * SCALE)):min(h, int(y1 * SCALE)), max(0, int(x0 * SCALE)):min(w, int(x1 * SCALE))]
    if crop.size == 0:
        return None
    text = pytesseract.image_to_string(crop, config="--psm 6")
    for raw in text.splitlines():
        t = clean_text(raw)
        if is_caption(t):
            return t
    return None


def analyse_page(args):
    try:
        return _analyse_page(*args)
    except Exception as e:  # keep going; report which page failed
        print(f"  page {args[1] + 1}: failed: {e!r}")
        return args[1], []


def _analyse_page(pdf_path, page_num):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    words = page_words(page)
    if not any(w[4].lower() == "problem" for w in words):
        return page_num, []
    width = page.rect.width

    pix = page.get_pixmap(dpi=DPI, colorspace=fitz.csGRAY)
    gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
    dark = darkness(gray)

    boards = find_boards(dark)
    boxes = [tuple(v / SCALE for v in b["box"]) for b in boards]
    results = []
    for board, (bx0, by0, bx1, by1), (cx0, cx1) in zip(boards, boxes, text_columns(boxes, width)):
        # Problem diagrams have their caption above, answer diagrams below.
        below = [l for l in lines_in(words, cx0, cx1, by1 - 5, by1 + 35)
                 if ANSWER_RE.match(l["text"]) or BARE_RE.match(l["text"])]
        above = [l for l in lines_in(words, cx0, cx1, by0 - 75, by0 + 5) if PROBLEM_RE.match(l["text"])]
        desc = notes = ""
        if below:
            text = below[0]["text"]
            desc = paragraph(words, cx0, cx1, below[0]["bbox"][3])
            notes = " ".join(l["text"] for l in lines_in(words, cx0, cx1, by1 - 5, below[0]["bbox"][1] - 1))
        elif above:
            text = above[-1]["text"]
        else:  # caption missing from the PDF's text layer: OCR it ourselves
            text = (ocr_caption(gray, bx0 - 10, by1, bx1 + 10, by1 + 30)
                    or ocr_caption(gray, bx0 - 10, by0 - 45, bx1 + 10, by0))
            if text is None:
                print(f"  page {page_num + 1}: no caption for board at ({bx0:.0f}, {by0:.0f}); skipped")
                continue

        if PROBLEM_RE.match(text):
            desc = paragraph(words, cx0, cx1, by1 - 5, max_gap=25)
        elif not desc:
            # The caption is repeated as a header beside the board, and the
            # explanation follows that header instead.
            for x0, x1 in ((0, width / 2), (width / 2, width)):
                for l in lines_in(words, x0, x1, 0, float("inf")):
                    if norm(l["text"]) == norm(text) and not (below and l["bbox"] == below[0]["bbox"]):
                        desc = desc or paragraph(words, x0, x1, l["bbox"][3])

        results.append({
            "page": page_num,
            "pos": (by0, bx0),
            "caption": text,
            "description": desc,
            "notes": notes,
            "stones": read_board(board),
            # the part of the 19x19 board the diagram shows
            "region": (board["col0"], board["row0"], len(board["xs"]), len(board["ys"])),
        })
    return page_num, results


# ---------------------------------------------------------------------------
# Go rules (only what is needed to replay a sequence)
# ---------------------------------------------------------------------------

def neighbours(p):
    x, y = p
    for q in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
        if 0 <= q[0] < 19 and 0 <= q[1] < 19:
            yield q


def group_and_liberties(board, p):
    color = board[p]
    group, libs, todo = {p}, set(), [p]
    while todo:
        for q in neighbours(todo.pop()):
            if q not in board:
                libs.add(q)
            elif board[q] == color and q not in group:
                group.add(q)
                todo.append(q)
    return group, libs


def play(board, color, p):
    board[p] = color
    other = "W" if color == "B" else "B"
    for q in neighbours(p):
        if board.get(q) == other:
            group, libs = group_and_liberties(board, q)
            if not libs:
                for s in group:
                    del board[s]


# ---------------------------------------------------------------------------
# SGF construction
# ---------------------------------------------------------------------------

def sgf_point(p):
    return "abcdefghijklmnopqrs"[p[0]] + "abcdefghijklmnopqrs"[p[1]]


def sgf_text(s):
    return s.replace("\\", "\\\\").replace("]", "\\]")


def sgf_points(prop, points):
    return prop + "".join(f"[{sgf_point(p)}]" for p in sorted(points)) if points else ""


def other(color):
    return "W" if color == "B" else "B"


# Diagram notes for moves that cannot be shown on the board, e.g. "8: connects
# at 1" or "Black 5: connects below 3" (played where an earlier stone was).
NOTE_RE = re.compile(r"(\d{1,3})\s*[:;,]?\s*(?:[A-Za-z]+\s+){0,3}?(at|below|under|above|left of|right of)\s+(\d{1,3})\b")
NOTE_OFFSETS = {"at": (0, 0), "below": (0, 1), "under": (0, 1), "above": (0, -1), "left of": (-1, 0), "right of": (1, 0)}


def build_sequence(initial, diagram, region, notes, label):
    """Turn an answer diagram into (setup_black, setup_white, removed, moves);
    moves are (number, colour, point), point None for a move not on the diagram."""
    numbered = {}
    unread = []
    for p, s in diagram.items():
        if s["num"] is not None:
            numbered.setdefault(s["num"], []).append((p, s["color"]))
        elif s["labelled"]:
            unread.append((p, s["color"], None))

    # Colour of the stones with odd numbers, by majority vote.
    votes = {"B": 0, "W": 0}
    for n, entries in numbered.items():
        for _, c in entries:
            votes[c if n % 2 else other(c)] += 1
    odd_color = "B" if votes["B"] >= votes["W"] else "W"

    def color_of(n):
        return odd_color if n % 2 else other(odd_color)

    moves = {}
    for n, entries in sorted(numbered.items()):
        good = [e for e in entries if e[1] == color_of(n)]
        if len(good) == 1 and len(entries) == 1:
            moves[n] = good[0][0]
        else:
            # A misread, or a number repeated from an earlier diagram for
            # context: decide below which stone (if any) is this move.
            unread.extend((p, c, n) for p, c in entries)

    for n, where, m in NOTE_RE.findall(notes):
        n, m = int(n), int(m)
        if n not in moves and m in moves:
            dx, dy = NOTE_OFFSETS[where]
            moves[n] = (moves[m][0] + dx, moves[m][1] + dy)

    if moves or unread:
        # Unread labels fill the holes in the numbering, then extend it.
        n = min(moves) if moves and min(moves) > len(unread) + 1 else 1
        while unread and (n <= max(moves, default=0) or any(u[1] == color_of(n) for u in unread)):
            cands = [u for u in unread if u[1] == color_of(n)]
            if n not in moves and cands:
                # Prefer a stone read as this number, then the one closest
                # to the neighbouring moves.
                near = [moves[k] for k in (n - 1, n + 1) if k in moves]
                best = min(cands, key=lambda u: (u[2] != n, sum(abs(u[0][0] - q[0]) + abs(u[0][1] - q[1]) for q in near)))
                moves[n] = best[0]
                unread.remove(best)
            n += 1
        if unread:
            print(f"    {label}: could not number stones at "
                  + ", ".join(sgf_point(u[0]) for u in unread) + "; added as setup stones")

    # A last move standing apart from the rest, with a hole of its parity
    # earlier on, is a misread number (e.g. 16 read as 18).
    while len(moves) > 1:
        top = max(moves)
        holes = [n for n in range(min(moves), top) if n not in moves and n % 2 == top % 2]
        if top - 1 in moves or not holes:
            break
        moves[holes[0]] = moves.pop(top)

    move_points = set(moves.values())
    setup_b = {p for p, s in diagram.items() if s["color"] == "B" and p not in move_points and initial.get(p) != "B"}
    setup_w = {p for p, s in diagram.items() if s["color"] == "W" and p not in move_points and initial.get(p) != "W"}
    for p, c, _ in unread:  # still unexplained: show them as setup stones
        (setup_b if c == "B" else setup_w).add(p)

    # Replay to find which initial stones vanish without being captured.
    position = {p: c for p, c in initial.items() if p not in setup_b and p not in setup_w}
    position.update({p: "B" for p in setup_b})
    position.update({p: "W" for p in setup_w})
    start = dict(position)
    seq = []
    cleared = set()  # starting stones that the diagram replays as numbered moves
    if moves:
        for n in range(min(moves), max(moves) + 1):
            seq.append((n, color_of(n), moves.get(n)))
            if n not in moves:
                print(f"    {label}: move {n} is not on the diagram; written as a pass")
                continue
            p = moves[n]
            if p in position and position[p] == start.get(p):
                # e.g. the problem diagram already shows "Black has invaded at 1"
                # and the answer numbers that stone again
                cleared.add(p)
                del position[p]
            play(position, color_of(n), p)
    c0, r0, ncols, nrows = region
    removed = cleared | {p for p in start if p not in diagram and p in position
                         and c0 <= p[0] < c0 + ncols and r0 <= p[1] < r0 + nrows}
    return setup_b, setup_w, removed, seq


def create_sgf(num, problem, answers):
    initial = {p: s["color"] for p, s in problem["stones"].items()}
    turn = "B" if problem["turn"].lower() == "black" else "W"
    marked = {p for p, s in problem["stones"].items() if s["marked"]}

    comment = problem["caption"] + ("\n\n" + problem["description"] if problem["description"] else "")
    out = [f"(;GM[1]FF[4]CA[UTF-8]SZ[19]AP[KIFU-extract]GN[Problem {num}]PL[{turn}]\n"]
    out.append(sgf_points("AB", {p for p, c in initial.items() if c == "B"}))
    out.append(sgf_points("AW", {p for p, c in initial.items() if c == "W"}))
    out.append(sgf_points("TR", marked))
    out.append(f"\nC[{sgf_text(comment)}]\n")

    for ans in answers:
        setup_b, setup_w, removed, seq = build_sequence(initial, ans["stones"], ans["region"], ans["notes"], ans["caption"])
        comment = "\n\n".join(t for t in (ans["caption"], ans["notes"], ans["description"]) if t)
        marks = sgf_points("TR", {p for p, s in ans["stones"].items() if s["marked"]})
        out.append("(")
        setup = sgf_points("AB", setup_b) + sgf_points("AW", setup_w) + sgf_points("AE", removed)
        if setup or not seq:
            out.append(f";{setup}{marks}C[{sgf_text(comment)}]")
            comment = None
        for k, (n, color, p) in enumerate(seq):
            out.append(f"\n;{color}[{sgf_point(p) if p else ''}]")
            if comment and k == 0:
                out.append(f"{marks}C[{sgf_text(comment)}]")
            elif p is None:
                out.append(f"C[{color_name(color)} {n} is not shown in the diagram.]")
        out.append(")\n")
    out.append(")\n")
    return "".join(out)


def color_name(color):
    return "Black" if color == "B" else "White"


# ---------------------------------------------------------------------------

def position_match(problem, diagram):
    """Fraction of the problem's stones (within the diagram's visible part)
    that the answer diagram shows too."""
    if problem is None:
        return None
    c0, r0, ncols, nrows = diagram["region"]
    stones = [(p, s["color"]) for p, s in problem["stones"].items()
              if c0 <= p[0] < c0 + ncols and r0 <= p[1] < r0 + nrows]
    if not stones:
        return None
    return sum(diagram["stones"].get(p, {}).get("color") == c for p, c in stones) / len(stones)


def reading_order(diagrams):
    """Sort diagrams page by page, row by row, left to right."""
    out = []
    for page in sorted({d["page"] for d in diagrams}):
        rest = sorted((d for d in diagrams if d["page"] == page), key=lambda d: d["pos"])
        while rest:
            top = rest[0]["pos"][0]
            row = [d for d in rest if d["pos"][0] - top < 60]
            rest = [d for d in rest if d["pos"][0] - top >= 60]
            out.extend(sorted(row, key=lambda d: d["pos"][1]))
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_sgf.py <pdf_file> [output_dir] [first_page-last_page]")
        sys.exit(1)
    pdf_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output_sgf"
    doc = fitz.open(pdf_path)
    first, last = 1, len(doc)
    if len(sys.argv) > 3:
        first, last = (int(v) for v in sys.argv[3].split("-"))
    pages = list(range(first - 1, last))

    diagrams = []
    # Each worker holds a full-resolution page in memory; keep the pool small.
    with ProcessPoolExecutor(max_workers=max(1, min(6, (os.cpu_count() or 2) - 1))) as pool:
        for page_num, results in pool.map(analyse_page, [(pdf_path, p) for p in pages]):
            print(f"page {page_num + 1}: {len(results)} diagram(s)")
            diagrams.extend(results)
    write_sgfs(diagrams, out_dir)


def write_sgfs(diagrams, out_dir):
    diagrams = reading_order(diagrams)

    problems, answers = {}, {}
    last_num = None
    for d in diagrams:
        m = PROBLEM_RE.match(d["caption"])
        if m:
            num = int(m.group(1))
            if num in problems:
                print(f"  duplicate diagram for Problem {num} on page {d['page'] + 1}; keeping the first")
                continue
            d["turn"] = m.group(2)
            d["caption"] = f"Problem {num}. {m.group(2).capitalize()} to play"
            problems[num] = d
            continue
        m = ANSWER_RE.match(d["caption"])
        if m:
            num, title = int(m.group(1)), m.group(2).strip()
        elif last_num is not None:  # bare "Failure 2": continues the previous answer's problem
            num, title = last_num, d["caption"]
        else:
            continue
        last_num = num
        d["caption"] = f"Problem {num}: {title}"
        answers.setdefault(num, []).append(d)

    # The book occasionally mislabels an answer diagram (e.g. Problem 50's
    # variations captioned "Problem 51"). An answer diagram contains nearly all
    # of its problem's stones, so move a diagram that clearly fits its
    # neighbour's position instead.
    for num in sorted(answers):
        for d in list(answers[num]):
            fit = position_match(problems.get(num), d)
            if fit is not None and fit >= 0.6:
                continue
            for other_num in (num - 1, num + 1):
                if (position_match(problems.get(other_num), d) or 0) >= 0.85:
                    print(f"  page {d['page'] + 1}: '{d['caption']}' matches the position of "
                          f"Problem {other_num}, not {num}; moved there")
                    answers[num].remove(d)
                    answers.setdefault(other_num, []).append(d)
                    break

    os.makedirs(out_dir, exist_ok=True)
    for num in sorted(problems):
        sgf = create_sgf(num, problems[num], answers.get(num, []))
        path = os.path.join(out_dir, f"Problem_{num}.sgf")
        with open(path, "w", encoding="utf-8") as f:
            f.write(sgf)
        print(f"Saved {path} ({len(answers.get(num, []))} answer diagram(s))")
    orphans = sorted(set(answers) - set(problems))
    if orphans:
        print("Answer diagrams without a problem diagram:", orphans)


if __name__ == "__main__":
    main()
