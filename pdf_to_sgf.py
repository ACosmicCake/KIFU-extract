import sys
import os
import fitz
import re
import cv2
import numpy as np
import pytesseract
from sklearn.cluster import AgglomerativeClustering

class BoardScanner:
    def get_grid(self, image, detected_stones):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        line_mask = gray.copy()
        for x, y, r in detected_stones:
            cv2.circle(line_mask, (x, y), r+2, 255, -1)

        edges = cv2.Canny(line_mask, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=10)

        h_lines, v_lines = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y1 - y2) < 10: h_lines.append((y1 + y2) / 2)
                elif abs(x1 - x2) < 10: v_lines.append((x1 + x2) / 2)

        def cluster_lines(lines_list):
            if not lines_list: return []
            lines_array = np.array(lines_list).reshape(-1, 1)
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=10, linkage='average')
            clustering.fit(lines_array)
            clusters = {}
            for i, label in enumerate(clustering.labels_):
                if label not in clusters: clusters[label] = []
                clusters[label].append(lines_list[i])
            return sorted([np.mean(c) for c in clusters.values()])

        h_centers = cluster_lines(h_lines)
        v_centers = cluster_lines(v_lines)

        def fit_grid(centers, image_size):
            if len(centers) < 2:
                margin = 20
                step = (image_size - 2*margin) / 18.0
                return [margin + i*step for i in range(19)]
            min_c, max_c = centers[0], centers[-1]
            diffs = np.diff(centers)
            median_space = np.median(diffs) if len(diffs) > 0 else 30
            num_gaps = int(round((max_c - min_c) / median_space)) if median_space > 0 else 18
            if num_gaps == 0: num_gaps = 18
            step = (max_c - min_c) / num_gaps
            return [min_c + i*step for i in range(19)]

        return fit_grid(v_centers, image.shape[1]), fit_grid(h_centers, image.shape[0])

    def scan_board(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=15,
            param1=50, param2=20, minRadius=10, maxRadius=25
        )

        detected_stones = []
        if circles is not None:
            for i in np.uint16(np.around(circles))[0, :]:
                detected_stones.append((i[0], i[1], i[2]))

        grid_x, grid_y = self.get_grid(image, detected_stones)

        black_stones = set()
        white_stones = set()
        ocr_moves = {}

        for sx, sy, sr in detected_stones:
            sx = int(sx)
            sy = int(sy)
            sr = int(sr)
            if not grid_x or not grid_y: continue
            gx = min(range(19), key=lambda i: abs(grid_x[i] - sx))
            gy = min(range(19), key=lambda i: abs(grid_y[i] - sy))

            roi = gray[max(0, sy-sr):min(image.shape[0], sy+sr), max(0, sx-sr):min(image.shape[1], sx+sr)]
            if roi.size == 0: continue

            mask = np.zeros(roi.shape, dtype=np.uint8)
            cv2.circle(mask, (sr, sr), sr-2, 255, -1)
            mean_val = cv2.mean(roi, mask=mask)[0]
            is_black = mean_val < 128

            center_roi = roi[max(0, roi.shape[0]//2-6):min(roi.shape[0], roi.shape[0]//2+6),
                             max(0, roi.shape[1]//2-6):min(roi.shape[1], roi.shape[1]//2+6)]
            std_dev = np.std(center_roi) if center_roi.size > 0 else 0

            number = None
            if std_dev > 15:
                roi_base = cv2.bitwise_not(roi) if is_black else roi
                found_num = None
                for scale in [3, 4, 5]:
                    roi_scaled = cv2.resize(roi_base, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    for thresh_type in [cv2.THRESH_BINARY | cv2.THRESH_OTSU, cv2.THRESH_BINARY]:
                        _, roi_thresh = cv2.threshold(roi_scaled, 128, 255, thresh_type)
                        for cfg in ['--psm 10', '--psm 7']:
                            config = f'{cfg} -c tessedit_char_whitelist=123456789'
                            text = pytesseract.image_to_string(roi_thresh, config=config).strip()
                            if text.isdigit() and 0 < int(text) < 30:
                                found_num = int(text)
                                break
                        if found_num: break
                    if found_num: break
                if found_num: number = found_num

            if number:
                ocr_moves[number] = ('B' if is_black else 'W', (gx, gy))
            if is_black:
                black_stones.add((gx, gy))
            else:
                white_stones.add((gx, gy))

        return {
            'black': black_stones,
            'white': white_stones,
            'ocr': ocr_moves
        }

class PDFExtractor:
    def __init__(self, pdf_path):
        self.doc = fitz.open(pdf_path)
        self.scanner = BoardScanner()

    def extract_all(self):
        problems = {}
        problem_re = re.compile(r"Problem\s+(\d+)\.\s+(Black|White)\s+to\s+play", re.IGNORECASE)
        answer_re = re.compile(r"Problem\s+(\d+):\s+(.*)", re.IGNORECASE)

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            pix = page.get_pixmap(dpi=300)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            cv_img = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR) if pix.n == 4 else cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            scale = 300 / 72.0
            text_blocks = []
            for b in page.get_text("blocks"):
                text_blocks.append({"text": b[4].strip(), "y0": b[1] * scale, "x0": b[0] * scale})

            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            horiz = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)), iterations=2)
            vert = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40)), iterations=2)
            grid = cv2.dilate(cv2.addWeighted(horiz, 0.5, vert, 0.5, 0.0), np.ones((5,5), np.uint8), iterations=2)
            contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            boards = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                aspect = float(w)/h
                if area > 300000 and 0.9 < aspect < 1.1:
                    boards.append({"y": y, "x": x, "is_full": True, "image": cv_img[max(0, y-10):min(cv_img.shape[0], y+h+10), max(0, x-10):min(cv_img.shape[1], x+w+10)]})
                elif area > 100000:
                    boards.append({"y": y, "x": x, "is_full": False, "image": None})

            for tb in text_blocks:
                text = tb["text"]

                m_probs = list(problem_re.finditer(text))
                if m_probs:
                    for i, m_prob in enumerate(m_probs):
                        prob_num = int(m_prob.group(1))
                        turn = m_prob.group(2)
                        if prob_num not in problems:
                            problems[prob_num] = {"turn": turn, "text": m_prob.group(0), "initial_image": None, "answers": []}

                        cands = [b for b in boards if b["y"] > tb["y0"]-50 and b["y"] < tb["y0"] + 1500]
                        cands.sort(key=lambda b: (b["y"], b["x"]))
                        if len(cands) > i:
                            problems[prob_num]["initial_image"] = cands[i]["image"]

                m_anses = list(answer_re.finditer(text))
                if m_anses:
                    for i, m_ans in enumerate(m_anses):
                        prob_num = int(m_ans.group(1))
                        ans_title = m_ans.group(2)
                        if prob_num in problems:
                            # Answers might also be grouped in one text block
                            cands = [(b["y"]-tb["y0"], b) for b in boards if b["y"] > tb["y0"]-50 and abs(b["x"]-tb["x0"]) < cv_img.shape[1]/2]
                            if cands:
                                b = sorted(cands, key=lambda c: c[0])[0][1]
                                if b["is_full"]:
                                    problems[prob_num]["answers"].append({"title": ans_title, "text": m_ans.group(0), "image": b["image"]})

        return problems

class SGFWriter:
    def __init__(self):
        self.coords = "abcdefghijklmnopqrstuvwxyz"

    def coord_to_sgf(self, x, y):
        return f"{self.coords[x]}{self.coords[y]}"

    def create_sgf(self, problem_data, scanner):
        init_scan = scanner.scan_board(problem_data['initial_image'])

        sgf = "(;FF[4]GM[1]SZ[19]\n"
        sgf += f"C[{problem_data['text']}]\n"
        sgf += "PL[B]\n" if problem_data['turn'] == 'Black' else "PL[W]\n"

        if init_scan['black']:
            sgf += "AB" + "".join([f"[{self.coord_to_sgf(x, y)}]" for x, y in init_scan['black']]) + "\n"
        if init_scan['white']:
            sgf += "AW" + "".join([f"[{self.coord_to_sgf(x, y)}]" for x, y in init_scan['white']]) + "\n"

        init_stones = {('B', p) for p in init_scan['black']} | {('W', p) for p in init_scan['white']}

        for ans in problem_data['answers']:
            ans_scan = scanner.scan_board(ans['image'])
            ans_stones = {('B', p) for p in ans_scan['black']} | {('W', p) for p in ans_scan['white']}
            new_stones = ans_stones - init_stones

            moves = ans_scan['ocr']
            found_coords = {v[1] for v in moves.values()}
            missing_stones = [s for s in new_stones if s[1] not in found_coords]

            # Simple heuristic: try to fill in gaps in move sequence
            expected_moves = len(moves) + len(missing_stones)
            for num in range(1, expected_moves + 1):
                if num not in moves and missing_stones:
                    s = missing_stones.pop()
                    moves[num] = s

            sgf += "\n("
            first_move = True
            for move_num, (color, (x, y)) in sorted(moves.items()):
                sgf += f";{color}[{self.coord_to_sgf(x, y)}]"
                if first_move:
                    sgf += f"C[{ans['text']}]"
                    first_move = False
                sgf += "\n"
            sgf += ")"

        sgf += "\n)\n"
        return sgf

def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_sgf.py <pdf_file>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    extractor = PDFExtractor(pdf_path)
    problems = extractor.extract_all()

    os.makedirs("output_sgf", exist_ok=True)
    writer = SGFWriter()

    for p, data in problems.items():
        if data['initial_image'] is not None:
            print(f"Generating SGF for Problem {p}...")
            try:
                sgf = writer.create_sgf(data, extractor.scanner)
                with open(f"output_sgf/Problem_{p}.sgf", "w") as f:
                    f.write(sgf)
                print(f"Saved output_sgf/Problem_{p}.sgf")
            except Exception as e:
                print(f"Failed to generate SGF for Problem {p}: {e}")
        else:
            print(f"Skipping Problem {p}: Initial board not found.")

if __name__ == "__main__":
    main()
