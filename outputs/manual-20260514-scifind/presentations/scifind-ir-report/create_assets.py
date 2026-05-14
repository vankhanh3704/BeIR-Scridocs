from pathlib import Path
import math
import random

from PIL import Image, ImageDraw, ImageFont, ImageFilter


ASSET_DIR = Path(__file__).resolve().parent / "assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)

W, H = 1280, 720
INK = "#171512"
PAPER = "#F7F2EA"
BONE = "#EFE7DA"
MUTED = "#7A756C"
BLUE = "#245B8E"
GREEN = "#167052"
COPPER = "#C4512B"
VIOLET = "#5C477A"


def font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def network_visual():
    img = Image.new("RGB", (W, H), INK)
    draw = ImageDraw.Draw(img, "RGBA")
    random.seed(12)

    for i in range(0, W, 80):
        draw.line([(i, 0), (i, H)], fill=(255, 255, 255, 10), width=1)
    for j in range(0, H, 80):
        draw.line([(0, j), (W, j)], fill=(255, 255, 255, 10), width=1)

    nodes = []
    for i in range(44):
        angle = random.random() * math.pi * 2
        radius = random.uniform(80, 300)
        cx = 760 + math.cos(angle) * radius + random.uniform(-40, 40)
        cy = 360 + math.sin(angle) * radius * 0.72 + random.uniform(-25, 25)
        nodes.append((cx, cy, random.choice([BLUE, GREEN, COPPER, VIOLET])))

    for i, (x1, y1, _) in enumerate(nodes):
        distances = sorted(
            [(math.hypot(x1 - x2, y1 - y2), x2, y2) for j, (x2, y2, _) in enumerate(nodes) if i != j]
        )[:2]
        for _, x2, y2 in distances:
            draw.line([(x1, y1), (x2, y2)], fill=(247, 242, 234, 48), width=1)

    for x, y, color in nodes:
        r = random.uniform(4, 10)
        draw.ellipse((x - r * 2.2, y - r * 2.2, x + r * 2.2, y + r * 2.2), fill=(255, 255, 255, 18))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(247, 242, 234, 130), width=1)

    draw.text((70, 72), "SCIDOCS", font=font(28, True), fill=PAPER)
    draw.text((70, 116), "citation graph / retrieval benchmark", font=font(20), fill=(247, 242, 234, 170))
    draw.rounded_rectangle((70, 560, 410, 636), radius=8, outline=(247, 242, 234, 90), width=1)
    draw.text((94, 580), "25,657 papers  ·  1,000 queries  ·  29,928 qrels", font=font(18), fill=PAPER)
    img = img.filter(ImageFilter.SMOOTH)
    img.save(ASSET_DIR / "citation-network.png")


def ui_snapshot():
    img = Image.new("RGB", (1180, 660), PAPER)
    draw = ImageDraw.Draw(img, "RGBA")

    draw.rectangle((0, 0, 1180, 58), fill=INK)
    draw.text((28, 17), "Sci", font=font(24, True), fill=PAPER)
    draw.text((67, 17), "Find", font=font(24, True), fill=COPPER)
    draw.text((850, 22), "SCIDOCS · 25,657 papers", font=font(14), fill=(247, 242, 234, 165))

    draw.rectangle((0, 58, 235, 660), fill="#FAF8F4")
    draw.line((235, 58, 235, 660), fill="#D8D0C4", width=1)
    draw.text((28, 92), "PHUONG PHAP", font=font(12, True), fill=MUTED)
    methods = [("TF-IDF", BLUE, True), ("Boolean Search", BLUE, False), ("LSA / SVD", GREEN, False), ("BM25", COPPER, False), ("So sanh tat ca", VIOLET, False)]
    y = 122
    for label, color, active in methods:
        if active:
            draw.rounded_rectangle((22, y - 8, 212, y + 30), radius=5, fill="#E8EEF8", outline=color, width=1)
        draw.ellipse((36, y + 4, 46, y + 14), fill=color)
        draw.text((58, y), label, font=font(15, active), fill=color if active else MUTED)
        y += 45

    draw.text((28, 382), "EVALUATION", font=font(12, True), fill=MUTED)
    eval_rows = [("TF-IDF NDCG@10", "0.1462", BLUE), ("TF-IDF MAP", "0.0942", BLUE), ("BM25 NDCG@10", "0.1611", COPPER), ("BM25 Recall@10", "0.1670", COPPER)]
    y = 414
    for label, value, color in eval_rows:
        draw.line((28, y + 25, 207, y + 25), fill="#D8D0C4", width=1)
        draw.text((28, y), label, font=font(12), fill=INK)
        draw.text((155, y), value, font=font(12, True), fill=color)
        y += 32

    draw.text((280, 93), "NHAP QUERY - TU KHOA HOAC CAU MO TA", font=font(12, True), fill=MUTED)
    draw.rounded_rectangle((280, 118, 1110, 174), radius=2, fill="#FAF8F4", outline=INK, width=2)
    draw.text((304, 136), "citation recommendation", font=font(22, True), fill=INK)
    draw.rectangle((972, 118, 1110, 174), fill=INK)
    draw.text((1003, 137), "SEARCH", font=font(16, True), fill=PAPER)

    draw.line((280, 222, 1110, 222), fill="#D8D0C4", width=1)
    draw.text((280, 196), "TF-IDF - citation recommendation", font=font(14, True), fill=MUTED)
    draw.text((930, 196), "SCIDOCS · query", font=font(13), fill=MUTED)

    cards = [
        ("01", "RefSeer: A citation recommendation system", "55.4%", BLUE),
        ("02", "Recommending citations with translation model", "43.2%", BLUE),
        ("03", "Result Diversification in Automatic Citation Recommendation", "38.3%", BLUE),
    ]
    y = 250
    for num, title, score, color in cards:
        draw.rounded_rectangle((280, y, 1110, y + 96), radius=2, fill="#FAF8F4", outline="#D8D0C4", width=1)
        draw.text((304, y + 22), num, font=font(14, True), fill=MUTED)
        draw.text((352, y + 18), title, font=font(18, True), fill=INK)
        draw.text((1035, y + 18), score, font=font(18, True), fill=color)
        draw.text((352, y + 52), "Citations are important in academic dissemination. A search result card shows title, id, abstract and score.", font=font(13), fill=MUTED)
        y += 112

    img.save(ASSET_DIR / "scifind-ui.png")


network_visual()
ui_snapshot()
