# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "0430"

MAPO = "\ub9c8\ud3ec\uad6c"
DONGS = ["\uc5f0\ub0a8\ub3d9", "\uc11c\uad50\ub3d9"]


def crop_card(name: str) -> Path:
    src = OUT_DIR / f"{MAPO}_{name}_describe.png"
    dst = OUT_DIR / f"{MAPO}_{name}_describe_cropped.png"
    img = Image.open(src)
    w, h = img.size

    # Keep only the lower describe table area.
    left = int(w * 0.105)
    top = int(h * 0.535)
    right = int(w * 0.965)
    bottom = int(h * 0.935)
    cropped = img.crop((left, top, right, bottom))

    title_h = 110
    canvas = Image.new("RGB", (cropped.width, cropped.height + title_h), "#F4F7FB")
    canvas.paste(cropped.convert("RGB"), (0, title_h))

    draw = ImageDraw.Draw(canvas)
    font_path = Path("C:/Windows/Fonts/malgunbd.ttf")
    font = ImageFont.truetype(str(font_path), 44) if font_path.exists() else ImageFont.load_default()
    title = f"{MAPO} {name} describe"
    draw.text((24, 28), title, fill="#172033", font=font)

    canvas.save(dst)
    return dst


def main() -> None:
    for name in DONGS:
        print(crop_card(name))


if __name__ == "__main__":
    main()
