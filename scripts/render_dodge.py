"""Render a dodge trace (from `neat_play --trace`) to an animated GIF.

    neat_play --env dodge --genome results/dodge/champion.genome --seed 3009 --trace trace.csv
    python scripts/render_dodge.py trace.csv docs/media/dodge.gif --every 4 --fps 30
"""
import argparse
import csv
from collections import defaultdict

from PIL import Image, ImageDraw

ROWS, WIDTH, PLAYER_X = 6, 720, 720 - 21


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=4, help="keep every n-th 60 Hz frame")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--scale", type=float, default=0.75)
    args = ap.parse_args()

    frames = defaultdict(lambda: {"player": 0, "projectiles": []})
    with open(args.trace) as f:
        for row in csv.DictReader(f):
            frame = frames[int(row["frame"])]
            frame["player"] = int(row["player_row"])
            if row["projectile_x"]:
                frame["projectiles"].append((float(row["projectile_x"]), int(row["projectile_row"])))

    w, row_h = int(WIDTH * args.scale), int(48 * args.scale)
    h = ROWS * row_h + int(26 * args.scale)
    images = []
    last = max(frames)
    for index in list(range(0, last, args.every)) + [last] * 20:  # hold the final frame
        frame = frames[index]
        img = Image.new("RGB", (w, h), (250, 250, 250))
        d = ImageDraw.Draw(img)
        top = int(26 * args.scale)
        for r in range(ROWS):
            d.rectangle([0, top + r * row_h, w, top + (r + 1) * row_h - 2], fill=(235, 238, 245) if r % 2 else (244, 246, 250))
        rad = 7 * args.scale
        for x, r in frame["projectiles"]:
            cx, cy = x * args.scale, top + (r + 0.5) * row_h
            d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(222, 45, 38))
        pr = 9 * args.scale
        px, py = PLAYER_X * args.scale, top + (frame["player"] + 0.5) * row_h
        color = (20, 20, 20) if index == last else (43, 140, 190)
        d.ellipse([px - pr, py - pr, px + pr, py + pr], fill=color)
        d.text((8, 4), f"t = {index / 60:5.1f} s", fill=(40, 40, 40))
        images.append(img.convert("P", palette=Image.ADAPTIVE, colors=32))
    images[0].save(args.out, save_all=True, append_images=images[1:], duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"wrote {args.out}: {len(images)} frames")


if __name__ == "__main__":
    main()
