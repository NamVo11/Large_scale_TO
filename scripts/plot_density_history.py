#!/usr/bin/env python3
import argparse
from pathlib import Path

from PIL import Image


def jet_palette():
    def clamp01(x):
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

    palette = []
    for i in range(256):
        x = i / 255.0
        r = clamp01(1.5 - abs(4.0 * x - 3.0))
        g = clamp01(1.5 - abs(4.0 * x - 2.0))
        b = clamp01(1.5 - abs(4.0 * x - 1.0))
        palette.extend([int(round(255 * r)), int(round(255 * g)), int(round(255 * b))])
    return palette


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert saved density PGM files to PNG frames and an animated GIF.")
    parser.add_argument("result_dir", help="Stage2 result directory, e.g. results/stage2_density_plot_test")
    parser.add_argument("--scale", type=int, default=4, help="Nearest-neighbor image scale factor.")
    parser.add_argument("--duration", type=int, default=250, help="GIF frame duration in milliseconds.")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    density_dir = result_dir / "density"
    pgms = sorted(density_dir.glob("density_iter_*.pgm"))
    if not pgms:
        raise SystemExit(f"No density_iter_*.pgm files found in {density_dir}")

    png_dir = density_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    for pgm in pgms:
        img = Image.open(pgm).convert("L")
        if args.scale > 1:
            img = img.resize((img.width * args.scale, img.height * args.scale), nearest)
        img = img.convert("P")
        img.putpalette(jet_palette())
        png = png_dir / f"{pgm.stem}.png"
        img.save(png)
        frames.append(img)

    gif_path = density_dir / "density_history.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration,
        loop=0,
    )
    print(f"Wrote {len(frames)} PNG frames to {png_dir}")
    print(f"Wrote {gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
