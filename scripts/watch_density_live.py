#!/usr/bin/env python3
import argparse
import csv
import http.server
import socketserver
import threading
import time
from pathlib import Path

from PIL import Image


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Large_scale_TO Density Live</title>
  <style>
    body { margin: 0; background: #181818; color: #e8e8e8; font-family: Arial, sans-serif; }
    header { padding: 12px 16px; border-bottom: 1px solid #333; display: flex; gap: 20px; align-items: baseline; }
    main { padding: 16px; }
    img { image-rendering: pixelated; max-width: 100%; height: auto; background: white; }
    .muted { color: #aaa; font-size: 13px; }
  </style>
</head>
<body>
  <header>
    <strong>Density Live</strong>
    <span id="stamp" class="muted"></span>
    <span class="muted">WATCH_DIR</span>
    <span id="frame" class="muted"></span>
  </header>
  <main>
    <img id="density" src="latest_density.png?t=0" alt="latest density">
    <p id="waiting" class="muted">Waiting for density frames...</p>
  </main>
  <script>
    const intervalMs = REFRESH_MS;
    function refresh() {
      const t = Date.now();
      const img = document.getElementById("density");
      img.onload = () => { document.getElementById("waiting").style.display = "none"; img.style.display = "block"; };
      img.onerror = () => { document.getElementById("waiting").style.display = "block"; img.style.display = "none"; };
      img.src = "latest_density.png?t=" + t;
      document.getElementById("stamp").textContent = new Date().toLocaleTimeString();
      fetch("latest_density.txt?t=" + t)
        .then(r => r.ok ? r.text() : "")
        .then(txt => { document.getElementById("frame").textContent = txt.trim(); })
        .catch(() => {});
    }
    setInterval(refresh, intervalMs);
    refresh();
  </script>
</body>
</html>
"""


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


def latest_frame(density_dir: Path):
    files = sorted(density_dir.glob("density_iter_*.csv"))
    if files:
        for csv_file in reversed(files):
            if csv_file.with_suffix(".pgm").exists():
                return csv_file
        return files[-1]
    files = sorted(density_dir.glob("density_iter_*.pgm"))
    return files[-1] if files else None


def csv_to_gray(csv_path: Path) -> Image.Image:
    rows = []
    nelx = 0
    nely = 0
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            i = int(row["i"])
            j = int(row["j"])
            x = float(row["xPhys"])
            rows.append((i, j, x))
            nelx = max(nelx, i + 1)
            nely = max(nely, j + 1)
    gray = Image.new("L", (nelx, nely), 0)
    pix = gray.load()
    for i, j, x in rows:
        # Browser images have y downward. MATLAB compare uses YDir=normal, so flip y
        # here while keeping the same CSV data and color scale [0, 1].
        pix[i, nely - 1 - j] = int(round(255 * max(0.0, min(1.0, x))))
    return gray


def convert_latest(density_dir: Path, scale: int, last_seen: Path | None):
    frame = latest_frame(density_dir)
    if frame is None or frame == last_seen:
        return last_seen, False
    nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    if frame.suffix.lower() == ".csv":
        gray = csv_to_gray(frame)
    else:
        gray = Image.open(frame).convert("L")
    if scale > 1:
        gray = gray.resize((gray.width * scale, gray.height * scale), nearest)
    color = gray.convert("P")
    color.putpalette(jet_palette())
    color.save(density_dir / "latest_density.png")
    (density_dir / "latest_density.txt").write_text(frame.name + "\n", encoding="utf-8")
    return frame, True


def serve(directory: Path, port: int):
    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(*args, directory=str(directory), **kwargs)
    with socketserver.TCPServer(("0.0.0.0", port), handler) as httpd:
        httpd.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch stage2 density files and expose a live browser view.")
    parser.add_argument("result_dir", help="Stage2 result directory, e.g. results/Nam1111")
    parser.add_argument("--scale", type=int, default=4, help="Nearest-neighbor display scale factor.")
    parser.add_argument("--refresh", type=float, default=0.5, help="Watcher refresh interval in seconds.")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port for browser view.")
    parser.add_argument("--once", action="store_true", help="Convert the latest frame once and exit.")
    parser.add_argument("--no-server", action="store_true", help="Do not start the HTTP server.")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    density_dir = result_dir / "density"
    density_dir.mkdir(parents=True, exist_ok=True)
    html = HTML.replace("REFRESH_MS", str(max(100, int(args.refresh * 1000))))
    html = html.replace("WATCH_DIR", str(result_dir))
    (density_dir / "index.html").write_text(html, encoding="utf-8")

    last_seen, updated = convert_latest(density_dir, args.scale, None)
    if args.once:
        if updated:
            print(f"Updated {density_dir / 'latest_density.png'} from {last_seen.name}")
        else:
            print(f"No density_iter_*.pgm found in {density_dir}")
        return 0

    if not args.no_server:
        thread = threading.Thread(target=serve, args=(density_dir, args.port), daemon=True)
        thread.start()
        print(f"Open http://localhost:{args.port}/index.html")
    print(f"Watching {density_dir}")

    while True:
        last_seen, updated = convert_latest(density_dir, args.scale, last_seen)
        if updated:
            print(f"Updated latest_density.png from {last_seen.name}", flush=True)
        time.sleep(args.refresh)


if __name__ == "__main__":
    raise SystemExit(main())
