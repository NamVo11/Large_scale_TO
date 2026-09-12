#!/usr/bin/env python3
"""Purpose:
Serve live optimization plots from stage2 CSV output.

Usage:
    python3 scripts/watch_optimization_live.py case_study/test1 --port 8765
"""

import argparse
import csv
import http.server
import math
import shutil
import socketserver
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


DEFAULT_CASE_DIR = "case_study/test1"
DEFAULT_SCALE = 8
DEFAULT_PORT = 8765


HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Large_scale_TO Live</title>
  <style>
    body { margin: 0; background: #f2f2f2; color: #111; font-family: Arial, sans-serif; }
    header { position: sticky; top: 0; z-index: 1; display: flex; gap: 18px; align-items: baseline;
      padding: 10px 14px; background: #151515; color: #eee; border-bottom: 1px solid #333; }
    main { padding: 12px; display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 12px; }
    section { background: white; border: 1px solid #d0d0d0; padding: 10px; min-height: 260px; }
    h2 { margin: 0 0 8px; font-size: 15px; text-align: center; }
    canvas { width: 100%; height: 250px; display: block; }
    img { image-rendering: pixelated; max-width: 100%; height: auto; display: block; margin: 0 auto; }
    .muted { color: #aaa; font-size: 13px; }
    .hidden { display: none; }
  </style>
</head>
<body>
  <header>
    <strong>Optimization Live</strong>
    <span id="stamp" class="muted"></span>
    <span class="muted">WATCH_DIR</span>
    <span id="frame" class="muted"></span>
  </header>
  <main>
    <section id="density_panel"><h2>Density</h2><img id="density" src="latest_density.png?t=0"><p id="density_wait" class="muted">Waiting for density frames...</p></section>
    <section id="history_panel"><h2>Objective and Constraint Value History</h2><canvas id="history"></canvas></section>
    <section id="forward_residual_panel"><h2>Forward ROM Residual</h2><canvas id="forward_residual"></canvas></section>
    <section id="obj_sens_residual_panel"><h2>Sensitivity ROM Residual</h2><canvas id="obj_sens_residual"></canvas></section>
    <section id="forward_rank_panel"><h2>Forward Basis Rank</h2><canvas id="forward_rank"></canvas></section>
    <section id="obj_sens_rank_panel"><h2>Sensitivity Basis Rank</h2><canvas id="obj_sens_rank"></canvas></section>
    <section id="forward_mode_panel"><h2>Forward FOM Mode</h2><canvas id="forward_mode"></canvas></section>
    <section id="obj_sens_mode_panel"><h2>Sensitivity FOM Mode</h2><canvas id="obj_sens_mode"></canvas></section>
    <section id="change_panel"><h2>Design Variable Change History</h2><canvas id="change"></canvas></section>
    <section id="obj_sens_error_panel"><h2>Objective Sensitivity ROM-FOM Solution Error</h2><canvas id="obj_sens_error"></canvas></section>
  </main>
<script>
const refreshMs = REFRESH_MS;

function finite(v) { return Number.isFinite(v); }
function showPanel(id, show) { document.getElementById(id).classList.toggle("hidden", !show); }

async function fetchText(path) {
  const r = await fetch(path + "?t=" + Date.now());
  if (!r.ok) throw new Error(path);
  return await r.text();
}

async function fetchResultText(path) {
  return await fetchText("result/" + path);
}

function parseNumber(x) {
  const v = Number(String(x).trim());
  return Number.isFinite(v) ? v : NaN;
}

async function readMatrix(path) {
  const txt = (await fetchResultText(path)).trim();
  if (!txt) return [];
  return txt.split(/\r?\n/).map(line => line.split(",").map(parseNumber));
}

async function readHistory() {
  const txt = (await fetchResultText("optimization_history.csv")).trim();
  if (!txt) return [];
  const lines = txt.split(/\r?\n/);
  const header = lines[0].split(",");
  return lines.slice(1).filter(Boolean).map(line => {
    const vals = line.split(",");
    const row = {};
    header.forEach((h, i) => row[h] = parseNumber(vals[i]));
    return row;
  });
}

function flattenMatrix(m) {
  const out = [];
  for (const row of m) for (const v of row) if (finite(v)) out.push(v);
  return out;
}

function matrixPoints(m, activeValue) {
  const pts = [];
  const bounds = {xmax: Math.max(0, m.length - 1), ymax: 0};
  for (let i = 0; i < m.length; ++i) {
    bounds.ymax = Math.max(bounds.ymax, m[i].length - 1);
    for (let j = 0; j < m[i].length; ++j) {
      const v = m[i][j];
      if (finite(v) && Math.abs(v - activeValue) < 0.5) pts.push({x: i, y: j});
    }
  }
  return {pts, bounds};
}

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(320, Math.floor(rect.width * dpr));
  canvas.height = Math.max(220, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {ctx, w: canvas.width / dpr, h: canvas.height / dpr};
}

function drawAxes(ctx, w, h, xmin, xmax, ymin, ymax) {
  const pad = {l: 58, r: 18, t: 14, b: 38};
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#bbb";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad.l, pad.t, w - pad.l - pad.r, h - pad.t - pad.b);
  ctx.fillStyle = "#333";
  ctx.font = "11px Arial";
  ctx.textAlign = "right";
  ctx.fillText(ymax.toExponential ? ymax.toPrecision(3) : ymax, pad.l - 6, pad.t + 4);
  ctx.fillText(ymin.toPrecision ? ymin.toPrecision(3) : ymin, pad.l - 6, h - pad.b);
  ctx.textAlign = "center";
  ctx.fillText(String(Math.round(xmin)), pad.l, h - 12);
  ctx.fillText(String(Math.round(xmax)), w - pad.r, h - 12);
  const sx = x => pad.l + (x - xmin) / Math.max(1e-14, xmax - xmin) * (w - pad.l - pad.r);
  const sy = y => h - pad.b - (y - ymin) / Math.max(1e-14, ymax - ymin) * (h - pad.t - pad.b);
  return {sx, sy, pad};
}

function drawLines(canvasId, series) {
  const canvas = document.getElementById(canvasId);
  const {ctx, w, h} = resizeCanvas(canvas);
  series = series.filter(s => s.y.some(finite));
  if (!series.length) return false;
  const xs = series.flatMap(s => s.x.filter(finite));
  const ys = series.flatMap(s => s.y.filter(finite));
  let xmin = Math.min(...xs), xmax = Math.max(...xs);
  let ymin = Math.min(...ys), ymax = Math.max(...ys);
  if (xmin === xmax) xmax = xmin + 1;
  if (ymin === ymax) { ymin -= 1; ymax += 1; }
  const ax = drawAxes(ctx, w, h, xmin, xmax, ymin, ymax);
  ctx.lineWidth = 1.6;
  series.forEach((s, si) => {
    ctx.strokeStyle = s.color;
    ctx.fillStyle = s.color;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < s.x.length; ++i) {
      if (!finite(s.x[i]) || !finite(s.y[i])) continue;
      const x = ax.sx(s.x[i]), y = ax.sy(s.y[i]);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    for (let i = 0; i < s.x.length; ++i) {
      if (!finite(s.x[i]) || !finite(s.y[i])) continue;
      ctx.beginPath(); ctx.arc(ax.sx(s.x[i]), ax.sy(s.y[i]), 2, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.fillText(s.name, ax.pad.l + 60 + si * 120, 15);
  });
  return true;
}

function drawScatter(canvasId, scatterData, color) {
  const canvas = document.getElementById(canvasId);
  const {ctx, w, h} = resizeCanvas(canvas);
  const pts = scatterData.pts || [];
  if (!pts.length) return false;
  const bounds = scatterData.bounds || {};
  const xmin = 0, xmax = Math.max(1, bounds.xmax ?? Math.max(1, ...pts.map(p => p.x)));
  const ymin = 0, ymax = Math.max(1, bounds.ymax ?? Math.max(1, ...pts.map(p => p.y)));
  const ax = drawAxes(ctx, w, h, xmin, xmax, ymin, ymax);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.2;
  for (const p of pts) {
    const x = ax.sx(p.x), y = ax.sy(p.y);
    ctx.beginPath();
    ctx.moveTo(x - 4, y - 4); ctx.lineTo(x + 4, y + 4);
    ctx.moveTo(x + 4, y - 4); ctx.lineTo(x - 4, y + 4);
    ctx.stroke();
  }
  return true;
}

function rowIndex(n) { return Array.from({length: n}, (_, i) => i); }

async function updateDensity() {
  const img = document.getElementById("density");
  img.onload = () => { img.style.display = "block"; document.getElementById("density_wait").style.display = "none"; };
  img.onerror = () => { img.style.display = "none"; document.getElementById("density_wait").style.display = "block"; };
  img.src = "latest_density.png?t=" + Date.now();
  try { document.getElementById("frame").textContent = (await fetchText("latest_density.txt")).trim(); } catch {}
}

async function updatePlots() {
  document.getElementById("stamp").textContent = new Date().toLocaleTimeString();
  updateDensity();
  try {
    const h = await readHistory();
    const x = h.map(r => r.Iter);
    showPanel("history_panel", drawLines("history", [
      {name: "Objective", x, y: h.map(r => r.objective), color: "#1f77b4"},
      {name: "Constraint", x, y: h.map(r => r.constraint), color: "#d95319"}
    ]));
    showPanel("change_panel", drawLines("change", [
      {name: "Max change", x, y: h.map(r => r.Iter === 0 ? NaN : r.change), color: "#1f77b4"},
      {name: "RMS change", x, y: h.map(r => r.Iter === 0 ? NaN : r.rms_change), color: "#d95319"}
    ]));
  } catch { showPanel("history_panel", false); showPanel("change_panel", false); }

  const lineMatrix = async (panel, canvas, mainFile, mainName, secondaryFile, secondaryName) => {
    try {
      const y = flattenMatrix(await readMatrix(mainFile));
      const x = rowIndex(y.length);
      const series = [{name: mainName, x, y, color: "#1f77b4"}];
      if (secondaryFile) {
        try {
          const ys = flattenMatrix(await readMatrix(secondaryFile));
          series.push({name: secondaryName, x: rowIndex(ys.length), y: ys, color: "#d95319"});
        } catch {}
      }
      showPanel(panel, drawLines(canvas, series));
    } catch { showPanel(panel, false); }
  };
  await lineMatrix("forward_residual_panel", "forward_residual", "forward_residual.csv", "Residual", "forward_threshold.csv", "Threshold");
  await lineMatrix("obj_sens_residual_panel", "obj_sens_residual", "obj_sens_residual.csv", "Residual", "obj_sens_threshold.csv", "Threshold");
  await lineMatrix("forward_rank_panel", "forward_rank", "forward_rank.csv", "Rank", "", "");
  await lineMatrix("obj_sens_rank_panel", "obj_sens_rank", "obj_sens_rank.csv", "Rank", "", "");
  await lineMatrix("obj_sens_error_panel", "obj_sens_error", "obj_sens_error.csv", "Error", "obj_sens_basis_error.csv", "Basis error");

  try { showPanel("forward_mode_panel", drawScatter("forward_mode", matrixPoints(await readMatrix("forward_fom_used.csv"), 1), "#d95319")); }
  catch { showPanel("forward_mode_panel", false); }
  try { showPanel("obj_sens_mode_panel", drawScatter("obj_sens_mode", matrixPoints(await readMatrix("obj_sens_fom_used.csv"), 1), "#d95319")); }
  catch { showPanel("obj_sens_mode_panel", false); }
}

setInterval(updatePlots, refreshMs);
updatePlots();
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
        pix[i, nely - 1 - j] = int(round(255 * max(0.0, min(1.0, x))))
    return gray


def convert_latest(density_dir: Path, latest_dir: Path, scale: int, last_seen: Path | None):
    frame = latest_frame(density_dir)
    if frame is None or frame == last_seen:
        return last_seen, False
    nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    gray = csv_to_gray(frame) if frame.suffix.lower() == ".csv" else Image.open(frame).convert("L")
    if scale > 1:
        gray = gray.resize((gray.width * scale, gray.height * scale), nearest)
    color = gray.convert("P")
    color.putpalette(jet_palette())
    color.save(latest_dir / "latest_density.png")
    (latest_dir / "latest_density.txt").write_text(frame.name + "\n", encoding="utf-8")
    return frame, True


def read_history_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: to_float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def to_float(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else math.nan
    except Exception:
        return math.nan


def read_matrix_csv(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            rows.append([to_float(v) for v in row])
    return rows


def flatten_matrix(rows):
    return [v for row in rows for v in row if math.isfinite(v)]


def has_values(values):
    return any(math.isfinite(v) for v in values)


def save_line_figure(path: Path, title: str, series, ylabel: str = "Value"):
    series = [(name, x, y) for name, x, y in series if has_values(y)]
    if not series:
        return False
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
    for name, x, y in series:
        ax.plot(x, y, "-o", linewidth=1.2, markersize=2.5, label=name)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def save_history_figure(result_dir: Path, figure_dir: Path, ext: str = "png"):
    rows = read_history_csv(result_dir / "optimization_history.csv")
    if not rows:
        return False
    x = [r.get("Iter", i) for i, r in enumerate(rows)]
    objective = [r.get("objective", math.nan) for r in rows]
    constraint = [r.get("constraint", math.nan) for r in rows]
    if not has_values(objective) and not has_values(constraint):
        return False

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), dpi=140, sharex=True)
    axes[0].plot(x, objective, "-o", linewidth=1.4, markersize=2.5)
    axes[0].set_title("Objective and Constraint Value History")
    axes[0].set_ylabel("Objective")
    axes[0].grid(True, alpha=0.35)
    axes[1].plot(x, constraint, "-x", linewidth=1.4, markersize=3.0)
    axes[1].set_xlabel("Optimization iteration")
    axes[1].set_ylabel("Constraint")
    axes[1].grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(figure_dir / f"objective_constraint_history.{ext}")
    plt.close(fig)
    return True


def save_change_figure(result_dir: Path, figure_dir: Path, ext: str = "png"):
    rows = read_history_csv(result_dir / "optimization_history.csv")
    if not rows:
        return False
    x = [r.get("Iter", i) for i, r in enumerate(rows)]
    change = [math.nan if r.get("Iter", i) == 0 else r.get("change", math.nan) for i, r in enumerate(rows)]
    rms = [math.nan if r.get("Iter", i) == 0 else r.get("rms_change", math.nan) for i, r in enumerate(rows)]
    return save_line_figure(
        figure_dir / f"design_change_history.{ext}",
        "Design Variable Change History",
        [("Max change", x, change), ("RMS change", x, rms)],
        "Change",
    )


def save_matrix_line_figure(result_dir: Path, figure_dir: Path, prefix: str, field: str, title: str, ylabel: str, ext: str = "png"):
    y = flatten_matrix(read_matrix_csv(result_dir / f"{prefix}_{field}.csv"))
    if not has_values(y):
        return False
    x = list(range(len(y)))
    series = [(field, x, y)]
    threshold = flatten_matrix(read_matrix_csv(result_dir / f"{prefix}_threshold.csv"))
    if field == "residual" and has_values(threshold):
        series.append(("threshold", list(range(len(threshold))), threshold))
    if field == "error":
        basis_error = flatten_matrix(read_matrix_csv(result_dir / f"{prefix}_basis_error.csv"))
        if has_values(basis_error):
            series.append(("basis_error", list(range(len(basis_error))), basis_error))
    return save_line_figure(figure_dir / f"{prefix}_{field}_history.{ext}", title, series, ylabel)


def save_mode_figure(result_dir: Path, figure_dir: Path, prefix: str, title: str, ext: str = "png"):
    mode = read_matrix_csv(result_dir / f"{prefix}_fom_used.csv")
    points_x = []
    points_y = []
    for i, row in enumerate(mode):
        for j, v in enumerate(row):
            if math.isfinite(v) and abs(v - 1.0) < 0.5:
                points_x.append(i)
                points_y.append(j)
    if not points_x:
        return False

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
    ax.scatter(points_x, points_y, s=15, marker="x", color=(0.85, 0.33, 0.10), label="FOM")
    ax.set_title(title)
    ax.set_xlabel("Optimization iteration")
    ax.set_ylabel("Time step")
    ax.set_xlim(0, max(1, len(mode) - 1))
    ax.set_ylim(0, max(1, max((len(row) for row in mode), default=1) - 1))
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / f"{prefix}_fom_mode_history.{ext}")
    plt.close(fig)
    return True


def save_density_snapshot(case_dir: Path, figure_dir: Path, ext: str):
    latest_density = case_dir / "latest_density.png"
    if not latest_density.exists():
        return False
    output_path = figure_dir / f"density_latest.{ext}"
    if ext == "png":
        shutil.copy2(latest_density, output_path)
    else:
        with Image.open(latest_density) as img:
            img.convert("RGB").save(output_path)
    return True


def save_all_figures(case_dir: Path, output_dir_name: str = "figure", ext: str = "png"):
    result_dir = case_dir / "result"
    figure_dir = case_dir / output_dir_name
    figure_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    saved += int(save_density_snapshot(case_dir, figure_dir, ext))
    saved += int(save_history_figure(result_dir, figure_dir, ext))
    saved += int(save_change_figure(result_dir, figure_dir, ext))
    saved += int(save_matrix_line_figure(result_dir, figure_dir, "forward", "residual", "Forward ROM Residual", "Residual", ext))
    saved += int(save_matrix_line_figure(result_dir, figure_dir, "obj_sens", "residual", "Sensitivity ROM Residual", "Residual", ext))
    saved += int(save_matrix_line_figure(result_dir, figure_dir, "forward", "rank", "Forward Basis Rank", "Rank", ext))
    saved += int(save_matrix_line_figure(result_dir, figure_dir, "obj_sens", "rank", "Sensitivity Basis Rank", "Rank", ext))
    saved += int(save_mode_figure(result_dir, figure_dir, "forward", "Forward FOM Mode", ext))
    saved += int(save_mode_figure(result_dir, figure_dir, "obj_sens", "Sensitivity FOM Mode", ext))
    saved += int(save_matrix_line_figure(result_dir, figure_dir, "obj_sens", "error", "Objective Sensitivity ROM-FOM Solution Error", "Error", ext))
    return saved


def serve(directory: Path, port: int):
    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(*args, directory=str(directory), **kwargs)
    with socketserver.TCPServer(("0.0.0.0", port), handler) as httpd:
        httpd.serve_forever()


def normalize_case_dir(path_text: str) -> Path:
    path_text = path_text.strip().strip('"').strip("'")
    if len(path_text) >= 3 and path_text[1] == ":" and path_text[2] in ("\\", "/"):
        drive = path_text[0].lower()
        tail = path_text[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{tail}")
    return Path(path_text)


def has_case_outputs(case_dir: Path) -> bool:
    return (case_dir / "result").is_dir()


def child_case_dirs(parent_dir: Path):
    if not parent_dir.is_dir():
        return []
    return sorted(p.parent for p in parent_dir.rglob("result") if p.is_dir())


def save_case_pdfs_once(case_dir: Path, scale: int) -> int:
    density_dir = case_dir / "density"
    last_seen, updated = convert_latest(density_dir, case_dir, scale, None)
    saved = save_all_figures(case_dir, "pdf", "pdf")
    if saved:
        print(f"Saved {saved} PDF figure(s) to {case_dir / 'pdf'}")
    if updated and last_seen:
        print(f"Updated latest_density.png from {last_seen.name}")
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch stage2 optimization CSV files and expose live plots.")
    parser.add_argument("case_dir", nargs="?", default=DEFAULT_CASE_DIR,
                        help="Case directory containing the .conf file, result/, and density/.")
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE, help="Nearest-neighbor density display scale factor.")
    parser.add_argument("--refresh", type=float, default=0.5, help="Refresh interval in seconds.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="HTTP port for browser view.")
    parser.add_argument("--once", action="store_true", help="Convert latest density and write HTML once.")
    parser.add_argument("--no-server", action="store_true", help="Do not start the HTTP server.")
    parser.add_argument("--no-save-figures", action="store_true", help="Do not export PNG figures to case_dir/figure.")
    parser.add_argument("--save-pdfs", action="store_true", help="Export PDF figures to case_dir/pdf.")
    parser.add_argument("--figure-refresh", type=float, default=5.0, help="Figure export interval in seconds.")
    args = parser.parse_args()

    case_dir = normalize_case_dir(args.case_dir)
    if args.save_pdfs and args.once:
        if has_case_outputs(case_dir):
            saved = save_case_pdfs_once(case_dir, args.scale)
            return 0 if saved else 1
        cases = child_case_dirs(case_dir)
        if not cases:
            print(f"Error: result directory not found under: {case_dir}")
            print("Please select a case folder, or a parent folder containing case folders such as FOM and ROM_*.") 
            return 2
        total = 0
        for child in cases:
            total += save_case_pdfs_once(child, args.scale)
        print(f"Saved {total} PDF figure(s) from {len(cases)} case folder(s).")
        return 0

    result_dir = case_dir / "result"
    density_dir = case_dir / "density"
    density_dir.mkdir(parents=True, exist_ok=True)
    html = HTML.replace("REFRESH_MS", str(max(100, int(args.refresh * 1000))))
    html = html.replace("WATCH_DIR", str(case_dir))
    (case_dir / "index.html").write_text(html, encoding="utf-8")

    last_seen, updated = convert_latest(density_dir, case_dir, args.scale, None)
    last_figure_save = 0.0
    if not args.no_save_figures:
        saved = save_all_figures(case_dir)
        last_figure_save = time.time()
        if saved:
            print(f"Saved {saved} figure(s) to {case_dir / 'figure'}")
    if args.save_pdfs:
        saved = save_all_figures(case_dir, "pdf", "pdf")
        if saved:
            print(f"Saved {saved} PDF figure(s) to {case_dir / 'pdf'}")
    if args.once:
        print(f"Wrote {case_dir / 'index.html'}")
        if updated and last_seen:
            print(f"Updated latest_density.png from {last_seen.name}")
        return 0

    if not args.no_server:
        thread = threading.Thread(target=serve, args=(case_dir, args.port), daemon=True)
        thread.start()
        print(f"Open http://localhost:{args.port}/index.html")
    print(f"Watching {case_dir}")

    while True:
        last_seen, updated = convert_latest(density_dir, case_dir, args.scale, last_seen)
        if updated and last_seen:
            print(f"Updated latest_density.png from {last_seen.name}", flush=True)
        if not args.no_save_figures and time.time() - last_figure_save >= max(1.0, args.figure_refresh):
            saved = save_all_figures(case_dir)
            last_figure_save = time.time()
            if saved:
                print(f"Saved {saved} figure(s) to {case_dir / 'figure'}", flush=True)
        time.sleep(args.refresh)


if __name__ == "__main__":
    raise SystemExit(main())
