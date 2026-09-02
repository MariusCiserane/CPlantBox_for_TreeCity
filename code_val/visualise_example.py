"""Write a self-contained HTML page to inspect one dataset example.

No external dependency: the grids are quantised to one byte per voxel and embedded
in the page, which draws three annotated 2D views and a rotatable 3D point cloud.
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

from dataset.tensors import CHANNEL_NAMES
from grid import VoxelGrid
from obstacles.masks import OBSTACLE_DISTANCE_CLIP_CM
from soil.archetypes import (
    BULK_DENSITY_RANGE_G_CM3,
    MOISTURE_RANGE_M3_M3,
    PENETROMETER_RANGE_MPA,
)

QUANTISATION_LEVELS = 255
PALETTE = ("#10163a", "#1f4b99", "#2ba0a0", "#c8d92a", "#fdfbe4")

TARGET_FIELD = "target — occupancy probability"
BINARY_FIELD = "single run — binary occupancy"
UNITLESS = ("", 0.0, 1.0)
CHANNEL_UNITS = {
    "penetrometer": ("MPa", *PENETROMETER_RANGE_MPA),
    "bulk_density": ("g/cm³", *BULK_DENSITY_RANGE_G_CM3),
    "moisture": ("m³/m³", 0.0, MOISTURE_RANGE_M3_M3[1]),
    "sand_fraction": UNITLESS,
    "clay_fraction": UNITLESS,
    "pipe_mask": UNITLESS,
    "pavement_mask": UNITLESS,
    "obstacle_distance": ("cm", 0.0, OBSTACLE_DISTANCE_CLIP_CM),
    "prior": UNITLESS,
}

PAGE_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
 body {font-family: system-ui, sans-serif; margin: 24px; background: #fafafa; color: #222;}
 h1 {font-size: 18px; margin: 0 0 4px;}
 .meta {color: #666; font-size: 13px; margin-bottom: 18px;}
 .controls {display: flex; gap: 22px; align-items: center; flex-wrap: wrap; margin-bottom: 16px;}
 .legend {display: flex; align-items: center; gap: 8px; font-size: 12px; color: #555;}
 .bar {width: 190px; height: 12px; border: 1px solid #ccc; background: __GRADIENT__;}
 .views {display: flex; gap: 24px; flex-wrap: wrap; align-items: flex-start;}
 figure {margin: 0;}
 figcaption {font-size: 12px; color: #666; margin-top: 4px; max-width: 460px;}
 canvas {background: #fff; border: 1px solid #ddd;}
 #scene {cursor: grab;}
 #scene:active {cursor: grabbing;}
 select, input {font-size: 14px;}
 code {background: #eee; padding: 1px 4px; border-radius: 3px;}
</style>
<h1>__TITLE__</h1>
<div class="meta">__META__</div>

<div class="controls">
  <label>field <select id="field">__OPTIONS__</select></label>
  <label>depth layer <input id="depth" type="range" min="0" max="__MAX_DEPTH__" value="0"></label>
  <span id="depthLabel"></span>
  <label>3D minimum value <input id="threshold" type="range" min="0" max="100" value="2"></label>
  <span id="thresholdLabel"></span>
</div>
<div class="controls legend">
  <span id="lowLabel"></span><div class="bar"></div><span id="highLabel"></span>
  <span id="countLabel"></span>
</div>

<div class="views">
  <figure><canvas id="slice"></canvas>
    <figcaption>Horizontal slice at the selected depth, seen from above. Trunk at the centre.</figcaption></figure>
  <figure><canvas id="top"></canvas>
    <figcaption>Maximum over all depths — the horizontal footprint.</figcaption></figure>
  <figure><canvas id="side"></canvas>
    <figcaption>Maximum over Y — the vertical profile, surface at the top.</figcaption></figure>
  <figure><canvas id="scene"></canvas>
    <figcaption>3D voxels above the minimum value. Opacity follows the value, so the whole
      cloud stays visible while the strongest voxels stand out &mdash; slide the minimum up to
      keep only the core. Drag to rotate, scroll to zoom.
      Box is __SPAN_CM__ &times; __SPAN_CM__ &times; __DEPTH_CM__ cm.</figcaption></figure>
</div>

<script>
const FIELDS = __FIELDS__, UNITS = __UNITS__, PALETTE = __PALETTE__;
const DEPTH = __DEPTH_CELLS__, ROWS = __ROWS__, COLUMNS = __COLUMNS__;
const CELL_WIDTH_CM = __CELL_WIDTH_CM__, CELL_HEIGHT_CM = __CELL_HEIGHT_CM__;
const HALF_SPAN_CM = __HALF_SPAN_CM__, DEPTH_CM = __DEPTH_CM__;
const MARGIN = 46, PLOT = 320, TICKS = 5, SIDE_HEIGHT = 150;
const MAX_SCENE_VOXELS = 30000, MINIMUM_ALPHA = 0.10;
const EXAGGERATION = (SIDE_HEIGHT / DEPTH_CM) / (PLOT / (2 * HALF_SPAN_CM));

function decode(encoded) {
  const binary = atob(encoded), bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function colourOf(value) {
  const position = (value / 255) * (PALETTE.length - 1);
  const low = Math.floor(position), high = Math.min(low + 1, PALETTE.length - 1);
  const blend = position - low;
  return PALETTE[low].map((channel, i) => channel + blend * (PALETTE[high][i] - channel));
}

function drawFrame(context, width, height, title, xAxis, yAxis) {
  context.clearRect(0, 0, width + 2 * MARGIN, height + 2 * MARGIN);
  context.fillStyle = "#222"; context.font = "13px system-ui"; context.textAlign = "left";
  context.fillText(title, MARGIN, MARGIN - 24);
  context.strokeStyle = "#999"; context.strokeRect(MARGIN, MARGIN, width, height);
  context.fillStyle = "#666"; context.font = "11px system-ui";

  for (let tick = 0; tick <= TICKS; tick++) {
    const fraction = tick / TICKS;
    const x = MARGIN + fraction * width, y = MARGIN + fraction * height;
    context.beginPath();
    context.moveTo(x, MARGIN + height); context.lineTo(x, MARGIN + height + 4);
    context.moveTo(MARGIN - 4, y); context.lineTo(MARGIN, y);
    context.stroke();
    context.textAlign = "center";
    context.fillText(Math.round(xAxis.from + fraction * (xAxis.to - xAxis.from)), x, MARGIN + height + 16);
    context.textAlign = "right";
    context.fillText(Math.round(yAxis.from + fraction * (yAxis.to - yAxis.from)), MARGIN - 7, y + 4);
  }
  context.textAlign = "center"; context.fillStyle = "#444"; context.font = "12px system-ui";
  context.fillText(xAxis.label, MARGIN + width / 2, MARGIN + height + 34);
  context.save();
  context.translate(MARGIN - 32, MARGIN + height / 2); context.rotate(-Math.PI / 2);
  context.fillText(yAxis.label, 0, 0);
  context.restore();
}

function paintField(canvas, cells, pixels, title, xAxis, yAxis, valueAt) {
  canvas.width = pixels.width + 2 * MARGIN;
  canvas.height = pixels.height + 2 * MARGIN;
  const context = canvas.getContext("2d");
  drawFrame(context, pixels.width, pixels.height, title, xAxis, yAxis);

  const buffer = document.createElement("canvas");
  buffer.width = cells.width; buffer.height = cells.height;
  const image = buffer.getContext("2d").createImageData(cells.width, cells.height);
  for (let y = 0; y < cells.height; y++) {
    for (let x = 0; x < cells.width; x++) {
      const [r, g, b] = colourOf(valueAt(x, y)), offset = 4 * (y * cells.width + x);
      image.data[offset] = r; image.data[offset + 1] = g; image.data[offset + 2] = b;
      image.data[offset + 3] = 255;
    }
  }
  buffer.getContext("2d").putImageData(image, 0, 0);
  context.imageSmoothingEnabled = false;
  context.drawImage(buffer, MARGIN, MARGIN, pixels.width, pixels.height);
}

let azimuth = 0.6, elevation = 0.45, zoom = 1.0, dragging = null;

function makeProjector() {
  const ca = Math.cos(azimuth), sa = Math.sin(azimuth);
  const ce = Math.cos(elevation), se = Math.sin(elevation);
  const right = [-sa, ca, 0], up = [-ca * se, -sa * se, ce], forward = [ca * ce, sa * ce, se];
  const dot = (a, p) => a[0] * p[0] + a[1] * p[1] + a[2] * p[2];
  return (point) => [dot(right, point), -dot(up, point), dot(forward, point)];
}

function collectVoxels(grid, threshold) {
  const voxels = [];
  for (let k = 0; k < DEPTH; k++) {
    for (let j = 0; j < ROWS; j++) {
      for (let i = 0; i < COLUMNS; i++) {
        const value = grid[(k * ROWS + j) * COLUMNS + i];
        if (value < threshold) continue;
        voxels.push([(i + 0.5) * CELL_WIDTH_CM - HALF_SPAN_CM,
                     (j + 0.5) * CELL_WIDTH_CM - HALF_SPAN_CM,
                     -(k + 0.5) * CELL_HEIGHT_CM, value]);
      }
    }
  }
  return voxels;
}

function boxEdges() {
  const s = HALF_SPAN_CM, d = -DEPTH_CM, corners = [];
  for (const x of [-s, s]) for (const y of [-s, s]) for (const z of [0, d]) corners.push([x, y, z]);
  const edges = [];
  for (let a = 0; a < corners.length; a++) {
    for (let b = a + 1; b < corners.length; b++) {
      const shared = corners[a].filter((value, index) => value === corners[b][index]).length;
      if (shared === 2) edges.push([corners[a], corners[b]]);
    }
  }
  return edges;
}

function paintScene(canvas, grid, threshold) {
  const size = 2 * (PLOT / 2 + MARGIN);
  canvas.width = size; canvas.height = size;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, size, size);

  const scale = zoom * (PLOT / (2.2 * HALF_SPAN_CM)), centre = size / 2, project = makeProjector();
  const toScreen = (point) => {
    const [x, y] = project(point);
    return [centre + x * scale, centre + y * scale];
  };

  context.strokeStyle = "#ccc";
  for (const [start, end] of boxEdges()) {
    const a = toScreen(start), b = toScreen(end);
    context.beginPath(); context.moveTo(a[0], a[1]); context.lineTo(b[0], b[1]); context.stroke();
  }

  context.fillStyle = "#444"; context.font = "12px system-ui"; context.textAlign = "center";
  for (const [label, point] of [["+X (cm)", [HALF_SPAN_CM * 1.15, 0, 0]],
                                ["+Y (cm)", [0, HALF_SPAN_CM * 1.15, 0]],
                                ["depth", [0, 0, -DEPTH_CM * 1.25]]]) {
    const [x, y] = toScreen(point);
    context.fillText(label, x, y);
  }

  let voxels = collectVoxels(grid, threshold);
  const total = voxels.length;
  if (total > MAX_SCENE_VOXELS) {
    voxels.sort((a, b) => b[3] - a[3]);
    voxels = voxels.slice(0, MAX_SCENE_VOXELS);
  }

  const drawn = voxels.map((voxel) => {
    const [x, y, depth] = project(voxel);
    return [centre + x * scale, centre + y * scale, depth, voxel[3]];
  });
  drawn.sort((a, b) => a[2] - b[2]);

  // Faint where the probability is low, solid where it is high: the whole cloud stays
  // visible while the voxels that matter stand out. Painted back to front so the
  // transparency composites in the right order.
  const side = Math.max(2, CELL_WIDTH_CM * scale);
  for (const [x, y, , value] of drawn) {
    const [r, g, b] = colourOf(value);
    const alpha = MINIMUM_ALPHA + (1 - MINIMUM_ALPHA) * (value / 255);
    context.fillStyle = `rgba(${r|0},${g|0},${b|0},${alpha.toFixed(3)})`;
    context.fillRect(x - side / 2, y - side / 2, side, side);
  }
  document.getElementById("countLabel").textContent = total > MAX_SCENE_VOXELS
    ? drawn.length.toLocaleString() + " highest-valued voxels of " + total.toLocaleString() + " above threshold"
    : drawn.length.toLocaleString() + " voxels above threshold";
}

function currentScene() {
  return [decode(FIELDS[document.getElementById("field").value]),
          Number(document.getElementById("threshold").value) / 100 * 255];
}

function renderScene() {
  paintScene(document.getElementById("scene"), ...currentScene());
}

function render() {
  const name = document.getElementById("field").value;
  const grid = decode(FIELDS[name]);
  const depth = Number(document.getElementById("depth").value);
  const threshold = Number(document.getElementById("threshold").value) / 100;
  const at = (k, j, i) => grid[(k * ROWS + j) * COLUMNS + i];
  const [unit, low, high] = UNITS[name];

  document.getElementById("depthLabel").textContent =
    "layer " + depth + " / " + (DEPTH - 1) + " — " + ((depth + 0.5) * CELL_HEIGHT_CM).toFixed(0) + " cm deep";
  document.getElementById("thresholdLabel").textContent =
    (low + threshold * (high - low)).toFixed(2) + " " + unit;
  document.getElementById("lowLabel").textContent = low.toFixed(2) + " " + unit;
  document.getElementById("highLabel").textContent = high.toFixed(2) + " " + unit;

  const square = {width: PLOT, height: PLOT};
  const profile = {width: PLOT, height: SIDE_HEIGHT};
  const acrossX = {label: "X (cm)", from: -HALF_SPAN_CM, to: HALF_SPAN_CM};
  const acrossY = {label: "Y (cm)", from: -HALF_SPAN_CM, to: HALF_SPAN_CM};
  const downward = {label: "depth (cm)", from: 0, to: DEPTH_CM};

  paintField(document.getElementById("slice"), {width: COLUMNS, height: ROWS}, square,
             "Slice at " + ((depth + 0.5) * CELL_HEIGHT_CM).toFixed(0) + " cm", acrossX, acrossY,
             (i, j) => at(depth, j, i));
  paintField(document.getElementById("top"), {width: COLUMNS, height: ROWS}, square,
             "Top view — maximum over depth", acrossX, acrossY,
             (i, j) => { let best = 0; for (let k = 0; k < DEPTH; k++) best = Math.max(best, at(k, j, i)); return best; });
  paintField(document.getElementById("side"), {width: COLUMNS, height: DEPTH}, profile,
             "Side view — maximum over Y (vertical exaggeration ×" + EXAGGERATION.toFixed(1) + ")",
             acrossX, downward,
             (i, k) => { let best = 0; for (let j = 0; j < ROWS; j++) best = Math.max(best, at(k, j, i)); return best; });
  paintScene(document.getElementById("scene"), grid, threshold * 255);
}

const scene = document.getElementById("scene");
scene.addEventListener("mousedown", (event) => { dragging = [event.clientX, event.clientY]; });
addEventListener("mouseup", () => { dragging = null; });
addEventListener("mousemove", (event) => {
  if (!dragging) return;
  azimuth += (event.clientX - dragging[0]) * 0.01;
  elevation = Math.max(-1.5, Math.min(1.5, elevation + (event.clientY - dragging[1]) * 0.01));
  dragging = [event.clientX, event.clientY];
  renderScene();
});
scene.addEventListener("wheel", (event) => {
  event.preventDefault();
  zoom = Math.max(0.4, Math.min(4.0, zoom * (event.deltaY < 0 ? 1.1 : 0.9)));
  renderScene();
});
document.getElementById("field").addEventListener("change", render);
document.getElementById("depth").addEventListener("input", render);
document.getElementById("threshold").addEventListener("input", render);
render();
</script>
"""


def load_fields(archive_path: Path, grid: VoxelGrid) -> dict[str, np.ndarray]:
    archive = np.load(archive_path)
    voxels_per_run = grid.depth_cells * grid.row_cells * grid.column_cells
    first_run = np.unpackbits(archive["occupancy_runs"])[:voxels_per_run].reshape(grid.shape)
    fields = {TARGET_FIELD: archive["target"], BINARY_FIELD: first_run}
    for name, channel in zip(CHANNEL_NAMES, archive["inputs"]):
        fields[f"input — {name}"] = channel
    return fields


def build_units(fields: dict[str, np.ndarray]) -> dict[str, tuple]:
    units = {}
    for name in fields:
        channel = name.split(" — ")[-1]
        units[name] = CHANNEL_UNITS.get(channel, UNITLESS)
    return units


def encode_field(field: np.ndarray) -> str:
    quantised = np.clip(field, 0.0, 1.0) * QUANTISATION_LEVELS
    return base64.b64encode(quantised.astype(np.uint8).tobytes()).decode("ascii")


def describe(descriptor_path: Path) -> str:
    if not descriptor_path.is_file():
        return ""
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    return (f"soil {descriptor['soil']['archetype_name']} &middot; "
            f"obstacles {descriptor['obstacles']['family']} &middot; "
            f"{descriptor['tree']['age_years']:.0f} years old &middot; "
            f"target mass {100 * descriptor['diagnostics']['mean_probability']:.3f} %")


def build_page(title: str, meta: str, fields: dict[str, np.ndarray], grid: VoxelGrid) -> str:
    palette = [[int(colour[i:i + 2], 16) for i in (1, 3, 5)] for colour in PALETTE]
    substitutions = {
        "__TITLE__": title,
        "__META__": meta,
        "__OPTIONS__": "".join(f"<option>{name}</option>" for name in fields),
        "__FIELDS__": json.dumps({name: encode_field(field) for name, field in fields.items()}),
        "__UNITS__": json.dumps(build_units(fields)),
        "__PALETTE__": json.dumps(palette),
        "__GRADIENT__": f"linear-gradient(to right, {', '.join(PALETTE)})",
        "__DEPTH_CELLS__": str(grid.depth_cells),
        "__ROWS__": str(grid.row_cells),
        "__COLUMNS__": str(grid.column_cells),
        "__MAX_DEPTH__": str(grid.depth_cells - 1),
        "__CELL_WIDTH_CM__": str(grid.cell_width_cm),
        "__CELL_HEIGHT_CM__": str(grid.cell_height_cm),
        "__HALF_SPAN_CM__": str(grid.half_width_cm),
        "__DEPTH_CM__": str(grid.depth_cm),
        "__SPAN_CM__": str(int(2 * grid.half_width_cm)),
    }
    page = PAGE_TEMPLATE
    for placeholder, value in substitutions.items():
        page = page.replace(placeholder, value)
    return page


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="an example_XXXX.npz file")
    parser.add_argument("--output", type=Path, help="HTML page to write (default: beside the .npz)")
    arguments = parser.parse_args()

    grid = VoxelGrid()
    output = arguments.output or arguments.archive.with_suffix(".html")
    page = build_page(arguments.archive.stem,
                      describe(arguments.archive.with_suffix(".json")),
                      load_fields(arguments.archive, grid), grid)
    output.write_text(page, encoding="utf-8")
    print(f"{output}  ({output.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
