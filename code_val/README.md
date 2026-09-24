# code_val — root-system dataset generator (AI3RSS)

Grows *Picea abies* root systems in CPlantBox inside a synthetic urban scene — layered soil,
planting pit, buried utility networks, joint leaks — and turns them into voxel grids a 3D U-Net
can be trained on: **where a structural root is likely to be**, given the soil and the street.

Two generators live here side by side. **Do not refactor v1 while working on v2**: v1 is frozen
on purpose and must keep running.

| | **v1** | **v2** |
|---|---|---|
| Entry point | `build_dataset.py` | `build_dataset_v2.py` |
| Scene code | `dataset/` + `obstacles/` | `generator/` + `network/` |
| Soil | one profile per scene | mosaic of 2–4 archetypes |
| Obstacles | 4 hand-made families, axis-aligned pipes | 0–10 real networks, any bearing, elbows, tees |
| Design | 10 archetypes × 4 families × 9 draws = 360 configs × 50 runs | 2 000 scenes × 20 runs |
| Target | centreline occupancy | radius occupancy **and** centreline, plus a volume channel |
| Input channels | 9 | 15 |

---

## The pipeline, step by step (v2)

Each step is one module; read the code in this order.

1. **Draw the scene** — `generator/scene.py`
   One `SceneConfiguration` per identifier, from a master seed split into five independent
   streams (tree, pit, soil, street, noise) so changing one domain never reshuffles the others.
   It draws the tree, the pit, the soil mosaic (`soil/mosaic.py`), the street
   (`network/street.py`) and its joint leaks (`network/leaks.py`).
2. **Build the environment** — `generator/environment.py`
   The order is the argument and cannot be permuted: pit → engineered fills → leaks → Rosetta on
   the composited field → geometry last. Produces the five soil channels, the matric potential,
   `f_se`, the water scale, the fraction fields and the two CPlantBox lookups.
3. **Turn the street into solids** — `network/shapes.py` + `network/coverage.py`
   Lumen, sand surround, loose backfill under the pit, recompacted trench backfill: a priority
   list consumed against a decreasing budget, which carves the rings without ever describing one.
   Each solid is rasterised as a *volume fraction*, not a mask.
4. **Write the materials into the soil** — `soil/fill.py`, then `soil/leak.py`
   Fills blend by fraction; a leak wets the soil *and* softens its penetration resistance.
5. **Soil → root physics** — `soil/hydraulics.py` (Rosetta → van Genuchten → θ and ψ) then
   `soil/growth_response.py` (Bengough impedance × water stress = `f_se`). Soil water is authored
   as a share of the available water capacity, not as a volumetric moisture: see
   [soil/README.md](soil/README.md).
6. **Grow** — `tree/growth.py`
   20 seeds per scene. `f_se` scales elongation at every apex, the interpolated water field
   drives hydrotropism on subtypes 5 and 6, and the pipe walls are an SDF difference.
7. **Correct the radii** — `secondary_growth/`
   CPlantBox freezes a root's radius at birth. The Murray profile, anchored on the age law at the
   collar, restores a tapered, age-dependent radius per node.
8. **Voxelise** — `generator/voxelise.py`
   Two occupancy grids (centreline and true radius) plus an accumulated root-volume channel.
9. **Assemble and write** — `generator/tensors.py`, `generator/writer.py`
   15 input channels, a FiLM conditioning vector, the `.npz`, the `.json` descriptor and a `.vti`
   for ParaView.

The v1 path is the same skeleton with fewer steps: `dataset/configuration.py` →
`dataset/example.py` → `dataset/occupancy.py` → `dataset/tensors.py` → `dataset/writer.py`.

---

## Directory map

| Folder | What lives there | README |
|---|---|---|
| `dataset/` | v1 example builder: configurations, Monte Carlo, 9 channels, writer | [dataset/README.md](dataset/README.md) |
| `generator/` | v2 scene builder: scene draw, environment, voxelisation, 15 channels, writer | [generator/README.md](generator/README.md) |
| `network/` | v2 buried networks: free-orientation pipes, trenches, coverage, walls, leaks | [network/README.md](network/README.md) |
| `obstacles/` | v1 obstacle families, **and** the axis-aligned street prototype `network/` replaced | [obstacles/README.md](obstacles/README.md) |
| `sandbox/` | throwaway experiments that measure one mechanism at a time | [sandbox/README.md](sandbox/README.md) |
| `secondary_growth/` | post-hoc root thickening, the radius CPlantBox will not give you | [secondary_growth/README.md](secondary_growth/README.md) |
| `soil/` | archetypes, mosaic, noise, pit, fills, leaks, Rosetta, `f_se` | [soil/README.md](soil/README.md) |
| `tree/` | tree attributes, and the CPlantBox growth call itself | [tree/README.md](tree/README.md) |
| `parameters/` | `Picea_Abies_combined_v6.xml`, 7 root subtypes | — |

Shared, at the root:

| File | Responsibility |
|---|---|
| `grid.py` | the voxel grid — 34 × 64 × 64, cell 15 × 15 × 9 cm, extent ±480 × ±480 cm and 306 cm deep, depth index 0 at the surface and increasing downward |
| `paraview.py` | writes any dict of grids as one binary `.vti` |
| `apply_secondary_growth.py` | CLI: re-radius an existing `.vtp` |
| `visualise_example.py` | CLI: one self-contained HTML page per v1 example |
| `build_dataset.py`, `build_dataset_v2.py` | CLI, parallelisation, resumability |

---

## Running

Build the image once, from the repository root:

```bash
docker build -t cplantbox-image .
```

Open a container with the repository mounted:

```bash
docker run --rm -it \
  -v "C:\Users\ValentinMarchand\Dev\CPlantBox_for_TreeCity:/workspaces/CPlantBox_for_TreeCity" \
  -w /workspaces/CPlantBox_for_TreeCity \
  cplantbox-image bash
```

Then:

```bash
python3 code_val/build_dataset_v2.py --pilot           # 20 scenes spread over the design
python3 code_val/build_dataset_v2.py                   # the 2000 scenes
python3 code_val/build_dataset_v2.py --scenes 50 --runs 5 --vtp
python3 code_val/build_dataset.py --pilot              # v1, 5 configurations
python3 code_val/sandbox/sandbox_v2.py --hydro         # one scene per archetype, to look at
```

Both generators are **resumable**: a scene already on disk is skipped. Both write a `design.json`
on the first run and **refuse a re-run with different settings** — a scene identifier only means
something inside one design, so mixing designs in one folder would silently produce incompatible
examples.

Without `--workers` the pool uses `os.cpu_count() - 1` **as seen inside the container**.

---

## Output

`dataset_v2_runs/`, or `pilot_v2/` with `--pilot`. Three files per scene:

| File | Contents |
|---|---|
| `scene_00001.npz` | `inputs` (15, 34, 64, 64) · `film` · `target` (radius) · `target_centreline` · `root_volume` · both `hit_count`s · both packed run stacks · the first run's polylines |
| `scene_00001.json` | the whole configuration, the grid, the secondary-growth parameters, the git commit, the diagnostics |
| `scene_00001.vti` | 11 voxel fields for ParaView, to open next to the `.vtp` |

The `.json` is written **last**, through a temporary file: its presence is what certifies that a
scene is complete. Individual Monte Carlo runs are kept bit-packed, so a probability map can be
recomputed over any subset of runs without simulating again.

---

## Key numbers

| | |
|---|---|
| Grid | 34 × 64 × 64 voxels, 15 × 15 × 9 cm each |
| Structural subtypes | 1, 2, 4, 6, 7 — subtypes 3 and 5 are fine roots, excluded from the target |
| Hydrotropic subtypes | 5 and 6, `tropismT = 3`, strength 0.10 |
| Time step | 10 days |
| Tree age | drawn uniformly in [5, 120] years |
| Largest `dx` in the XML | 3.0 cm — the 4 cm pipe exclusion margin has to exceed it, and an assertion checks that at startup |

## Limits that apply to the whole pipeline

- **Beyond 60 years the XML is extrapolated** (`simulationTime` = 21 900 days), and it carries no
  botanical citation: it was calibrated on visual plausibility.
- **No dynamic water balance.** The matric potential is static; a climate would only make sense
  as a factor of the experimental design, with its own FiLM slots.
- **The `prior` channel is a strong radial prior** — exactly what a network copies while ignoring
  the soil. Train once with it and once without.
- **No surface pavement in v2**: what reaches the roots is the excavation, not the slab above it.
- Anything narrower than that is documented in the package README that owns it.
