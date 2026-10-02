# code_val — root-system dataset generator (AI3RSS)

Grows *Picea abies* root systems in CPlantBox inside a synthetic urban scene — layered soil,
planting pit, buried utility networks, joint leaks — and turns them into voxel grids a 3D U-Net
can be trained on: **where a structural root is likely to be**, given the soil and the street.

Two generators live here side by side. **Do not refactor v1 while working on v3**: v1 is frozen
on purpose and must keep running. v2 is no longer a generator but a dataset on disk: `generator/`
has evolved into v3 and does not reproduce it (see the v2 column).

| | **v1** | **v2** (on disk, 2026-09-22) | **v3** |
|---|---|---|---|
| Entry point | `build_dataset.py` | — (`build_dataset_v2_hybrid.py` derives from it) | `build_dataset_v3.py` |
| Scene code | `dataset/` + `obstacles/` | `generator/` at `1b52cfb` | `generator/` + `network/` + `hybrid/` |
| Soil water | volumetric θ | volumetric θ → ψ → sigmoid stress | share of the available water capacity |
| Soil | one profile per scene | mosaic of 2–4 archetypes | mosaic of 1–4 archetypes |
| Soil hooks | `f_se` | `f_se` | `f_se` and `f_sa` |
| Obstacles | 4 hand-made families, axis-aligned pipes | 0–10 real networks, any bearing | same as v2 |
| Design | 360 configs × 50 runs | 2 000 scenes × 20 runs | 2 000 scenes × 20 runs |
| Target | centreline occupancy | radius **and** centreline, plus volume | same, plus *major* / *minor* diameter classes |
| Input channels | 9 | 15 (17 in `dataset_v2_hybrid/`) | 19 = 15 baseline + 4 hybrid |
| Metadata | descriptor | descriptor | descriptor + a `site_request` per scene |

---

## The pipeline, step by step (v3)

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
   20 seeds per scene. `f_se` scales elongation at every apex, `f_sa` narrows the insertion angle
   of laterals born in hard soil, the interpolated water field drives hydrotropism on subtypes 5
   and 6, and the pipe walls are an SDF difference.
7. **Correct the radii** — `secondary_growth/`
   CPlantBox freezes a root's radius at birth. The Murray profile, anchored on the age law at the
   collar, restores a tapered, age-dependent radius per node.
8. **Voxelise** — `generator/voxelise.py`
   Centreline and true-radius occupancy, each split into *major* (diameter ≥ 2 cm) and *minor*,
   plus an accumulated root-volume channel.
9. **Assemble and write** — `generator/tensors.py`, `generator/writer.py`
   19 input channels, `prior_v2`, a FiLM conditioning vector, the `.npz`, the `.json` descriptor,
   a `.vti` for ParaView, and the scene rewritten in the inference contract
   (`generator/site_request.py`, `site_request.schema.json`).

The v1 path is the same skeleton with fewer steps: `dataset/configuration.py` →
`dataset/example.py` → `dataset/occupancy.py` → `dataset/tensors.py` → `dataset/writer.py`.

---

## Directory map

| Folder | What lives there | README |
|---|---|---|
| `dataset/` | v1 example builder: configurations, Monte Carlo, 9 channels, writer | [dataset/README.md](dataset/README.md) |
| `generator/` | v3 scene builder: scene draw, environment, voxelisation by diameter class, 19 channels, site request, writer | [generator/README.md](generator/README.md) |
| `hybrid/` | what the hybrid model adds: 4 channels, `prior_v2`, mask zones, and the measures that decide phases B and C | [hybrid/README.md](hybrid/README.md) |
| `network/` | v2/v3 buried networks: free-orientation pipes, trenches, coverage, walls, leaks | [network/README.md](network/README.md) |
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
| `paraview.py` | writes any dict of grids as one binary `.vti`, and reads it back |
| `versions.py` | the current git commit, stamped into every file written |
| `site_request.schema.json` | the inference contract (`site_request/v2`), with the training ranges; `site_request.example.json` |
| `apply_secondary_growth.py` | CLI: re-radius an existing `.vtp` |
| `visualise_example.py` | CLI: one self-contained HTML page per v1 example |
| `build_dataset.py`, `build_dataset_v3.py` | CLI, parallelisation, resumability |
| `build_dataset_v2_hybrid.py` | CLI: dataset v2 → 17 hybrid channels + `prior_v2`, no simulation |
| `measure_hybrid_readiness.py` | CLI: Dice of the priors and target mass under the masks, on any hybrid-format folder |
| `measure_training_ranges.py` | CLI: validates every site request of a dataset and rewrites the contract's training ranges |

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
python3 code_val/build_dataset_v3.py --pilot           # 20 scenes spread over the design
python3 code_val/build_dataset_v3.py                   # the 2000 scenes, about 1 h 15 on 23 workers, 20 GB
python3 code_val/build_dataset_v3.py --scenes 50 --runs 5 --vtp
python3 code_val/build_dataset.py --pilot              # v1, 5 configurations
python3 code_val/sandbox/sandbox_v2.py --hydro         # one scene per archetype, to look at
```

The three analysis CLIs need only numpy, scipy and jsonschema, and run on the host:

```bash
python code_val/measure_training_ranges.py                         # after a v3 run
python code_val/measure_hybrid_readiness.py --dataset code_val/dataset_v3
python code_val/build_dataset_v2_hybrid.py                         # once, for the v2 files
```

Both generators are **resumable**: a scene already on disk is skipped. Both write a `design.json`
on the first run and **refuse a re-run with different settings** — a scene identifier only means
something inside one design, so mixing designs in one folder would silently produce incompatible
examples.

Without `--workers` the pool uses `os.cpu_count() - 1` **as seen inside the container**.

---

## Output

`dataset_v3/`, or `pilot_v3/` with `--pilot`. Four files per scene:

| File | Contents |
|---|---|
| `scene_00001.npz` | `inputs` (19, 34, 64, 64) · `channel_names` · `film` · `prior_v2` · `target` (radius), `target_centreline`, `target_major`, `target_minor` · `root_volume` · six `hit_count`s · four packed run stacks · the first run's polylines |
| `scene_00001.json` | the whole configuration, the grid, the three channel lists, the secondary-growth parameters, the git commit, the diagnostics |
| `scene_00001.vti` | 14 voxel fields for ParaView, to open next to the `.vtp` |
| `site_request/scene_00001.json` | the scene as a user would have entered it, valid against `site_request.schema.json` |

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
- **No surface pavement since v2**: what reaches the roots is the excavation, not the slab above it.
- **Penetration resistance comes from the archetypes**, not from texture and density: the
  pedotransfer the contract relies on when `penetrometer_mpa` is null is not in the generator.
- Anything narrower than that is documented in the package README that owns it.
