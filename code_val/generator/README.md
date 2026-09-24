# generator — the v2 scene builder

One scene, from its random seed to the three files on disk. This is where the v2 pipeline is
assembled; the physics lives in [`soil/`](../soil/README.md), the street in
[`network/`](../network/README.md), the growth in [`tree/`](../tree/README.md) and the radii in
[`secondary_growth/`](../secondary_growth/README.md).

Driven by `../build_dataset_v2.py`.

## Step by step

`build_dataset_v2.py` draws all the configurations up front, then hands one to each worker.
Per scene:

1. **`scene.py::draw_scene` — everything random, decided here and nowhere else.**
   The master seed is split into five independent streams — `TREE`, `PIT`, `SOIL`, `STREET`,
   `NOISE` — through `default_rng([master_seed, identifier, domain])`. Re-tuning one domain
   therefore never reshuffles the others: the same scene number keeps the same tree when you
   change the street code. Draws the tree, the pit, the soil mosaic, the street and its leaks.
2. **`environment.py::build_environment` — the soil the roots will actually feel.**
   The order carries the reasoning and must not be permuted:

   | Order | Why |
   |---|---|
   | pit before fills | a network crosses the planting pit like any other soil |
   | fills before leaks | a fill arrives with its resistance and moisture already consistent; it must not then be rescaled by the moisture → resistance coupling a leak triggers |
   | Rosetta on the composited field | not on the original profile — the fills and the leak have already moved texture and moisture |
   | geometry last | `grow_plant` calls `setSoil()` then `initialize()`, and `initialize()` builds the tropism callbacks from whatever soil the plant holds at that instant |

   The returned `SceneEnvironment` is kept alive for the whole Monte Carlo loop: **CPlantBox
   holds the geometry by `weak_ptr`**, and a freed geometry stops confining roots with no error.
3. **`build.py::grow_once`, once per seed (20 by default).**
   Grow → `polylines.py` reads the live plant → `secondary_growth` recomputes a radius per node →
   `voxelise.py` stamps the two occupancy grids and accumulates the volume →
   `network/intrusion.py` checks nothing ended up inside a pipe.
4. **`build.py::build_scene` — the ensemble.** Averages the runs into `target_radius` and
   `target_centreline`, packs every individual run bit by bit, keeps the first run's polylines,
   and collects the diagnostics.
5. **`tensors.py`** builds the 15 input channels and the FiLM vector.
6. **`writer.py`** writes `.npz`, `.vti` and — last — `.json`.

## Who does what

| File | Responsibility |
|---|---|
| `scene.py` | the experimental design and every seed; `SceneConfiguration` is the complete description of a scene |
| `environment.py` | soil + street → the fields, the two CPlantBox lookups and the growth geometry |
| `polylines.py` | reads the polylines out of a *live* plant, without a round trip through a `.vtp` |
| `voxelise.py` | polylines → centreline grid, radius grid, root-volume grid |
| `tensors.py` | the 15 input channels and the FiLM vector |
| `build.py` | the loop above, the ensemble, the diagnostics |
| `writer.py` | the three files, atomically |

## The two targets

| Target | What it is |
|---|---|
| `target_centreline` | v1's definition: a voxel is taken if a root axis passes through it |
| `target` (radius) | a voxel is taken as soon as a root thick enough passes close enough to bite into it |

The radius test is exact — does a sphere of radius *r* centred on the sample intersect the voxel
box — so a root lying on a face spills into the neighbour while the same root through the centre
does not. A morphological dilation per radius class would instead inflate every marked voxel
identically, after throwing away the sub-voxel position. `RADIUS_BIN_EDGES_CM` only groups
samples so one offset table can serve a batch; the per-sample radius is still what is tested.

`root_volume` accumulates π·r²·length over **all** roots, fine ones included: it is a biomass
channel, and fine roots carry biomass.

## The 15 input channels

`penetrometer`, `bulk_density`, `moisture`, `sand_fraction`, `clay_fraction`, `water_scale`,
`growth_scale`, `pipe_fraction`, `pipe_clearance`, `backfill_fraction`, `surround_fraction`,
`leak_plume`, `pit_mask`, `depth`, `prior`.

Compared with v1: the pavement mask is gone (there is no surface slab any more) and the obstacle
distance is gone (it assumed an infinite axis-aligned pipe). The binary pipe mask became a
**fraction** — a 15 cm pipe has a 177 cm² section against 135 cm² for a voxel column, so no voxel
is ever full and a mask could only draw a staircase.

`water_scale` and `growth_scale` are deliberately redundant with the first five channels: they
are the exact fields CPlantBox read, so the network does not have to rediscover the Rosetta →
van Genuchten → Bengough chain to reach the label.

FiLM is 3 species slots + 3 architecture slots + 7 scalars, ending with the utility count.

## Key numbers

| | |
|---|---|
| Scenes | `SCENE_COUNT = 2000` |
| Runs per scene | `MONTE_CARLO_RUNS = 20` |
| Master seed | 20260922 |
| Sampling along a segment | 1 cm |
| VTP written | the first run only, and only with `--vtp` or `--pilot` |

## Choices and limits

- **`is_complete()` tests the `.json`, not the `.npz`.** v1 tested the archive, so an archive
  truncated by an interruption looked finished. Both the `.npz` and the `.json` are now written
  to a temporary file and renamed.
- **The descriptor is a mirror of the dataclass.** Everything drawn appears in the `.json`
  because it appears in `SceneConfiguration`, with no separate list to keep in sync.
- **`build_final_radii` is the cheap door.** It computes only what the fusion consumes; the pipe
  model and da Vinci models are extra full traversals and are only computed by the
  `apply_secondary_growth.py` CLI, which writes them as calibration witnesses.
- **Intrusions are reported, not asserted.** `diagnostics["intrusions"]` carries the penetration
  depth per pipe; a graze and a three-centimetre push are different bugs.
- **The scene varies here, not in the sandbox.** `sandbox/sandbox_v2.py` deliberately freezes the
  street so a difference between two runs can only come from the soil.
