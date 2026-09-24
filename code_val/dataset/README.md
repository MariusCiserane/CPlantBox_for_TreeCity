# dataset — the v1 example builder

Turns one *configuration* — a soil archetype, an obstacle family, a tree — into one training
example: several stochastic root systems grown in the same environment, voxelised and averaged
into an occupancy probability.

This is the **v1** pipeline. It is frozen on purpose: the v2 generator lives in
[`generator/`](../generator/README.md) and does not import anything from here except
`STRUCTURAL_SUBTYPES` and a few normalisation helpers.

Driven by `../build_dataset.py`.

## Step by step

One call to `build_example()` runs the whole chain. In order:

1. **`configuration.py` — decide what this example is.**
   Crosses the 10 soil archetypes with the 4 obstacle families, `DRAWS_PER_COMBINATION = 9`
   times each, so 360 configurations. Each draws its own soil profile, pit substrate, pit size,
   obstacle layout, tree attributes, and the seeds it will use. Every random number comes from
   `default_rng([MASTER_SEED, identifier])`, so a configuration is fully reproducible from its
   number alone.
2. **`example.py::build_soil_environment` — build the soil.**
   Ambient field and pit field (`soil/field.py`), pit zoning on top (`soil/pit.py`), then the
   road-base override where a pavement compacts the sub-base. Order matters: the pit is applied
   first and the obstacle override second, because **a pipe crosses the pit like anything else**
   — the pit is not a protected bubble. Then ψ (`soil/hydraulics.py`) and `f_se`
   (`soil/growth_response.py`).
3. **`example.py::build_example` — build the growth domain once.**
   The SDF (`obstacles/geometry.py`) and the `EquidistantGrid3D` lookup are built before the loop
   and kept alive through it: **CPlantBox holds the geometry by weak reference**, and a freed
   geometry stops confining roots silently.
4. **Monte Carlo loop, 50 runs.** Each seed grows a root system (`tree/growth.py`), which is
   immediately voxelised. Only the first run writes a `.vtp` (`VTP_RUNS_PER_CONFIGURATION`).
5. **`occupancy.py` — segments to grids.**
   Keep the structural subtypes, sample every segment at 1 cm, drop each sample into its voxel.
   Then average the 50 binary grids into the probability target.
6. **`tensors.py` — the network's inputs.** 9 channels, plus the FiLM conditioning vector.
7. **`writer.py` — to disk.** `.npz` for the arrays, `.json` for everything that describes them.

## Who does what

| File | Responsibility |
|---|---|
| `configuration.py` | the experimental design: archetypes × obstacle families × draws, and every seed |
| `example.py` | the chain above, plus the diagnostics |
| `occupancy.py` | segments → binary grid → probability grid |
| `tensors.py` | the 9 input channels, the FiLM vector, and the depth × radius `prior` |
| `writer.py` | `example_0001.npz` and `example_0001.json` |

## The 9 input channels

`penetrometer`, `bulk_density`, `moisture`, `sand_fraction`, `clay_fraction`, `pipe_mask`,
`pavement_mask`, `obstacle_distance`, `prior` — in that order, all normalised to roughly [0, 1]
by the physical ranges in `soil/archetypes.py`. `CHANNEL_SCHEMA = "v1"` is written into every
descriptor, so a stale `.npz` can always be identified.

The FiLM vector is 3 species slots + 3 architecture slots + 5 scalars (age, DBH, height, crown
depth, topsoil depth), each normalised by its declared range.

## Key numbers

| | |
|---|---|
| Configurations | 10 × 4 × 9 = **360** |
| Monte Carlo runs | **50** per configuration → 18 000 simulations |
| Sampling along a segment | 1 cm, at least 2 samples per segment |
| Structural subtypes | 1, 2, 4, 6, 7 |
| Seeds | `identifier × 1000 + run`, so run seeds never collide across configurations |

## Reading the diagnostics

Two different measures of rarity, not to be confused:

- **`support_voxel_fraction`** — the share of voxels touched by *at least one* run. It grows with
  the number of runs; it describes the extent of the root system, not the class imbalance.
- **`mean_probability`** — the mean of the target. Independent of the number of runs: this is the
  true positive rate the loss function sees. It has measured around 0.3 %, an order of magnitude
  under the 5 % threshold, so **the loss has to combine BCE and Dice** — a plain BCE will predict
  zero everywhere and score beautifully.

## Choices and limits

- **Centreline only.** The XML radii are one to two orders of magnitude too small (CPlantBox has
  no secondary thickening), so dilating by radius would propagate a known error into the target.
  v2 fixes this with [`secondary_growth/`](../secondary_growth/README.md) and a radius-aware
  stamp; this package deliberately was not changed.
- **Independent noise per channel** in the v1 soil, so a voxel could be dense, wet and weak at
  once. Fixed in `soil/field.py` for v2; the frozen v1 behaviour now lives in
  `sandbox/sandbox_v1.py`.
- **`prior` to watch.** A strong radial prior is exactly what a network copies while ignoring the
  soil. Train once with it and once without.
- **Only the pipe axis is masked.** `obstacle_distance` assumes an infinite axis-aligned pipe and
  cannot describe a diagonal or an elbow, which is why v2 dropped the channel entirely.
