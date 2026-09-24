# obstacles — v1 obstacle families, and the axis-aligned street prototype

Two generations of code share this folder, and they are used by different callers. Nothing here
is imported by the v2 generator, which moved the street to
[`network/`](../network/README.md) — but this package must keep working, because
`dataset/` (v1) and `sandbox/sandbox_v2.py` both depend on it.

| Group | Files | Used by |
|---|---|---|
| **v1 obstacle families** | `layouts.py`, `geometry.py`, `masks.py` | `dataset/example.py`, `visualise_example.py` |
| **axis-aligned street prototype** | `street.py`, `shapes.py`, `coverage.py`, `walls.py`, `leakage.py`, `intrusion.py` | `sandbox/sandbox_v2.py` only |

The prototype is where the three-mechanism street model was designed and measured; `network/`
is the same model with the axis constraint removed (free bearing, elbows, tees, polylines).

---

## Part 1 — the v1 obstacle families

Four families, drawn per configuration so that no two layouts are identical.

| Family | Contents |
|---|---|
| `none` | nothing |
| `single_pipe` | one pipe along Y |
| `crossing_pipes` | one along Y, one along X |
| `pipe_and_pavement` | one pipe along Y plus a pavement — either a sidewalk half-plane or a slab with a square tree pit cut out of it |

### Step by step

1. **`layouts.py::draw_obstacle_layout`** — the family fixes what is present, the numbers are
   drawn: pipe offset (±100 to ±400 cm), depth (60 to 150 cm), diameter (15 to 40 cm), pavement
   style and boundary.
2. **`geometry.py::build_growth_geometry`** — the CPlantBox domain: the soil box minus every
   pipe, minus the pavement slab. A pipe is a `SDF_PlantContainer` tipped a quarter turn so it
   runs horizontally. The soil box itself is what confines the root system to the grid extent.
3. **`masks.py::build_obstacle_masks`** — the voxel masks and the distance channel, computed
   analytically from the layout rather than sampled from the SDF: exact, and independent of how
   the geometry happens to be composed. A cell belongs to a pipe as soon as the cylinder passes
   *through the cell box*, not through its centre: pipes are 15 to 40 cm across, so testing
   centres would mark a pipe one cell thick, or miss it entirely.
4. `soil/field.py::apply_road_base_override` then compacts and dries the sub-base under a
   pavement, using the `ROAD_BASE` constants declared in `layouts.py`.

---

## Part 2 — the axis-aligned street prototype

Same three mechanisms as `network/`: a **wall** (absolute, lumens only), a **fractional penalty**
on the five soil channels (trench backfill, sand surround, loose backfill under the pit) and an
**attraction** (leak plume → moisture → ψ → `setSoil()` + `tropismT = 3`).

### Step by step

1. **`street.py::draw_street_scene`** — six templates, all present, all along one axis: telecom,
   electricity, gas, water, sewer (parallel to the street) and a sewer connection crossing it.
   Offsets are measured from the curb, which keeps the sewer on the opposite side.
2. **`shapes.py::build_scene_shapes`** — cylinders and cuboids, ordered by fill priority: lumens,
   sand surrounds, loose backfill where a trench crosses the pit column, recompacted trench
   backfill. A trench is excavated **from the surface**, so it is a slot, not a sleeve.
3. **`coverage.py::build_coverage`** — the fraction of each voxel a solid occupies. An
   axis-aligned box separates exactly into three 1D overlaps, so it costs nothing and carries no
   quadrature error; a horizontal cylinder spans the domain along its axis, leaving a 2D cross
   section that is oversampled 32 × 32.
4. **`walls.py::build_street_geometry`** — the soil box minus every lumen inflated by 4 cm.
5. **`leakage.py`** — the joint nearest the trunk, and an ellipsoidal plume long along the trench
   and short across it.
6. **`intrusion.py::find_intrusions`** — the after-the-fact check, exact because in the plane
   cutting the pipe the axis is a point, so it reduces to point-to-segment.

### Measured on this code

With `sandbox/sandbox_v2.py`, 120 years, 10 seeds, the simplified scene against the same scene
with the networks removed:

| archetype | mass | mean radius | support |
|---|---|---|---|
| loose_uniform | −36.2 % | −24.4 % | −40.1 % |
| medium_uniform | −75.1 % | −52.0 % | −78.6 % |

For comparison the full street cut (with surface roadway) gave −44.6 % and −73.0 % on the mass:
**removing the surface roadway cost the signal almost nothing**, because the trenches now reach
the surface and take over the blocking role `SIDEWALK_BASE` used to play. v1's obstacle families
gave +4.6 %, the wrong sign.

Other measurements on this code: cuboid fractions exact to machine precision, cylinder fractions
within 0.44 % at worst (Ø11) and under 0.1 % beyond; **0 sign disagreements out of 960 000** tests
between the numpy distance and `sdf.getDist`; and zero root points inside the six lumens over
10 seeds × 2 archetypes × 4 modes.

---

## Limits

- **The v1 families are not a street.** A single pipe at a random offset does not reproduce the
  corridor structure, the trench backfill or the leak, which is why v2 exists.
- **`obstacle_distance` assumes an infinite axis-aligned pipe** and can say nothing about a
  diagonal or an elbow. The channel was dropped in v2.
- **The prototype's trench backfill reaches the surface.** With no sidewalk above it,
  `RECOMPACTED_BACKFILL` at 3.0 MPa gives `apply_bengough = 0` from 0 cm down to the bedding. The
  identified knob, if targets come out too empty, is a softer surface-reinstatement horizon over
  the top 15–20 cm of trench — not the return of surface roadway.
- **Rosetta breaks down above a bulk density of 2.0**: θ_S follows porosity up to BD 2.00, then
  jumps back to 0.367 and stops moving. Only affects the `hard_base_at_50` archetype (BD 2.10).
  One-line fix: bound θ_S by the porosity the density implies.
- **Do not refactor this package to match `network/`.** They are kept apart on purpose; v1 must
  keep producing identical output.
