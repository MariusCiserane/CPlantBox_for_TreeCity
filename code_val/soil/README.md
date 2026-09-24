# soil — from an archetype to the number CPlantBox reads

Everything between "this scene sits on a compacted urban fill" and `f_se`, the dimensionless
elongation scale a root apex reads at its own position. Both generators share this package.

## The chain, step by step

Each step is one file, and each takes the previous one's output.

1. **`archetypes.py` — the reference soils.** 10 archetypes, each two horizons (topsoil,
   subsoil) of five channels: penetration resistance, bulk density, moisture, sand and clay
   fraction. Also the physical bounds every later step clips to, and `PIT_SUBSTRATE_ARCHETYPE`,
   the amended backfill of a planting pit.
2. **`profile.py::draw_soil_profile` — jitter, do not copy.** Draws a concrete profile *around*
   an archetype: ±15 % per channel, ±10 cm on the horizon depth, with sand + clay held under 0.95
   so the texture stays a real texture.
3. **`mosaic.py::draw_soil_mosaic` — several archetypes in one domain (v2 only).**
   An urban fill is not one profile copied everywhere; it is materials tipped side by side.
   2 to 4 patches, each with its own profile and its own horizon depth. A patch's weight at a
   point is a softmax of the distance to its seed: weights sum to 1 everywhere with no
   renormalisation, and the softmax temperature *is* the transition width in centimetres. The
   distance is perturbed by a correlated field, otherwise the boundaries would be the bisectors
   of a Voronoi diagram — far too straight for an excavation. One scene in four takes a short
   transition, which represents a vertical trench joint, so the dataset holds both contact
   regimes. The main patch seeds near the trunk, since the near field carries most of the root
   mass. Weights are not stored: the boundary seed and three lengths reproduce them.
4. **`field.py::build_mosaic_field` — the five grids.** Blends the patches with the same weights
   for every channel, which keeps a blended voxel on the segment between two real soils instead
   of somewhere no soil exists, then adds spatial noise. **Two noise fields, not five**: one for
   compaction (resistance ↑, density ↑, moisture ↓ together) and one for texture (sand and clay
   move apart). Noising every channel independently, as v1 did, let a voxel be dense, wet and weak
   at the same time.
5. **`pit.py::apply_planting_pit` — the planting pit.** A second soil zone, horizontal rather
   than layered: a street tree is planted in an amended backfilled pit, not in the ambient fill.
   Without it a hard topsoil gives `f_se = 0` exactly where the seed germinates and the tree is
   stillborn. The pit **has a bottom**, otherwise it would open a soft chimney through every hard
   layer.
6. **`fill.py::apply_fills` — engineered materials (v2).** The pit swaps whole voxels because it
   is metres across; a pipe surround is centimetres across, so it is blended by the fraction of
   the voxel it actually occupies. Zones overlap, so the list is a **priority order consumed
   against a decreasing budget** — which is what carves the rings without any shape being
   described as a ring: the lumen reserves its volume and writes nothing, the surround takes what
   is left inside it, the backfill takes the rest.
7. **`leak.py::apply_leaks` — the leak's water (v2).** Blends the moisture *towards* the plume
   target rather than overwriting it, so the ramp keeps sloping all the way out; a maximum would
   flatten it wherever the soil is already wetter than the tail, and a flat field steers nothing.
   It then rescales the penetration resistance, which is what stops a leak from reaching the
   water half of `f_se` while the mechanical half never noticed.
8. **`hydraulics.py` + `rosetta_batch.py` — moisture to matric potential.** Every voxel gets its
   own retention curve, predicted by Rosetta from its own noised texture and density, so textural
   variation reaches the retention physics instead of only sliding moisture along one curve per
   horizon. Rosetta takes a batch, so the whole grid goes in one call: the records are rounded
   first and only the distinct ones are sent, so the cost follows how spread out the soil is, not
   how many voxels there are.
9. **`growth_response.py` — the number CPlantBox reads.**
   `f_se = bengough(resistance) × water_stress(ψ)`; `f_sa`, the lateral insertion angle scale, is
   built from the mechanical term alone.

`wet_spots.py` is off this chain: spherical saturated pockets used only by the hydrotropism
sandbox, to give a tropism something it can demonstrably follow.

## Who does what

| File | Responsibility |
|---|---|
| `archetypes.py` | the 10 reference soils, the pit substrate, and the physical bounds |
| `profile.py` | draws a concrete profile around an archetype |
| `mosaic.py` | several archetypes in one domain, blended by patch weights (v2) |
| `field.py` | profile(s) → the five channel grids, with correlated noise; road-base override; moisture → resistance coupling |
| `pit.py` | the planting pit mask, and its application |
| `fill.py` | engineered materials blended by volume fraction, against a decreasing budget |
| `leak.py` | injects a leak plume's moisture and softens the resistance with it |
| `hydraulics.py` | moisture + retention parameters → pressure head |
| `rosetta_batch.py` | one Rosetta call for the whole grid, deduplicated and rounded |
| `growth_response.py` | Bengough impedance, water stress, `f_se` and `f_sa` |

## Key numbers

| | |
|---|---|
| Jitter around an archetype | ±15 % per channel, ±10 cm on the horizon depth |
| Noise | amplitude 8 %, correlation length 2 voxels |
| Bengough | free below 0.5 MPa, stalled above 2.5 MPa, linear in between |
| Water stress | half elongation at 5 100 cm of suction, floor 0.05 |
| `f_sa` | floor 0.35 — a lateral that cannot push sideways emerges closer to its parent's heading |
| Moisture → resistance | exponent −2, scale clipped to [0.25, 2.0], applied only to the natural share |
| Patches | 2 to 4, seeds at least 250 cm apart, transition 40–90 cm (or 20 cm, 25 % of the time) |
| Rosetta batch | rounded to 1 % texture and 0.02 g/cm³, coarsened until under 12 000 distinct records |
| Pit | radius 50–150 cm, depth 60–100 cm |

## Choices and limits

- **`HALF_ELONGATION_SUCTION_CM = 5100` puts half-reduction at pF 3.7**, while field capacity is
  at pF 2.0–2.5: the water term only bites near wilting point. On the wetter archetypes `f_se` is
  therefore driven by Bengough alone. Worth recalibrating if the water term is meant to carry
  information — all the more so now that the leak plume is the only moisture structure.
- **Rosetta breaks down above bulk density 2.0**: θ_S tracks porosity up to BD 2.00, then returns
  to 0.367 and ignores density. Only `hard_base_at_50` (BD 2.10) is concerned. One-line fix: bound
  θ_S by the porosity the density implies.
- **Noise is shared across patches, on purpose.** One noise field per patch would have to be
  blended too, and blending independent gaussians shrinks the variance right inside the
  transition band — the one place it should not.
- **The moisture → resistance coupling only applies to the natural share.** A fully engineered
  fill arrives with its resistance already consistent and must not be rescaled, which also means
  a leak cannot soften its own backfill.
- **Static hydrology.** No infiltration, no drainage, no root water uptake.
- **`f_se` can reach zero**, which stalls a root permanently; in v1 it was floored at 0.05. The
  frozen v1 behaviour lives in `sandbox/sandbox_v1.py`. Note that a lasting `f_se < 1` makes
  CPlantBox compute negative lateral emergence delays — harmless for structural subtypes, but a
  `Release` build would trip the `Root::calcLength() negative root age` assertion. Stay on
  `RelWithDebInfo`, or apply `std::max(ageLG - ageLN, 0.)` in `Organ::getLatGrowthDelay`.
