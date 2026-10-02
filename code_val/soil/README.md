# soil — from an archetype to the number CPlantBox reads

Everything between "this scene sits on a compacted urban fill" and `f_se`, the dimensionless
elongation scale a root apex reads at its own position. Both generators share this package.

## The chain, step by step

Each step is one file, and each takes the previous one's output.

1. **`archetypes.py` — the reference soils.** 10 archetypes, each two horizons (topsoil,
   subsoil) of five channels: penetration resistance, bulk density, **available water fraction**,
   sand and clay fraction. Also the physical bounds every later step clips to, and
   `PIT_SUBSTRATE_ARCHETYPE`, the amended backfill of a planting pit.
   Water is authored as a **share of the available water capacity**, 0 at the wilting point and 1
   at field capacity — never as a volumetric moisture. θ is not comparable across textures:
   θ = 0.12 is above field capacity in a sand and below the wilting point in a clay, so the same
   number on two lines would describe two opposite soils, and a ±15 % jitter would cross a sand's
   entire reserve (0.014 m³/m³ wide) eight times over. The fraction means the same thing
   everywhere, is bounded by construction, and `hydraulics.py` turns it back into θ per voxel.
2. **`profile.py::draw_soil_profile` — jitter, do not copy.** Draws a concrete profile *around*
   an archetype: ±15 % per channel, ±10 cm on the horizon depth, with sand + clay held under 0.95
   so the texture stays a real texture.
3. **`mosaic.py::draw_soil_mosaic` — several archetypes in one domain (v2 and v3).**
   An urban fill is not one profile copied everywhere; it is materials tipped side by side.
   1 to 4 patches, each with its own profile and its own horizon depth. A single patch — a
   laterally uniform soil — falls in one scene out of four since v3: the inference contract
   accepts it, so the network has to have seen it. A patch's weight at a
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
   compaction (resistance ↑, density ↑, water ↓ together) and one for texture (sand and clay
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
7. **`leak.py::apply_leaks` — the leak's water (v2).** Blends the water fraction *towards* the
   plume target rather than overwriting it, so the ramp keeps sloping all the way out; a maximum would
   flatten it wherever the soil is already wetter than the tail, and a flat field steers nothing.
   It then rescales the penetration resistance, which is what stops a leak from reaching the
   water half of `f_se` while the mechanical half never noticed.
8. **`hydraulics.py` + `rosetta_batch.py` — water fraction to θ and matric potential.** Every
   voxel gets its own retention curve, predicted by Rosetta from its own noised texture and density, so textural
   variation reaches the retention physics instead of only sliding moisture along one curve per
   horizon. Rosetta takes a batch, so the whole grid goes in one call: the records are rounded
   first and only the distinct ones are sent, so the cost follows how spread out the soil is, not
   how many voxels there are.
9. **`growth_response.py` — the number CPlantBox reads.**
   `f_se = bengough(resistance) × water_stress(available water fraction)`; `f_sa`, the lateral
   insertion angle scale, is built from the mechanical term alone and is wired into the growth
   since v3 (v2 read `f_se` only). Both responses are piecewise
   linear on a quantity that is already comparable across soils, so neither needs a retention
   curve to be read.

`wet_spots.py` is off this chain: spherical saturated pockets used only by the hydrotropism
sandbox, to give a tropism something it can demonstrably follow.

## Who does what

| File | Responsibility |
|---|---|
| `archetypes.py` | the 10 reference soils, the pit substrate, and the physical bounds |
| `profile.py` | draws a concrete profile around an archetype |
| `mosaic.py` | several archetypes in one domain, blended by patch weights (v2) |
| `field.py` | profile(s) → the five channel grids, with correlated noise; road-base override; water → resistance coupling |
| `pit.py` | the planting pit mask, and its application |
| `fill.py` | engineered materials blended by volume fraction, against a decreasing budget |
| `leak.py` | injects a leak plume's water fraction and softens the resistance with it |
| `hydraulics.py` | water fraction + retention parameters → θ and pressure head |
| `rosetta_batch.py` | one Rosetta call for the whole grid, deduplicated and rounded |
| `growth_response.py` | Bengough impedance, water stress, `f_se` and `f_sa` |

## Key numbers

| | |
|---|---|
| Jitter around an archetype | ±15 % per channel, ±10 cm on the horizon depth |
| Noise | amplitude 8 %, correlation length 2 voxels |
| Bengough | free below 0.5 MPa, stalled above 2.5 MPa, linear in between |
| Water stress | full speed above 0.40 of the reserve, linear below, floor 0.05 |
| Available water capacity | between field capacity (−100 cm) and wilting point (−15 000 cm) |
| `f_sa` | floor 0.35 — a lateral that cannot push sideways emerges closer to its parent's heading |
| Water → resistance | exponent −2, scale clipped to [0.25, 2.0], applied only to the natural share |
| Patches | 1 to 4 (2 to 4 in v2), seeds at least 250 cm apart, transition 40–90 cm (or 20 cm, 25 % of the time) |
| Rosetta batch | rounded to 1 % texture and 0.02 g/cm³, coarsened until under 12 000 distinct records |
| Pit | radius 50–150 cm, depth 60–100 cm |

## Choices and limits

- **The water stress threshold is 0.40 of the reserve**, taken from the transpiration
  literature, where the equivalent threshold sits at 0.3–0.4 across species. It is not a
  root-elongation measurement: no published curve was found for woody root elongation against
  available water, so this is the closest defensible proxy and it should be revisited against
  species data. The previous suction-based form put half-reduction at pF 3.7, near wilting point,
  which made the water term inert in every archetype.
- **The water → resistance exponent is published against volumetric moisture**, and is applied
  here to the ratio of available water fractions, because the coupling runs before Rosetta and θ
  is not available yet. The direction and the [0.25, 2.0] clip are unchanged.
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
