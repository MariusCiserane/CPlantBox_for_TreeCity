# hybrid — what the hybrid model adds to the dataset, and how to tell if it is worth it

The hybrid U-Net 3D + FiLM (vision document: `../TreeCity — Pipeline du modèle hybride
racinaire (1).md`) starts from physics instead of from zero: new atomic channels (phase A), a
prior added in logit space (phase B), hard masks (phase C). This package holds the one
implementation of each, shared by the v3 generator and by the post-processing of dataset v2.

## Who does what

| File | Responsibility |
|---|---|
| `channels.py` | the 4 added channels (`bengough`, 3 gradients of `water_scale`), and which baseline channels the hybrid model drops |
| `prior.py` | `prior_v2`, its calibration `k = base rate / mean prior`, and the soft Dice of the training loss |
| `masks.py` | the zones a hard mask would forbid, and the target mass they would destroy |

Two CLIs at the root use it:

```bash
python code_val/build_dataset_v2_hybrid.py        # dataset_v2/ → dataset_v2_hybrid/, no simulation
python code_val/measure_hybrid_readiness.py --dataset code_val/dataset_v2_hybrid
```

Neither needs CPlantBox: numpy and scipy are enough, on the host. The conversion reads
`bengough`, `eau` and `f_se` back from each `.vti` — the only float32 copy of what CPlantBox
read — because dataset v2 grew under the old θ → ψ water model that the current code no longer
computes. It takes about 5 minutes for the 2000 scenes, and writes 4.4 GB.

## Measured on 2026-09-25/26, 2000 scenes each

| Check | v2 | v3 | Reading |
|---|---|---|---|
| Dice of `p = t` | 0.239 | 0.256 | the document gives 0.233: this Dice is the training one |
| Dice of the geometric prior, raw | 0.029 | 0.024 | the document gives 0.028 |
| Dice of `prior_v2`, raw | 0.065 | 0.062 | × 2.2 to 2.6 over the geometric prior |
| Dice of `prior_v2`, calibrated (k) | 0.036 (0.149) | 0.034 (0.159) | against 0.24–0.26 for a perfect calibrated prediction |
| Effect of the leak boost B ∈ {0, 0.5, 1, 2} | < 1e-5 | < 1e-4 | B cannot be tuned on this criterion |
| Radius-target mass in the lumen (`pipe_fraction > 0.9`) | 4e-6, 35 scenes | 2e-6, 21 scenes | the pipe mask is safe |
| Radius-target mass in blocking soil (`f_se < 1e-3`) | 5.3 %, 1965 scenes | 5.4 %, 1956 scenes | a raw soil mask would destroy real mass |
| Same, blocking soil eroded by one voxel | 0.08 % | 0.08 % | the eroded mask is safe (centreline: 0.006 %) |

Calibration lowers the Dice: the soft Dice rewards a larger scale, so a raw and a calibrated
Dice are not comparable. Compare a calibrated prior with `p = t`, never with the raw figure.

## Choices and limits

- **Gradients per metre.** Per centimetre, a smooth gradient falls into the subnormals of the
  float16 the channels are stored in.
- **The masks are eroded by one voxel on purpose.** CPlantBox reads `f_se` at the apex before it
  advances, so a root finishes its last step inside the first blocked cell: 99.9 % of the mass in
  blocking soil sits one voxel from growable soil.
- **Thresholds are settings, not truths.** Bengough's 0.5 / 2.5 MPa were measured on maize,
  cotton and peanut, not on woody roots.
- **k is computed over every scene.** The training code must recompute it on its train split.
