"""Le sol d'une scene, du remblai brut aux objets que CPlantBox lit.

L'ordre porte le raisonnement et ne se permute pas :

  fosse avant remblais    - un reseau traverse la fosse comme le reste du sol ;
  remblais avant fuites   - un remblai arrive avec sa resistance et son humidite deja
                            coherentes, il ne doit pas etre re-echelonne par le couplage
                            humidite vers resistance que la fuite declenche ;
  Rosetta sur le champ composite, pas sur le profil d'origine ;
  geometrie en dernier    - grow_root_system appelle setSoil() puis initialize(), et
                            initialize() construit les tropismes a partir de ce qu'il tient a
                            cet instant.

La geometrie doit rester referencee pendant toute la boucle des tirages : CPlantBox la tient
par weak_ptr, et si elle est liberee le confinement s'arrete en silence.
"""
from dataclasses import dataclass

import numpy as np

from generator.scene import SceneConfiguration
from grid import VoxelGrid
from network.coverage import build_coverage, build_label_field, measure_cell_axes_cm
from network.leaks import build_plume_ramp
from network.primitives import measure_signed_distance
from network.shapes import build_scene_shapes
from network.walls import build_street_geometry
from soil.field import build_mosaic_field, build_soil_field
from soil.fill import apply_fills, measure_natural_share
from soil.growth_response import apply_water_stress, compute_growth_scale
from soil.hydraulics import build_pressure_head_field
from soil.leak import apply_leaks
from soil.mosaic import build_patch_weights
from soil.pit import apply_planting_pit, build_pit_mask

CLEARANCE_CLIP_CM = 300.0


@dataclass
class SceneEnvironment:
    field: object
    pressure_head: np.ndarray
    growth_scale: np.ndarray
    water_scale: np.ndarray
    pipe_fraction: np.ndarray
    surround_fraction: np.ndarray
    backfill_fraction: np.ndarray
    pipe_clearance_cm: np.ndarray
    leak_plume: np.ndarray
    pit_mask: np.ndarray
    pipe_labels: np.ndarray
    patch_labels: np.ndarray
    lumens: tuple
    elongation_grid: object
    water_lookup: object
    geometry: object


def measure_cell_points(grid: VoxelGrid) -> np.ndarray:
    depths, rows, columns = measure_cell_axes_cm(grid)
    depth, row, column = np.meshgrid(depths, rows, columns, indexing="ij")
    return np.stack([column.ravel(), row.ravel(), depth.ravel()], axis=1)


def measure_pipe_clearance_cm(lumens: tuple, grid: VoxelGrid) -> np.ndarray:
    if not lumens:
        return np.full(grid.shape, CLEARANCE_CLIP_CM, np.float32)
    points = measure_cell_points(grid)
    distances = np.min([measure_signed_distance(lumen, points) for lumen in lumens], axis=0)
    return np.clip(distances, 0.0, CLEARANCE_CLIP_CM).reshape(grid.shape).astype(np.float32)


def build_soil(configuration: SceneConfiguration, grid: VoxelGrid, shapes, coverages, weights):
    rng = np.random.default_rng(configuration.soil_noise_seed)
    ambient = build_mosaic_field(configuration.soil, weights, grid, rng)
    pit = build_soil_field(configuration.pit_soil, grid, rng)
    field = apply_planting_pit(ambient, pit, build_pit_mask(configuration.pit_radius_cm,
                                                            configuration.pit_depth_cm, grid))

    field = apply_fills(field, tuple(horizon for _, horizon in shapes.fills), coverages)
    sources = tuple((build_plume_ramp(leak, grid), leak.moisture_m3_m3)
                    for leak in configuration.leaks)
    natural_share = measure_natural_share(coverages, grid.shape)
    return apply_leaks(field, sources, natural_share), sources, natural_share


def build_environment(configuration: SceneConfiguration, grid: VoxelGrid) -> SceneEnvironment:
    from tree.growth import build_interpolated_soil_grid, build_soil_grid

    shapes = build_scene_shapes(configuration.street, grid)
    coverages = tuple(build_coverage(shape, grid) for shape, _ in shapes.fills)
    weights = build_patch_weights(configuration.soil, grid)
    field, sources, natural_share = build_soil(configuration, grid, shapes, coverages, weights)

    pressure_head = build_pressure_head_field(field)
    growth_scale = compute_growth_scale(field.penetrometer_mpa, pressure_head)
    water_scale = apply_water_stress(pressure_head).astype(np.float32)

    utility_count = len(configuration.street.utilities)
    pipe_fraction = np.clip(np.sum(coverages[:utility_count], axis=0), 0.0, 1.0) if coverages \
        else np.zeros(grid.shape, np.float32)
    surround_fraction = np.clip(np.sum(coverages[utility_count:2 * utility_count], axis=0)
                                - pipe_fraction, 0.0, 1.0) if coverages \
        else np.zeros(grid.shape, np.float32)
    plume = np.max([ramp for ramp, _ in sources], axis=0) if sources \
        else np.zeros(grid.shape, np.float32)

    return SceneEnvironment(
        field=field,
        pressure_head=pressure_head,
        growth_scale=growth_scale,
        water_scale=water_scale,
        pipe_fraction=pipe_fraction.astype(np.float32),
        surround_fraction=surround_fraction.astype(np.float32),
        backfill_fraction=(1.0 - natural_share).astype(np.float32),
        pipe_clearance_cm=measure_pipe_clearance_cm(shapes.lumens, grid),
        leak_plume=plume.astype(np.float32),
        pit_mask=build_pit_mask(configuration.pit_radius_cm, configuration.pit_depth_cm, grid),
        pipe_labels=build_label_field(shapes.lumens, grid),
        patch_labels=np.broadcast_to(weights.argmax(axis=0) + 1.0, grid.shape).astype(np.float32),
        lumens=shapes.lumens,
        elongation_grid=build_soil_grid(growth_scale, grid),
        water_lookup=build_interpolated_soil_grid(water_scale, grid),
        geometry=build_street_geometry(shapes.lumens, grid),
    )
