"""Une scene du generateur ecrite dans le contrat d'inference site_request.

C'est ce qu'un utilisateur aurait saisi pour cette scene : les horizons tels que tires, avant le
bruit spatial, et le trace des reseaux dont le channel builder deduit lui-meme remblais et fuites.
Les fuites sont ecrites explicitement, puisque le generateur sait ou elles sont.

L'eau suit le contrat : une humidite volumique theta, la grandeur qu'on mesure sur le terrain,
reconstruite depuis la fraction de reserve avec la courbe de retention de chaque horizon.

Les axes sont ceux de grid.py, row = y et column = x ; le generateur stocke ses points en
(x, y, z), d'ou la permutation.
"""
import numpy as np

from generator.scene import SceneConfiguration
from soil.hydraulics import convert_available_water_to_moisture
from soil.rosetta_batch import estimate_van_genuchten_parameters_field

SCHEMA_VERSION = "site_request/v2"
REQUEST_TEMPLATE = "{dataset}/scene_{identifier:05d}"
PROVENANCE = "generator"
DESCRIBED_BLOCKS = ("tree", "pit", "soil", "street", "leaks")


def estimate_moistures(horizons: list) -> dict:
    """Un seul appel Rosetta pour tous les horizons de la scene."""
    sand, clay, density = (np.array([getattr(horizon, name) for horizon in horizons])
                           for name in ("sand_fraction", "clay_fraction", "bulk_density_g_cm3"))
    parameters = estimate_van_genuchten_parameters_field(sand, clay, density)
    fractions = np.array([horizon.available_water_fraction for horizon in horizons])
    return dict(zip(horizons, convert_available_water_to_moisture(fractions, parameters).tolist()))


def describe_layer(horizon, moistures: dict) -> dict:
    return {"sand_fraction": horizon.sand_fraction, "clay_fraction": horizon.clay_fraction,
            "bulk_density_g_cm3": horizon.bulk_density_g_cm3,
            "moisture_m3_m3": moistures[horizon], "penetrometer_mpa": horizon.penetrometer_mpa}


def describe_profile(profile, moistures: dict) -> dict:
    return {"topsoil_depth_cm": profile.topsoil_depth_cm,
            "topsoil": describe_layer(profile.topsoil, moistures),
            "subsoil": describe_layer(profile.subsoil, moistures)}


def describe_soil(mosaic, moistures: dict) -> dict:
    primary, *others = mosaic.patches
    return {"default_profile": describe_profile(primary.profile, moistures),
            "default_center_row_cm": primary.centre_row_cm,
            "default_center_column_cm": primary.centre_column_cm,
            "patches": [{"center_row_cm": patch.centre_row_cm,
                         "center_column_cm": patch.centre_column_cm,
                         "profile": describe_profile(patch.profile, moistures)} for patch in others],
            "blending": {"transition_length_cm": mosaic.transition_length_cm,
                         "boundary_wander_cm": mosaic.boundary_wander_cm,
                         "boundary_wander_length_cm": mosaic.boundary_wander_length_cm,
                         "boundary_seed": mosaic.boundary_seed}}


def describe_utility(run) -> dict:
    return {"name": run.name, "role": run.role, "parent": run.parent,
            "nodes_cm": [[y, x, z] for x, y, z in run.nodes_cm],
            "radius_cm": run.radius_cm, "trench_half_width_cm": run.trench_half_width_cm,
            "bedding_depth_cm": run.bedding_depth_cm, "surround_cover_cm": run.surround_cover_cm,
            "joint_spacing_cm": run.joint_spacing_cm, "leakage_status": run.leakage_status}


def describe_leak(leak) -> dict:
    x, y, z = leak.origin_cm
    along_x, along_y = leak.along_unit
    return {"name": leak.name, "origin_cm": [y, x, z], "along_unit": [along_y, along_x],
            "available_water_fraction": leak.available_water_fraction,
            "reach_scale": leak.reach_scale}


def build_site_request(configuration: SceneConfiguration, dataset_name: str) -> dict:
    horizons = [configuration.pit_soil.topsoil] + [
        horizon for patch in configuration.soil.patches
        for horizon in (patch.profile.topsoil, patch.profile.subsoil)]
    moistures = estimate_moistures(horizons)
    tree = configuration.tree
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": REQUEST_TEMPLATE.format(dataset=dataset_name, identifier=configuration.identifier),
        "tree": {"species": tree.species, "age_years": tree.age_years, "dbh_cm": tree.dbh_cm,
                 "height_m": tree.height_m, "crown_depth_m": tree.crown_depth_m},
        "pit": {"radius_cm": configuration.pit_radius_cm, "depth_cm": configuration.pit_depth_cm,
                "substrate": describe_layer(configuration.pit_soil.topsoil, moistures)},
        "soil": describe_soil(configuration.soil, moistures),
        "street": {"bearing_deg": configuration.street.bearing_deg,
                   "curb_offset_cm": configuration.street.curb_offset_cm,
                   "utilities": [describe_utility(run) for run in configuration.street.utilities]},
        "leaks": [describe_leak(leak) for leak in configuration.leaks],
        "provenance": {block: PROVENANCE for block in DESCRIBED_BLOCKS},
    }
