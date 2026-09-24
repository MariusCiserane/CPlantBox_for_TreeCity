"""Les reseaux enterres autour d'un arbre de rue, et comment on les pose.

La voirie de surface n'est pas modelisee : ce qui atteint les racines, c'est l'excavation, pas
la dalle posee dessus. La bordure survit comme une simple coordonnee, la ligne depuis laquelle
les corridors sont decales, ce qui garde telecom, eclairage, elec, gaz, chauffage et eau d'un
cote et l'egout de l'autre.

Les diagonales viennent d'un azimut de rue unique, pas d'un alea par tuyau : c'est ce qui fait
une rue plutot qu'un tas de tuyaux. Un raccordement n'est tire que si son reseau principal est
la, et il se termine sur la polyligne de celui-ci, ce qui produit les tes.
"""
import math
from dataclasses import dataclass

import numpy as np

from grid import VoxelGrid
from network.primitives import PipeRun, measure_signed_distance
from soil.archetypes import SoilHorizon

CORRIDOR, CONNECTION, PIT_SERVICE = "corridor", "connection", "pit_service"

CURB_DISTANCE_RANGE_CM = (150.0, 280.0)
TRENCH_WORKING_SPACE_CM = 30.0
BEDDING_DEPTH_RANGE_CM = (10.0, 15.0)
SURROUND_COVER_RANGE_CM = (15.0, 30.0)

RECOMPACTED_BACKFILL = SoilHorizon(3.0, 1.95, 0.25, 0.82, 0.08)
LOOSE_BACKFILL = SoilHorizon(0.7, 1.30, 0.50, 0.55, 0.15)
PIPE_SURROUND = SoilHorizon(0.5, 1.50, 0.20, 0.95, 0.01)

# LeakageStatusValue du paquet utilities CityGML de Team 3.
NO_LEAK, MINOR_LEAK, MAJOR_LEAK = "none", "minor", "major"
LEAKAGE_STATUSES = (NO_LEAK, MINOR_LEAK, MAJOR_LEAK)
LEAKAGE_WEIGHTS = (0.34, 0.33, 0.33)

DENSITY_CLASSES = (("bare", 0, 1), ("residential", 2, 4), ("dense", 5, 7), ("city_centre", 8, 10))
DENSITY_WEIGHTS = (0.10, 0.30, 0.35, 0.25)

STRAIGHT_BEARING_RANGE_DEG = (-10.0, 10.0)
OBLIQUE_BEARING_RANGE_DEG = (-40.0, 40.0)
OBLIQUE_STREET_PROBABILITY = 0.35
RUN_BEARING_JITTER_DEG = 3.0

ELBOW_PROBABILITY = 0.25
ELBOW_TURN_RANGE_DEG = (15.0, 45.0)
ELBOW_POSITION_RANGE = (0.3, 0.7)
CROSSING_TURN_RANGE_DEG = (-20.0, 20.0)
PIT_SERVICE_PROBABILITY = 0.30
PIT_SERVICE_CLEARANCE = 1.2

MINIMUM_LUMEN_CLEARANCE_CM = 10.0
DEEPENING_LIMIT_CM = 40.0
PLACEMENT_ATTEMPTS = 20
CLEARANCE_SAMPLE_STEP_CM = 2.0


@dataclass(frozen=True)
class NetworkTemplate:
    name: str
    role: str
    diameter_range_cm: tuple[float, float]
    crown_depth_range_cm: tuple[float, float]
    offset_range_cm: tuple[float, float]
    joint_spacing_cm: float
    presence_weight: float
    parent: str = ""


# Les decalages se mesurent depuis la bordure. Negatif du cote trottoir, positif cote chaussee.
NETWORK_TEMPLATES = (
    NetworkTemplate("lighting", CORRIDOR, (6.0, 10.0), (40.0, 60.0), (-30.0, -10.0), 0.0, 0.55),
    NetworkTemplate("telecom", CORRIDOR, (10.0, 14.0), (40.0, 70.0), (-90.0, -30.0), 0.0, 0.75),
    NetworkTemplate("electricity", CORRIDOR, (14.0, 18.0), (60.0, 85.0), (-140.0, -80.0), 0.0, 0.80),
    NetworkTemplate("gas", CORRIDOR, (9.0, 13.0), (70.0, 100.0), (-190.0, -130.0), 0.0, 0.70),
    NetworkTemplate("water", CORRIDOR, (12.0, 25.0), (90.0, 130.0), (-250.0, -170.0), 600.0, 0.85),
    NetworkTemplate("heating", CORRIDOR, (25.0, 45.0), (80.0, 120.0), (-330.0, -260.0), 0.0, 0.35),
    NetworkTemplate("sewer", CORRIDOR, (40.0, 80.0), (180.0, 300.0), (100.0, 170.0), 250.0, 0.90),
    NetworkTemplate("water_connection", CONNECTION, (8.0, 14.0), (90.0, 130.0), (), 600.0, 0.60,
                    "water"),
    NetworkTemplate("sewer_connection", CONNECTION, (12.0, 20.0), (100.0, 200.0), (), 250.0, 0.65,
                    "sewer"),
)


@dataclass(frozen=True)
class UtilityRun:
    name: str
    role: str
    nodes_cm: tuple[tuple[float, float, float], ...]
    radius_cm: float
    trench_half_width_cm: float
    bedding_depth_cm: float
    surround_cover_cm: float
    joint_spacing_cm: float
    leakage_status: str
    parent: str


@dataclass(frozen=True)
class StreetScene:
    bearing_deg: float
    density_class: str
    curb_offset_cm: float
    pit_radius_cm: float
    pit_depth_cm: float
    utilities: tuple[UtilityRun, ...]


def build_lumen(run: UtilityRun) -> PipeRun:
    return PipeRun(run.name, run.nodes_cm, run.radius_cm)


def draw_bearing_deg(rng: np.random.Generator) -> float:
    oblique = rng.random() < OBLIQUE_STREET_PROBABILITY
    return float(rng.uniform(*(OBLIQUE_BEARING_RANGE_DEG if oblique
                               else STRAIGHT_BEARING_RANGE_DEG)))


def draw_density(rng: np.random.Generator) -> tuple:
    name, low, high = DENSITY_CLASSES[rng.choice(len(DENSITY_CLASSES), p=DENSITY_WEIGHTS)]
    return name, int(rng.integers(low, high + 1))


def select_templates(count: int, rng: np.random.Generator) -> list:
    """Tirage pondere sans remise. Un raccordement n'est eligible qu'une fois son reseau la."""
    weights = np.array([template.presence_weight for template in NETWORK_TEMPLATES])
    order = np.argsort(-rng.random(len(NETWORK_TEMPLATES)) ** (1.0 / weights))

    selected, names = [], set()
    for index in order:
        template = NETWORK_TEMPLATES[index]
        if len(selected) == count:
            break
        if template.parent and template.parent not in names:
            continue
        selected.append(template)
        names.add(template.name)
    return selected


def build_frame(bearing_deg: float) -> tuple:
    radians = math.radians(bearing_deg)
    return (np.array([math.cos(radians), math.sin(radians)]),
            np.array([-math.sin(radians), math.cos(radians)]))


def build_corridor_nodes(lateral_cm: float, depth_cm: float, bearing_deg: float,
                         reach_cm: float, rng: np.random.Generator) -> tuple:
    along, across = build_frame(bearing_deg + float(rng.uniform(-RUN_BEARING_JITTER_DEG,
                                                               RUN_BEARING_JITTER_DEG)))
    centre = across * lateral_cm
    start, end = centre - along * reach_cm, centre + along * reach_cm
    if rng.random() >= ELBOW_PROBABILITY:
        return ((*start, -depth_cm), (*end, -depth_cm))

    corner = start + (end - start) * float(rng.uniform(*ELBOW_POSITION_RANGE))
    turn = math.radians(float(rng.uniform(*ELBOW_TURN_RANGE_DEG)) * (1.0 if rng.random() < 0.5 else -1.0))
    turned = np.array([math.cos(turn) * along[0] - math.sin(turn) * along[1],
                       math.sin(turn) * along[0] + math.cos(turn) * along[1]])
    return ((*start, -depth_cm), (*corner, -depth_cm), (*(corner + turned * reach_cm), -depth_cm))


def build_connection_nodes(parent: UtilityRun, depth_cm: float, bearing_deg: float,
                           reach_cm: float, span_cm: float, rng: np.random.Generator) -> tuple:
    """Depuis le bord de la scene jusqu'a un te pose sur le reseau principal, dans le domaine."""
    nodes = np.asarray(parent.nodes_cm, float)
    axis = nodes[1][:2] - nodes[0][:2]
    axis = axis / np.linalg.norm(axis)
    closest = nodes[0][:2] + axis * (-nodes[0][:2] @ axis)
    tee = closest + axis * float(rng.uniform(-span_cm, span_cm))

    heading, _ = build_frame(bearing_deg + 90.0 + float(rng.uniform(*CROSSING_TURN_RANGE_DEG)))
    side = 1.0 if rng.random() < 0.5 else -1.0
    return ((*(tee + side * heading * reach_cm), -depth_cm), (*tee, -depth_cm))


def build_pit_service_nodes(pit_radius_cm: float, depth_cm: float, bearing_deg: float,
                            reach_cm: float, rng: np.random.Generator) -> tuple:
    heading, _ = build_frame(float(rng.uniform(0.0, 360.0)))
    end = heading * PIT_SERVICE_CLEARANCE * pit_radius_cm
    return ((*(end + heading * reach_cm), -depth_cm), (*end, -depth_cm))


def sample_run(run: UtilityRun) -> np.ndarray:
    nodes = np.asarray(run.nodes_cm, float)
    return np.concatenate([np.linspace(start, end, int(np.linalg.norm(end - start)
                                                       / CLEARANCE_SAMPLE_STEP_CM) + 2)
                           for start, end in zip(nodes[:-1], nodes[1:])])


def measure_clearance_cm(run: UtilityRun, other: UtilityRun) -> float:
    """Mesuree dans les deux sens : echantillonner une seule des deux polylignes surestime."""
    return min(float(measure_signed_distance(build_lumen(other), sample_run(run)).min()) - run.radius_cm,
               float(measure_signed_distance(build_lumen(run), sample_run(other)).min()) - other.radius_cm)


def measure_deficit_cm(run: UtilityRun, accepted: dict) -> float:
    """De combien ce lumen manque de place, les reseaux qu'il raccorde mis a part."""
    clearances = [measure_clearance_cm(run, other) for other in accepted.values()
                  if other.name != run.parent and other.parent != run.name]
    return MINIMUM_LUMEN_CLEARANCE_CM - min(clearances, default=np.inf)


def draw_utility(template: NetworkTemplate, nodes, depth_cm: float, radius_cm: float,
                 rng: np.random.Generator) -> UtilityRun:
    leaks = template.joint_spacing_cm > 0.0
    return UtilityRun(
        name=template.name,
        role=template.role,
        nodes_cm=tuple(tuple(float(value) for value in node) for node in nodes),
        radius_cm=radius_cm,
        trench_half_width_cm=radius_cm + TRENCH_WORKING_SPACE_CM,
        bedding_depth_cm=float(rng.uniform(*BEDDING_DEPTH_RANGE_CM)),
        surround_cover_cm=float(rng.uniform(*SURROUND_COVER_RANGE_CM)),
        joint_spacing_cm=template.joint_spacing_cm,
        leakage_status=str(rng.choice(LEAKAGE_STATUSES, p=LEAKAGE_WEIGHTS)) if leaks else NO_LEAK,
        parent=template.parent,
    )


def build_run(template: NetworkTemplate, scene_runs: dict, curb_offset_cm: float,
              bearing_deg: float, pit_radius_cm: float, grid: VoxelGrid,
              rng: np.random.Generator) -> UtilityRun:
    reach_cm = 2.0 * (grid.half_width_cm + grid.half_length_cm)
    radius_cm = 0.5 * float(rng.uniform(*template.diameter_range_cm))
    crown_cm = float(rng.uniform(*template.crown_depth_range_cm))

    if template.role == CORRIDOR:
        limit_cm = grid.half_width_cm - radius_cm - TRENCH_WORKING_SPACE_CM
        lateral_cm = float(np.clip(curb_offset_cm + rng.uniform(*template.offset_range_cm),
                                   -limit_cm, limit_cm))
        nodes = build_corridor_nodes(lateral_cm, crown_cm + radius_cm, bearing_deg, reach_cm, rng)
    elif template.role == CONNECTION:
        nodes = build_connection_nodes(scene_runs[template.parent], crown_cm + radius_cm,
                                       bearing_deg, reach_cm, grid.half_length_cm, rng)
    else:
        nodes = build_pit_service_nodes(pit_radius_cm, crown_cm + radius_cm, bearing_deg,
                                        reach_cm, rng)
    return draw_utility(template, nodes, crown_cm + radius_cm, radius_cm, rng)


def deepen(run: UtilityRun, drop_cm: float) -> UtilityRun:
    nodes = tuple((x, y, z - drop_cm) for x, y, z in run.nodes_cm)
    return UtilityRun(run.name, run.role, nodes, run.radius_cm, run.trench_half_width_cm,
                      run.bedding_depth_cm, run.surround_cover_cm, run.joint_spacing_cm,
                      run.leakage_status, run.parent)


def place_run(template: NetworkTemplate, accepted: dict, curb_offset_cm: float, bearing_deg: float,
              pit_radius_cm: float, grid: VoxelGrid, rng: np.random.Generator):
    """Deux lumens ne se touchent jamais : un service qui genait passe dessous, sinon on retire."""
    for _ in range(PLACEMENT_ATTEMPTS):
        run = build_run(template, accepted, curb_offset_cm, bearing_deg, pit_radius_cm, grid, rng)
        deficit = measure_deficit_cm(run, accepted)
        if deficit <= 0.0:
            return run
        if deficit <= DEEPENING_LIMIT_CM:
            deeper = deepen(run, deficit)
            if measure_deficit_cm(deeper, accepted) <= 0.0:
                return deeper
    return None


def draw_street_scene(pit_radius_cm: float, pit_depth_cm: float, grid: VoxelGrid,
                      rng: np.random.Generator) -> StreetScene:
    bearing_deg = draw_bearing_deg(rng)
    density_class, count = draw_density(rng)
    curb_offset_cm = float(rng.uniform(*CURB_DISTANCE_RANGE_CM))

    accepted = {}
    for template in select_templates(count, rng):
        # Un raccordement dont le reseau principal n'a pas trouve sa place n'a rien a raccorder.
        if template.parent and template.parent not in accepted:
            continue
        run = place_run(template, accepted, curb_offset_cm, bearing_deg, pit_radius_cm, grid, rng)
        if run is not None:
            accepted[run.name] = run

    if accepted and rng.random() < PIT_SERVICE_PROBABILITY:
        template = NetworkTemplate("pit_service", PIT_SERVICE, (6.0, 12.0), (50.0, 90.0), (), 0.0, 1.0)
        run = place_run(template, accepted, curb_offset_cm, bearing_deg, pit_radius_cm, grid, rng)
        if run is not None:
            accepted[run.name] = run

    return StreetScene(bearing_deg, density_class, curb_offset_cm, pit_radius_cm, pit_depth_cm,
                       tuple(accepted.values()))
