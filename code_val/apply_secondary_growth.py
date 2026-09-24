"""Reassigne les rayons racinaires d'un vtp CPlantBox par croissance secondaire.

CPlantBox fige le rayon de chaque racine a sa creation. Ce post-processing le
recalcule apres coup selon secondary_root_growth_math.md et reecrit le vtp enrichi :
la geometrie et le CellData d'origine sont conserves, seuls des tableaux de rayons
par noeud sont ajoutes.
"""
import argparse
from pathlib import Path

from secondary_growth.fusion import measure_taper_ratio
from secondary_growth.parameters import SecondaryGrowthParameters
from secondary_growth.polydata import read_polydata, write_polydata
from secondary_growth.radii import FINAL_RADIUS_NAME, build_radius_arrays

VTP_PATTERN = "*.vtp"
OUTPUT_SUFFIX = "_thickened.vtp"


def describe(path: Path, arrays: dict, tree) -> str:
    radii = arrays[FINAL_RADIUS_NAME]
    # Un seul noeud doit rester sans parent, le collet : les autres sont des laterales non raccordees.
    unattached = len(tree.levels[0]) - 1
    return (f"{path.name} : {len(radii)} noeuds, "
            f"rayon {radii.min():.3f}-{radii.max():.2f} cm "
            f"(primaire {tree.primary_radii.max():.3f} cm), "
            f"effilement {measure_taper_ratio(radii, tree.levels):.2f}, "
            f"{unattached} laterales non raccordees, "
            f"{tree.ambiguous_links} raccords ambigus")


def build_output_path(source: Path, destination: Path, directory_mode: bool) -> Path:
    if destination is None:
        return source.with_name(source.stem + OUTPUT_SUFFIX)
    return destination / source.name if directory_mode else destination


def apply_to_file(source: Path, output: Path, parameters: SecondaryGrowthParameters) -> str:
    polydata = read_polydata(source)
    arrays, tree = build_radius_arrays(polydata, parameters)
    write_polydata(output, polydata, arrays)
    return describe(output, arrays, tree)


def collect_sources(entry: Path) -> list:
    return sorted(entry.glob(VTP_PATTERN)) if entry.is_dir() else [entry]


def parse_arguments() -> argparse.Namespace:
    defaults = SecondaryGrowthParameters()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="un fichier .vtp ou un dossier en contenant")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="fichier ou dossier de sortie, sinon <nom>_thickened.vtp a cote")
    parser.add_argument("--growth-rate", type=float, default=defaults.secondary_growth_rate,
                        help="k, coefficient de croissance secondaire en cm.jour^-alpha")
    parser.add_argument("--allometric-exponent", type=float, default=defaults.allometric_exponent,
                        help="alpha, exposant allometrique de la loi en puissance")
    parser.add_argument("--pipe-efficiency", type=float, default=defaults.pipe_efficiency,
                        help="kappa, efficacite du pipe model de Shinozaki")
    parser.add_argument("--area-conservation", type=float, default=defaults.area_conservation,
                        help="SGC, conservation des aires de la regle de da Vinci")
    parser.add_argument("--murray-exponent", type=float, default=defaults.murray_exponent,
                        help="Delta, exposant de la loi de Murray entre 2 et 3")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    parameters = SecondaryGrowthParameters(
        secondary_growth_rate=arguments.growth_rate,
        allometric_exponent=arguments.allometric_exponent,
        pipe_efficiency=arguments.pipe_efficiency,
        area_conservation=arguments.area_conservation,
        murray_exponent=arguments.murray_exponent,
    )
    sources = collect_sources(arguments.input)
    if not sources:
        raise SystemExit(f"aucun fichier .vtp dans {arguments.input}")

    directory_mode = arguments.input.is_dir()
    for source in sources:
        output = build_output_path(source, arguments.output, directory_mode)
        print(apply_to_file(source, output, parameters), flush=True)


if __name__ == "__main__":
    main()
