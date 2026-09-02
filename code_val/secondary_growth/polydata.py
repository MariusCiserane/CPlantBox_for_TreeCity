"""Lecture et reecriture d'un vtp CPlantBox sans passer par le module vtk.

Organism::writeVTP produit du XML ASCII : ElementTree suffit. La reecriture
conserve le document d'origine et n'ajoute que des DataArray dans <PointData>,
donc les points, les polylignes et le CellData restent identiques a l'entree.
"""
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import NamedTuple

import numpy as np

POINT_DATA_PATH = "PolyData/Piece/PointData"
CELL_DATA_PATH = "PolyData/Piece/CellData"
COORDINATES_PATH = "PolyData/Piece/Points/DataArray"
OFFSETS_PATH = "PolyData/Piece/Lines/DataArray[@Name='offsets']"
COORDINATE_COMPONENTS = 3
ARRAY_TEMPLATE = '<DataArray type="Float32" Name="{name}" NumberOfComponents="1" format="ascii">{text}</DataArray>'
VALUE_FORMAT = "%.9g"


class PolyData(NamedTuple):
    points: np.ndarray          # (n_points, 3)
    offsets: np.ndarray         # (n_lines,) fin exclusive de chaque polyligne
    cell_arrays: dict           # une valeur par polyligne
    point_arrays: dict          # une valeur par point
    document: ElementTree.ElementTree


def read_array(element: ElementTree.Element) -> np.ndarray:
    return np.fromstring(element.text, sep=" ")


def read_section(root: ElementTree.Element, path: str) -> dict:
    section = root.find(path)
    if section is None:
        return {}
    return {array.get("Name"): read_array(array) for array in section.findall("DataArray")}


def read_polydata(path: Path) -> PolyData:
    document = ElementTree.parse(path)
    root = document.getroot()
    coordinates = read_array(root.find(COORDINATES_PATH))
    return PolyData(
        points=coordinates.reshape(-1, COORDINATE_COMPONENTS),
        offsets=read_array(root.find(OFFSETS_PATH)).astype(np.intp),
        cell_arrays=read_section(root, CELL_DATA_PATH),
        point_arrays=read_section(root, POINT_DATA_PATH),
        document=document,
    )


def build_array_element(name: str, values: np.ndarray) -> ElementTree.Element:
    text = " ".join(VALUE_FORMAT % value for value in values)
    return ElementTree.fromstring(ARRAY_TEMPLATE.format(name=name, text=text))


def write_polydata(path: Path, polydata: PolyData, point_arrays: dict) -> None:
    section = polydata.document.getroot().find(POINT_DATA_PATH)
    for name, values in point_arrays.items():
        section.append(build_array_element(name, values))
    path.parent.mkdir(parents=True, exist_ok=True)
    polydata.document.write(path, encoding="utf-8", xml_declaration=True)
