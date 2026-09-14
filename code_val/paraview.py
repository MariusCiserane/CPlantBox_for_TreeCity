"""Write voxel fields as a .vti, to open in ParaView next to the root .vtp files.
"""
from pathlib import Path

import numpy as np

from grid import VoxelGrid

DOCUMENT_TEMPLATE = """<?xml version="1.0"?>
<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">
  <ImageData WholeExtent="{extent}" Origin="{origin}" Spacing="{spacing}">
    <Piece Extent="{extent}">
      <CellData Scalars="{scalars}">
{arrays}
      </CellData>
    </Piece>
  </ImageData>
</VTKFile>
"""
ARRAY_TEMPLATE = ('        <DataArray type="Float32" Name="{name}" '
                  'NumberOfComponents="1" format="ascii">{text}</DataArray>')
VALUE_FORMAT = "%.6g"


def build_array_text(values: np.ndarray) -> str:
    """VTK walks x fastest then y then z upward, while the pipeline indexes
    (depth, row, column) with depth 0 at the surface, hence the same flip as the
    CPlantBox lookup grids."""
    return " ".join(VALUE_FORMAT % value for value in values[::-1, :, :].ravel())


def write_voxel_grids(path: Path, fields: dict[str, np.ndarray], grid: VoxelGrid) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = "\n".join(ARRAY_TEMPLATE.format(name=name, text=build_array_text(values))
                       for name, values in fields.items())
    path.write_text(DOCUMENT_TEMPLATE.format(
        extent=f"0 {grid.column_cells} 0 {grid.row_cells} 0 {grid.depth_cells}",
        origin=f"{-grid.half_width_cm:g} {-grid.half_length_cm:g} {-grid.depth_cm:g}",
        spacing=f"{grid.cell_width_cm:g} {grid.cell_width_cm:g} {grid.cell_height_cm:g}",
        scalars=next(iter(fields)),
        arrays=arrays), encoding="utf-8")
    return path
