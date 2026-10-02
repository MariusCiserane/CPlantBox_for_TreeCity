"""Write voxel fields as a .vti, to open in ParaView next to the root .vtp files, and read them
back: the .vti is the only float32 copy of the fields CPlantBox actually read.
"""
import re
from pathlib import Path

import numpy as np

from grid import VoxelGrid

DOCUMENT_TEMPLATE = """<?xml version="1.0"?>
<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian" header_type="UInt32">
  <ImageData WholeExtent="{extent}" Origin="{origin}" Spacing="{spacing}">
    <Piece Extent="{extent}">
      <CellData Scalars="{scalars}">
{arrays}
      </CellData>
    </Piece>
  </ImageData>
  <AppendedData encoding="raw">
_"""
DOCUMENT_TAIL = """
  </AppendedData>
</VTKFile>
"""
ARRAY_TEMPLATE = ('        <DataArray type="Float32" Name="{name}" NumberOfComponents="1" '
                  'format="appended" offset="{offset}"/>')
ARRAY_PATTERN = re.compile(rb'Name="([^"]+)"[^>]*offset="(\d+)"')
APPENDED_MARKER = b'<AppendedData encoding="raw">\n_'
LENGTH_PREFIX_BYTES = 4


def build_array_bytes(values: np.ndarray) -> bytes:
    """VTK walks x fastest then y then z upward, while the pipeline indexes
    (depth, row, column) with depth 0 at the surface, hence the flip."""
    payload = np.ascontiguousarray(values[::-1, :, :].ravel(), np.float32).tobytes()
    return np.uint32(len(payload)).tobytes() + payload


def write_voxel_grids(path: Path, fields: dict, grid: VoxelGrid) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payloads, declarations, offset = [], [], 0
    for name, values in fields.items():
        declarations.append(ARRAY_TEMPLATE.format(name=name, offset=offset))
        payloads.append(build_array_bytes(values))
        offset += len(payloads[-1])

    header = DOCUMENT_TEMPLATE.format(
        extent=f"0 {grid.column_cells} 0 {grid.row_cells} 0 {grid.depth_cells}",
        origin=f"{-grid.half_width_cm:g} {-grid.half_length_cm:g} {-grid.depth_cm:g}",
        spacing=f"{grid.cell_width_cm:g} {grid.cell_width_cm:g} {grid.cell_height_cm:g}",
        scalars=next(iter(fields)),
        arrays="\n".join(declarations))
    path.write_bytes(header.encode("utf-8") + b"".join(payloads) + DOCUMENT_TAIL.encode("utf-8"))
    return path


def read_voxel_grids(path: Path, grid: VoxelGrid) -> dict:
    """The exact inverse of write_voxel_grids, depth flip included."""
    content = path.read_bytes()
    start = content.index(APPENDED_MARKER) + len(APPENDED_MARKER)
    fields = {}
    for name, offset in ARRAY_PATTERN.findall(content[:start]):
        position = start + int(offset)
        size = int(np.frombuffer(content, np.uint32, 1, position)[0])
        values = np.frombuffer(content, np.float32, size // np.float32().itemsize,
                               position + LENGTH_PREFIX_BYTES)
        fields[name.decode()] = values.reshape(grid.shape)[::-1, :, :].copy()
    return fields
