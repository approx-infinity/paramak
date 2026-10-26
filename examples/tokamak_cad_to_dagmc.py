"""Create a DAGMC H5M file from the tokamak CAD assembly with cad_to_dagmc.

This uses the same reactor, divertor, and coil geometry as
``tokamak_structured_surface_mesh.py``. Unlike that example, cad_to_dagmc
meshes the CadQuery solids directly. Assembly part names are used as DAGMC
material tags.

Requires the optional ``cad_to_dagmc`` dependencies, including Gmsh.
"""

from pathlib import Path

from cad_to_dagmc import CadToDagmc

import paramak
from time import perf_counter


radial_build = [
    (paramak.LayerType.GAP, 10),
    (paramak.LayerType.SOLID, 30),
    (paramak.LayerType.SOLID, 50),
    (paramak.LayerType.GAP, 10),
    (paramak.LayerType.SOLID, 10),
    (paramak.LayerType.GAP, 10),
    (paramak.LayerType.SOLID, 30),
    (paramak.LayerType.SOLID, 50),
    (paramak.LayerType.SOLID, 2),
    (paramak.LayerType.SOLID, 0.2),
    (paramak.LayerType.GAP, 60),
    (paramak.LayerType.PLASMA, 300),
    (paramak.LayerType.GAP, 60),
    (paramak.LayerType.SOLID, 0.2, "armor"),
    (paramak.LayerType.SOLID, 2, "first_wall"),
    (paramak.LayerType.SOLID, 50, "bz"),
    (paramak.LayerType.SOLID, 30, "bss"),
    (paramak.LayerType.GAP, 10),
    (paramak.LayerType.SOLID, 10, "vv"),
]

arc_length = paramak.poloidal_arc_length(radial_build, elongation=2.0, triangularity=0.55)
segment_length = (arc_length - 20) / 2
first_wall_segments = [
    (paramak.LayerType.SOLID, segment_length, "upper"),
    (paramak.LayerType.GAP, 10),
    (paramak.LayerType.SOLID, segment_length, "lower"),
    (paramak.LayerType.GAP, 10),
]

rotation_angle = 180
toroidal_arc = paramak.toroidal_arc_length(radial_build, rotation_angle=rotation_angle)
number_of_sectors = 4
sector_gap = 5
sector_length = (toroidal_arc - number_of_sectors * sector_gap) / number_of_sectors
toroidal_sectors = [
    (paramak.LayerType.SOLID, sector_length, "sector"),
    (paramak.LayerType.GAP, sector_gap),
] * number_of_sectors

reactor = paramak.tokamak_from_plasma(
    radial_build=radial_build,
    poloidal_build=[None, first_wall_segments, None, None, None, None, None],
    toroidal_build=[None, toroidal_sectors, toroidal_sectors, None, None, None, None],
    elongation=2.0,
    triangularity=0.55,
    rotation_angle=rotation_angle,
)
reactor = reactor.add_divertor(
    poloidal_start_angle=75,
    poloidal_extent_angle=30,
    divertor_layer_names=[
        "upper_divertor_armor",
        "upper_divertor_heatsink",
        "upper_divertor_shield",
    ],
    divertor_layer_thicknesses=[10, 30, 50],
)
reactor = reactor.add_divertor(
    poloidal_start_angle=255,
    poloidal_extent_angle=30,
    divertor_layer_names=[
        "lower_divertor_armor",
        "lower_divertor_heatsink",
        "lower_divertor_shield",
    ],
    divertor_layer_thicknesses=[10, 30, 50],
)

toroidal_coil = paramak.toroidal_field_coil_rectangle(
    horizontal_start_point=(850, 800),
    vertical_mid_point=(1100, 0),
    thickness=40,
    distance=30,
    rotation_angle=rotation_angle,
    azimuthal_placement_angles=[0, 60, 120, 180],
    name="toroidal_field_coil",
)
reactor.add(toroidal_coil, name="toroidal_field_coil")

for index, vertical_position in enumerate((650, -650), start=1):
    poloidal_coil = paramak.poloidal_field_coil(
        height=50,
        width=50,
        center_point=(1250, vertical_position),
        rotation_angle=rotation_angle,
        name=f"poloidal_field_coil_{index}",
    )
    reactor.add(poloidal_coil, name=poloidal_coil.name)

# The plasma is not a solid material region for this DAGMC model.
reactor = reactor.remove(name="plasma")

dagmc_model = CadToDagmc()
# Match the solids in the CadQuery compound order. Explicit tags avoid the
# UUID-prefixed names produced by cad_to_dagmc's ``assembly_names`` option for
# nested assemblies. Toroidal sectors are split into inboard/outboard solids.
material_tags = [
    "layer_1",
    "layer_2",
    "armor_upper_s1_inboard",
    "armor_upper_s1_outboard",
    "armor_upper_s2_inboard",
    "armor_upper_s2_outboard",
    "armor_upper_s3_inboard",
    "armor_upper_s3_outboard",
    "armor_upper_s4_inboard",
    "armor_upper_s4_outboard",
    "armor_lower_s1_inboard",
    "armor_lower_s1_outboard",
    "armor_lower_s2_inboard",
    "armor_lower_s2_outboard",
    "armor_lower_s3_inboard",
    "armor_lower_s3_outboard",
    "armor_lower_s4_inboard",
    "armor_lower_s4_outboard",
    "first_wall_s1_inboard",
    "first_wall_s1_outboard",
    "first_wall_s2_inboard",
    "first_wall_s2_outboard",
    "first_wall_s3_inboard",
    "first_wall_s3_outboard",
    "first_wall_s4_inboard",
    "first_wall_s4_outboard",
    "bz_inboard",
    "bz_outboard",
    "bss_inboard",
    "bss_outboard",
    "vv",
    "upper_divertor_armor",
    "upper_divertor_heatsink",
    "upper_divertor_shield",
    "lower_divertor_armor",
    "lower_divertor_heatsink",
    "lower_divertor_shield",
    "toroidal_field_coil",
    "toroidal_field_coil",
    "toroidal_field_coil",
    "toroidal_field_coil",
    "poloidal_field_coil_1",
    "poloidal_field_coil_2",
]
dagmc_model.add_cadquery_object(
    cadquery_object=reactor,
    material_tags=material_tags,
)

output_file = Path(__file__).resolve().parent / "tokamak_cad_to_dagmc.h5m"
start_time = perf_counter()
dagmc_model.export_dagmc_h5m_file(
    filename=output_file,
    meshing_backend="cad-to-dagmc-mesher",
    min_mesh_size=10.0,
    max_mesh_size=20.0,
)
elapsed = perf_counter() - start_time
print(f"Wrote {output_file} in {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
print(f"Wrote {output_file}")
