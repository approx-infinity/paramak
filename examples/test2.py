# Naming tokamak with several customizations

'''
Some reactor components may contain multiple disconnected solids. To view the name assigned to each solid individually, first call my_reactor.split_solids() and then use my_reactor.names() on the returned reactor object.

For example, a toroidal field (TF) coil set may be represented as a single component but consist of eight separate TF coils. After splitting the solids, names() will return the names of all eight coils in the order they appear in the geometry.
'''

import paramak
import cadquery as cq

# Create divertor shape
points = [(450, -700), (450, 0), (550, 0), (550, -700)]
divertor_lower = cq.Workplane("XZ", origin=(0, 0, 0)).polyline(points).close().revolve(180)

# Create TF coil shape
tf = paramak.toroidal_field_coil_princeton_d(
    r1=200,
    r2=1000,
    thickness=50,
    distance=60,
    rotation_angle=180,
    with_inner_leg=True,
    azimuthal_placement_angles=[0, 30, 60, 90, 120, 150, 180],
)

extra_cut_shapes = [tf]

# Create PF coils and their cases
for case_thickness, height, width, center_point in zip(
    [10, 15, 15, 10], [20, 50, 50, 20], [20, 50, 50, 20], [(1030, 450), (1110, 250), (1110, -250), (1030, -450)]
):
    extra_cut_shapes.append(
        paramak.poloidal_field_coil(
            height=height, 
            width=width, 
            center_point=center_point, 
            rotation_angle=180,
        )
    )
    extra_cut_shapes.append(
        paramak.poloidal_field_coil_case(
            coil_height=height,
            coil_width=width,
            casing_thickness=case_thickness,
            rotation_angle=180,
            center_point=center_point,
        )
    )

my_reactor = paramak.tokamak_from_plasma(
    radial_build=[
        (paramak.LayerType.GAP, 150),
        (paramak.LayerType.SOLID, 50),
        (paramak.LayerType.GAP, 80),
        (paramak.LayerType.SOLID, 10),
        (paramak.LayerType.SOLID, 60),
        (paramak.LayerType.SOLID, 60),
        (paramak.LayerType.SOLID, 10),
        (paramak.LayerType.GAP, 60),
        (paramak.LayerType.PLASMA, 300),
        (paramak.LayerType.GAP, 60),
        (paramak.LayerType.SOLID, 10),
        (paramak.LayerType.SOLID, 60),
        (paramak.LayerType.SOLID, 60),
        (paramak.LayerType.SOLID, 10),
    ],
    rotation_angle=180,
    extra_cut_shapes=extra_cut_shapes,
    extra_intersect_shapes=[divertor_lower],
)

my_reactor = (
    my_reactor
    .rename("extra_intersect_shapes", "divertor")
    .rename("toroidal_field_coil", "toroidal_coil")
    .rename("poloidal_field_coil", "poloidal_coil")
    .rename("poloidal_field_coil_case", "poloidal_coil_case")
    .rename("layer_1", "CS Coil")
    .rename("layer_2", "W Armor")
    .rename("layer_3", "First Wall")
    .rename("layer_4", "Blanket")
    .rename("layer_5", "Vacuum Vessel")
)

# Use split_solids() to inspect individual solid names before assigning material tags.
my_reactor = my_reactor.split_solids()
print(my_reactor.names())
'''
['toroidal_coil_1_1', 'toroidal_coil_1_2', 'toroidal_coil_1_3', 'toroidal_coil_1_4', 'toroidal_coil_1_5', 
'toroidal_coil_1_6', 'toroidal_coil_1_7', 'poloidal_coil_2', 'poloidal_coil_case_3', 'poloidal_coil_4', 
'poloidal_coil_case_5', 'poloidal_coil_6', 'poloidal_coil_case_7', 'poloidal_coil_8', 'poloidal_coil_case_9', 
'divertor_1', 'CS Coil', 'W Armor', 'First Wall', 'Blanket', 'Vacuum Vessel', 'plasma']
'''