"""
Example: multi-layer poloidal segmentation for `tokamak_from_plasma`.

Demonstrates:
- computing poloidal arc lengths with `poloidal_arc_length`
- using cumulative offsets from the plasma surface
- building `poloidal_build` for more than one layer
- repeating segment names and inserting gaps
- inspecting generated part names

Run:
    python examples/poloidal_segmentation_example.py
"""

import paramak

# Reactor geometry parameters
elongation = 2.0
triangularity = 0.55
rotation_angle = 180.0

# This build gives major_radius=230 mm and minor_radius=150 mm.
# The first solid pair starts 40 mm from the plasma surface.
# The blanket pair starts at 50 mm (40 + 10).
radial_build = [
    (paramak.LayerType.GAP, 100),
    (paramak.LayerType.SOLID, 20, "cs coil"),
    (paramak.LayerType.GAP, 10),
    (paramak.LayerType.SOLID, 50, "vaccum vessel"),
    (paramak.LayerType.SOLID, 60, "blanket"),
    (paramak.LayerType.SOLID, 10, "first wall"),
    (paramak.LayerType.GAP, 40),
    (paramak.LayerType.PLASMA, 300),
    (paramak.LayerType.GAP, 40),
    (paramak.LayerType.SOLID, 10),
    (paramak.LayerType.SOLID, 60),
    (paramak.LayerType.SOLID, 50),
]

# Step 1: create a baseline tokamak so we can read computed radii
base = paramak.tokamak_from_plasma(
    radial_build=radial_build,
    elongation=elongation,
    triangularity=triangularity,
    rotation_angle=rotation_angle,
)

major_radius = base.major_radius
minor_radius = base.minor_radius
print(f"Computed major_radius={major_radius}, minor_radius={minor_radius}")

# Step 2: compute the poloidal arc lengths at the inner surfaces of the
# two layers we want to segment. Offsets are cumulative from the plasma
# surface, so the blanket begins at 40 + 10 = 50 mm.
arc = paramak.poloidal_arc_length(
    offsets=[40, 50],
    major_radius=major_radius,
    minor_radius=minor_radius,
    triangularity=triangularity,
    elongation=elongation,
)
print(f"Poloidal arc length at offset 40 mm: {arc[0]:.6f} mm")
print(f"Poloidal arc length at offset 50 mm: {arc[1]:.6f} mm")

# Step 3: design segmentations for both layers.
gap_mm = 50.0
first_wall_total = arc[0] - 2 * gap_mm
blanket_total = arc[1] - 2 * gap_mm

first_wall_outboard = first_wall_total * 0.25
first_wall_inboard = first_wall_total * 0.50
first_wall_inboard_segment = first_wall_inboard / 5

blanket_outboard = blanket_total * 0.25
blanket_inboard = blanket_total * 0.50
blanket_inboard_segment = blanket_inboard / 5

# layout measured from outboard midplane, counter-clockwise
poloidal_build = [
    None,  # outer gap pair
    [
        ("outboard", first_wall_outboard),
        ("gap", gap_mm),
        ("inboard", first_wall_inboard_segment),
        ("inboard", first_wall_inboard_segment),
        ("inboard", first_wall_inboard_segment),
        ("inboard", first_wall_inboard_segment),
        ("inboard", first_wall_inboard_segment),
        ("gap", gap_mm),
        ("outboard", first_wall_outboard),
    ],
    [
        ("outboard", blanket_outboard),
        ("gap", gap_mm),
        ("inboard", blanket_inboard_segment),
        ("inboard", blanket_inboard_segment),
        ("inboard", blanket_inboard_segment),
        ("inboard", blanket_inboard_segment),
        ("inboard", blanket_inboard_segment),
        ("gap", gap_mm),
        ("outboard", blanket_outboard),
    ],
    None,
]

# Step 4: build reactor with poloidal segmentation
segmented = paramak.tokamak_from_plasma(
    radial_build=radial_build,
    poloidal_build=poloidal_build,
    elongation=elongation,
    triangularity=triangularity,
    rotation_angle=rotation_angle,
)

# Inspect names (segments are disambiguated if repeated)
print("Generated parts:")
for name in segmented.names():
    print(" -", name)

# Optionally save a STEP for viewing
try:
    segmented.save("tokamak_poloidal_segmented.step")
    print("Saved tokamak_poloidal_segmented.step")
except Exception:
    print("STEP export failed (cadquery / STEP writer not available).")
