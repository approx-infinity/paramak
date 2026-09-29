import pytest

import paramak


def test_colors():
    "passing in the colors dictionary should not raise an error"

    paramak.tokamak_from_plasma(
        radial_build=[
            (paramak.LayerType.GAP, 10),
            (paramak.LayerType.SOLID, 30),
            (paramak.LayerType.SOLID, 50),
            (paramak.LayerType.SOLID, 10),
            (paramak.LayerType.SOLID, 120),
            (paramak.LayerType.SOLID, 20),
            (paramak.LayerType.GAP, 60),
            (paramak.LayerType.PLASMA, 300),
            (paramak.LayerType.GAP, 60),
            (paramak.LayerType.SOLID, 20),
            (paramak.LayerType.SOLID, 120),
            (paramak.LayerType.SOLID, 10),
        ],
        elongation=2,
        triangularity=0.55,
        rotation_angle=180,
        colors={
            "layer_1": (0.4, 0.9, 0.4),
            "layer_2": (0.6, 0.8, 0.6),
            "plasma": (1., 0.7, 0.8, 0.6),
            "layer_3": (0.1, 0.1, 0.9),
            "layer_4": (0.4, 0.4, 0.8),
            "layer_5": (0.5, 0.5, 0.8),
        }
    )


def test_layer_names_are_contiguous_with_interior_gaps():
    "layer names should be sequential layer_1..layer_N even when gaps sit between solid layers"

    my_reactor = paramak.tokamak(
        radial_build=[
            (paramak.LayerType.GAP, 50),
            (paramak.LayerType.SOLID, 50),
            (paramak.LayerType.GAP, 10),
            (paramak.LayerType.SOLID, 10),
            (paramak.LayerType.SOLID, 60),
            (paramak.LayerType.GAP, 20),
            (paramak.LayerType.SOLID, 60),
            (paramak.LayerType.SOLID, 10),
            (paramak.LayerType.GAP, 60),
            (paramak.LayerType.PLASMA, 300),
            (paramak.LayerType.GAP, 60),
            (paramak.LayerType.SOLID, 10),
            (paramak.LayerType.SOLID, 60),
            (paramak.LayerType.GAP, 20),
            (paramak.LayerType.SOLID, 60),
            (paramak.LayerType.SOLID, 10),
        ],
        vertical_build=[
            (paramak.LayerType.SOLID, 10),
            (paramak.LayerType.SOLID, 60),
            (paramak.LayerType.GAP, 20),
            (paramak.LayerType.SOLID, 60),
            (paramak.LayerType.SOLID, 10),
            (paramak.LayerType.GAP, 60),
            (paramak.LayerType.PLASMA, 650),
            (paramak.LayerType.GAP, 60),
            (paramak.LayerType.SOLID, 10),
            (paramak.LayerType.SOLID, 60),
            (paramak.LayerType.GAP, 20),
            (paramak.LayerType.SOLID, 60),
            (paramak.LayerType.SOLID, 10),
        ],
        rotation_angle=180,
    )

    assert my_reactor.names() == ["layer_1", "layer_2", "layer_3", "layer_4", "layer_5", "plasma"]


def test_named_layers_tokamak():
    "layers can be named in the radial_build, or with rename() after building"

    from_radial_build = paramak.tokamak_from_plasma(
        radial_build=[
            (paramak.LayerType.GAP, 10),
            (paramak.LayerType.SOLID, 30, "central column"),
            (paramak.LayerType.SOLID, 20, "blanket"),
            (paramak.LayerType.SOLID, 10, "first wall"),
            (paramak.LayerType.GAP, 60),
            (paramak.LayerType.PLASMA, 300),
            (paramak.LayerType.GAP, 60),
            (paramak.LayerType.SOLID, 20),
            (paramak.LayerType.SOLID, 10),
        ],
        rotation_angle=180,
    )
    assert from_radial_build.names() == ["central column", "first wall", "blanket", "plasma"]

    renamed = (
        paramak.tokamak_from_plasma(
            radial_build=[
                (paramak.LayerType.GAP, 10),
                (paramak.LayerType.SOLID, 30),
                (paramak.LayerType.SOLID, 20),
                (paramak.LayerType.SOLID, 10),
                (paramak.LayerType.GAP, 60),
                (paramak.LayerType.PLASMA, 300),
                (paramak.LayerType.GAP, 60),
                (paramak.LayerType.SOLID, 20),
                (paramak.LayerType.SOLID, 10),
            ],
            rotation_angle=180,
        )
        .rename("layer_1", "central column")
        .rename("layer_2", "first wall")
        .rename("layer_3", "blanket")
    )
    assert renamed.names() == ["central column", "first wall", "blanket", "plasma"]


def test_poloidal_arc_length_and_segmentation():
    helper_arc_lengths = paramak.poloidal_arc_length(
        offsets=[0, 10],
        major_radius=450,
        minor_radius=150,
        triangularity=0.55,
        elongation=2.0,
    )

    assert isinstance(helper_arc_lengths, list)
    assert len(helper_arc_lengths) == 2
    assert helper_arc_lengths[0] > 0
    assert helper_arc_lengths[1] > helper_arc_lengths[0]

    reactor_arc_length = paramak.poloidal_arc_length(
        offsets=0,
        major_radius=180,
        minor_radius=150,
        triangularity=0.55,
        elongation=2.0,
    )

    reactor = paramak.tokamak_from_plasma(
        radial_build=[
            (paramak.LayerType.GAP, 10),
            (paramak.LayerType.SOLID, 20),
            (paramak.LayerType.PLASMA, 300),
            (paramak.LayerType.SOLID, 20),
            (paramak.LayerType.GAP, 10),
        ],
        poloidal_build=[
            [
                    ("outboard", reactor_arc_length / 2),
                    ("inboard", reactor_arc_length / 2),
            ],
            None,
        ],
        elongation=2.0,
        triangularity=0.55,
        rotation_angle=180,
    )

    assert reactor.names() == ["layer_1_outboard", "layer_1_inboard", "plasma"]


def test_poloidal_build_validation():
    arc_length_at_reactor = paramak.poloidal_arc_length(
        offsets=0,
        major_radius=180,
        minor_radius=150,
        triangularity=0.55,
        elongation=2.0,
    )

    with pytest.raises(ValueError, match="poloidal_build must contain"):
        paramak.tokamak_from_plasma(
            radial_build=[
                (paramak.LayerType.GAP, 10),
                (paramak.LayerType.SOLID, 20),
                (paramak.LayerType.PLASMA, 300),
                (paramak.LayerType.SOLID, 20),
                (paramak.LayerType.GAP, 10),
            ],
            poloidal_build=[[("outboard", 1.0)]],
            rotation_angle=180,
        )

    with pytest.raises(ValueError, match="poloidal_build must be None for GAP entries"):
        paramak.tokamak_from_plasma(
            radial_build=[
                (paramak.LayerType.GAP, 10),
                (paramak.LayerType.SOLID, 20),
                (paramak.LayerType.PLASMA, 300),
                (paramak.LayerType.SOLID, 20),
                (paramak.LayerType.GAP, 10),
            ],
            poloidal_build=[
                [("outboard", arc_length_at_reactor / 2), ("gap", 0.0), ("inboard", arc_length_at_reactor / 2)],
                [("gap", 0.0)],
            ],
            rotation_angle=180,
        )


def test_multi_layer_poloidal_segmentation():
    radial_build = [
        (paramak.LayerType.GAP, 10),
        (paramak.LayerType.SOLID, 20, "blanket"),
        (paramak.LayerType.SOLID, 10, "first wall"),
        (paramak.LayerType.GAP, 40),
        (paramak.LayerType.PLASMA, 300),
        (paramak.LayerType.GAP, 40),
        (paramak.LayerType.SOLID, 10),
        (paramak.LayerType.SOLID, 20),
        (paramak.LayerType.GAP, 10),
    ]

    base = paramak.tokamak_from_plasma(
        radial_build=radial_build,
        elongation=2.0,
        triangularity=0.55,
        rotation_angle=180,
    )

    arcs = paramak.poloidal_arc_length(
        offsets=[40, 50],
        major_radius=base.major_radius,
        minor_radius=base.minor_radius,
        triangularity=0.55,
        elongation=2.0,
    )

    gap_mm = 50.0
    first_wall_total = arcs[0] - 2 * gap_mm
    blanket_total = arcs[1] - 2 * gap_mm

    segmented = paramak.tokamak_from_plasma(
        radial_build=radial_build,
        poloidal_build=[
            None,
            [
                ("outboard", first_wall_total * 0.2),
                ("gap", gap_mm),
                ("inboard", first_wall_total * 0.6),
                ("gap", gap_mm),
                ("outboard", first_wall_total * 0.2),
            ],
            [
                ("outboard", blanket_total * 0.2),
                ("gap", gap_mm),
                ("inboard", blanket_total * 0.6),
                ("gap", gap_mm),
                ("outboard", blanket_total * 0.2),
            ],
            None,
        ],
        elongation=2.0,
        triangularity=0.55,
        rotation_angle=180,
    )

    assert segmented.names() == [
        "first wall_outboard_1",
        "first wall_inboard",
        "first wall_outboard_2",
        "blanket_outboard_1",
        "blanket_inboard",
        "blanket_outboard_2",
        "plasma",
    ]