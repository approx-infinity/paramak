from collections import Counter
from typing import Sequence, Tuple

import cadquery as cq
import numpy as np
import warnings
from .assembly import Assembly

from ..utils import (
    get_plasma_index, 
    get_layer_name, 
    get_assembly_names, 
    validate_vertical_build_names, 
    validate_unique_assembly_names, 
    LayerType
)
from ..workplanes.blanket_from_plasma import blanket_from_plasma, create_offset_points
from ..workplanes.center_column_shield_cylinder import center_column_shield_cylinder
from ..workplanes.plasma_simplified import plasma_simplified
from .spherical_tokamak import get_plasma_value, sum_up_to_plasma


_POLOIDAL_ARC_LENGTH_NUM_POINTS = 720


def _as_offset_list(offsets):
    if isinstance(offsets, (list, tuple)):
        return list(offsets), True
    return [offsets], False


def _poloidal_arc_length_profile(
    offset,
    major_radius,
    minor_radius,
    triangularity,
    elongation,
    num_points=_POLOIDAL_ARC_LENGTH_NUM_POINTS,
):
    thetas = np.linspace(0.0, 360.0, num=num_points, endpoint=True)
    points, overlapping_shape = create_offset_points(
        major_radius=major_radius,
        minor_radius=minor_radius,
        triangularity=triangularity,
        elongation=elongation,
        vertical_displacement=0.0,
        thetas=thetas,
        offset=lambda theta: offset,
    )

    if overlapping_shape:
        raise ValueError(
            f"poloidal_arc_length cannot be computed for offset {offset} because the resulting shape has negative R coordinates."
        )

    if len(points) != len(thetas):
        raise ValueError(
            f"poloidal_arc_length requires a non-overlapping closed loop; received {len(points)} points for {len(thetas)} angles."
        )

    coordinates = np.asarray([(point[0], point[1]) for point in points], dtype=float)
    segment_lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    cumulative_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    return thetas, cumulative_lengths


def poloidal_arc_length(
    offsets,
    major_radius,
    minor_radius,
    triangularity,
    elongation,
    num_points=_POLOIDAL_ARC_LENGTH_NUM_POINTS,
):
    """Return the poloidal arc length at one or more offsets from the plasma surface."""

    offsets_list, is_sequence = _as_offset_list(offsets)
    arc_lengths = []
    for offset in offsets_list:
        _, cumulative_lengths = _poloidal_arc_length_profile(
            offset=offset,
            major_radius=major_radius,
            minor_radius=minor_radius,
            triangularity=triangularity,
            elongation=elongation,
            num_points=num_points,
        )
        arc_lengths.append(float(cumulative_lengths[-1]))

    return arc_lengths if is_sequence else arc_lengths[0]


def _validate_poloidal_build(radial_build, poloidal_build):
    if poloidal_build is None:
        return

    plasma_index = get_plasma_index(radial_build)
    expected_length = len(radial_build) - plasma_index - 1
    if len(poloidal_build) != expected_length:
        raise ValueError(
            f"poloidal_build must contain {expected_length} entries for tokamak_from_plasma(), one for each radial pair after the plasma entry, not {len(poloidal_build)}."
        )


def _segment_name(layer_name, segment_name, segment_counts, segment_index):
    if segment_counts[segment_name] > 1:
        return f"{layer_name}_{segment_name}_{segment_index}"
    return f"{layer_name}_{segment_name}"


def _create_segmented_layer(
    layer_name,
    segment_build,
    minor_radius,
    major_radius,
    triangularity,
    elongation,
    rotation_angle,
    inner_offset,
    upper_layer_thickness,
    lower_layer_thickness,
    inner_layer_thickness,
    outer_layer_thickness,
    cumulative_thickness_uvb,
    cumulative_thickness_lvb,
    cumulative_thickness_irb,
    cumulative_thickness_orb,
):
    segment_counts = Counter(name for name, thickness in segment_build if name != "gap")
    segment_occurrence_counts = Counter()

    segment_start_lengths = [0.0]
    for _, thickness in segment_build:
        if thickness <= 0:
            raise ValueError(f"poloidal_build entries must have positive lengths, not {thickness}.")
        segment_start_lengths.append(segment_start_lengths[-1] + thickness)

    total_arc_length = segment_start_lengths[-1]
    thetas, cumulative_lengths = _poloidal_arc_length_profile(
        offset=inner_offset,
        major_radius=major_radius,
        minor_radius=minor_radius,
        triangularity=triangularity,
        elongation=elongation,
    )

    actual_arc_length = float(cumulative_lengths[-1])
    if abs(total_arc_length - actual_arc_length) > 1e-3:
        raise ValueError(
            f"poloidal_build for {layer_name} sums to {total_arc_length} mm, but the arc length at the layer's inner surface (offset {inner_offset}) is {actual_arc_length:.6f} mm."
        )

    layers = []

    def thickness_profile(theta):
        theta = theta % 360.0
        return float(
            np.interp(
                theta,
                [0.0, 90.0, 180.0, 270.0, 360.0],
                [
                    outer_layer_thickness,
                    upper_layer_thickness,
                    inner_layer_thickness,
                    lower_layer_thickness,
                    outer_layer_thickness,
                ],
            )
        )

    def offset_profile(theta):
        theta = theta % 360.0
        return float(
            np.interp(
                theta,
                [0.0, 90.0, 180.0, 270.0, 360.0],
                [
                    cumulative_thickness_orb,
                    cumulative_thickness_uvb,
                    cumulative_thickness_irb,
                    cumulative_thickness_lvb,
                    cumulative_thickness_orb,
                ],
            )
        )

    for segment_index, (segment_name, segment_length) in enumerate(segment_build):
        start_length = segment_start_lengths[segment_index]
        stop_length = segment_start_lengths[segment_index + 1]

        if segment_name == "gap":
            continue

        segment_occurrence_counts[segment_name] += 1
        name = _segment_name(
            layer_name=layer_name,
            segment_name=segment_name,
            segment_counts=segment_counts,
            segment_index=segment_occurrence_counts[segment_name],
        )

        start_angle = float(np.interp(start_length, cumulative_lengths, thetas))
        stop_angle = float(np.interp(stop_length, cumulative_lengths, thetas))

        segment = blanket_from_plasma(
            minor_radius=minor_radius,
            major_radius=major_radius,
            triangularity=triangularity,
            elongation=elongation,
            thickness=thickness_profile,
            offset_from_plasma=offset_profile,
            start_angle=start_angle,
            stop_angle=stop_angle,
            rotation_angle=rotation_angle,
            color=(0.5, 0.5, 0.5),
            name=name,
            allow_overlapping_shape=True,
        )
        segment.name = name
        layers.append(segment)

    return layers


def count_cylinder_layers(radial_build):
    before_plasma = 0
    after_plasma = 0
    found_plasma = False

    for item in radial_build:
        if item[0] == LayerType.PLASMA:
            found_plasma = True
        elif item[0] == LayerType.SOLID:
            if not found_plasma:
                before_plasma += 1
            else:
                after_plasma += 1

    return before_plasma - after_plasma


def create_center_column_shield_cylinders(radial_build, rotation_angle, center_column_shield_height):
    cylinders = []
    total_sum = 0
    layer_count = 0

    number_of_cylinder_layers = count_cylinder_layers(radial_build)

    for _, item in enumerate(radial_build):
        if item[0] == LayerType.PLASMA:
            break

        if item[0] == LayerType.GAP:
            total_sum += item[1]
            continue

        thickness = item[1]
        layer_count += 1

        if layer_count > number_of_cylinder_layers:
            break

        layer_name = get_layer_name(item, layer_count)

        cylinder = center_column_shield_cylinder(
            inner_radius=total_sum,
            thickness=item[1],
            name=layer_name,
            rotation_angle=rotation_angle,
            height=center_column_shield_height,
        )
        total_sum += thickness
        cylinders.append(cylinder)
    return cylinders


def distance_to_plasma(radial_build, index):
    distance = 0
    for item in radial_build[index + 1 :]:
        if item[0] == LayerType.PLASMA:
            break
        distance += item[1]
    return distance


def create_layers_from_plasma(
    radial_build,
    vertical_build,
    minor_radius,
    major_radius,
    triangularity,
    elongation,
    rotation_angle,
    center_column,
    poloidal_build=None,
    layer_count=0,
):

    plasma_index_rb = get_plasma_index(radial_build)
    plasma_index_vb = get_plasma_index(vertical_build)
    indexes_from_plasma_to_end = len(radial_build) - plasma_index_rb
    layers = []

    _validate_poloidal_build(radial_build, poloidal_build)
    if poloidal_build is not None:
        warnings.warn(
            "poloidal_build provided: segment lengths are measured along the layer inner surface; "
            "the inner arc is not automatically added. GAP entries must have a corresponding None in poloidal_build.",
            UserWarning,
        )

    cumulative_thickness_orb = 0
    cumulative_thickness_irb = 0
    cumulative_thickness_uvb = 0
    cumulative_thickness_lvb = 0

    for index_delta in range(indexes_from_plasma_to_end):

        if radial_build[plasma_index_rb + index_delta][0] == LayerType.PLASMA:
            continue
        outer_layer_thickness = radial_build[plasma_index_rb + index_delta][1]
        inner_layer_thickness = radial_build[plasma_index_rb - index_delta][1]
        upper_layer_thickness = vertical_build[plasma_index_vb - index_delta][1]
        lower_layer_thickness = vertical_build[plasma_index_vb + index_delta][1]
        poloidal_entry = None
        if poloidal_build is not None and index_delta > 0:
            poloidal_entry = poloidal_build[index_delta - 1]

        if radial_build[plasma_index_rb + index_delta][0] == LayerType.GAP:
            if poloidal_entry is not None:
                raise ValueError(
                    f"poloidal_build entries corresponding to GAP layers must be None; received {poloidal_entry} for radial layer index {index_delta}."
                )
            cumulative_thickness_orb += outer_layer_thickness
            cumulative_thickness_irb += inner_layer_thickness
            cumulative_thickness_uvb += upper_layer_thickness
            cumulative_thickness_lvb += lower_layer_thickness
            continue

        layer_count += 1
        if len(radial_build[plasma_index_rb - index_delta]) == 3:
            layer_name = radial_build[plasma_index_rb - index_delta][2]
        elif len(radial_build[plasma_index_rb + index_delta]) == 3:
            layer_name = radial_build[plasma_index_rb + index_delta][2]
        else:
            layer_name = f"layer_{layer_count}"

        # build outer layer
        if radial_build[plasma_index_rb + index_delta][0] == LayerType.SOLID:
            if poloidal_entry is None:
                outer_layer = blanket_from_plasma(
                    minor_radius=minor_radius,
                    major_radius=major_radius,
                    triangularity=triangularity,
                    elongation=elongation,
                    thickness=[upper_layer_thickness, outer_layer_thickness, lower_layer_thickness],
                    offset_from_plasma=[cumulative_thickness_uvb, cumulative_thickness_orb, cumulative_thickness_lvb],
                    start_angle=90,
                    stop_angle=-90,
                    rotation_angle=rotation_angle,
                    color=(0.5, 0.5, 0.5),
                    name=layer_name,
                    allow_overlapping_shape=True,
                )
                inner_layer = blanket_from_plasma(
                    minor_radius=minor_radius,
                    major_radius=major_radius,
                    triangularity=triangularity,
                    elongation=elongation,
                    thickness=[
                        lower_layer_thickness,
                        inner_layer_thickness,
                        upper_layer_thickness,
                    ],
                    offset_from_plasma=[
                        cumulative_thickness_lvb,
                        cumulative_thickness_irb,
                        cumulative_thickness_uvb,
                    ],
                    start_angle=-90,
                    stop_angle=-270,
                    rotation_angle=rotation_angle,
                    color=(0.5, 0.5, 0.5),
                    name=layer_name,
                    allow_overlapping_shape=True,
                )
                layer = outer_layer.union(inner_layer)
                layer.name = layer_name
                layers.append(layer)
            else:
                layers.extend(
                    _create_segmented_layer(
                        layer_name=layer_name,
                        segment_build=poloidal_entry,
                        minor_radius=minor_radius,
                        major_radius=major_radius,
                        triangularity=triangularity,
                        elongation=elongation,
                        rotation_angle=rotation_angle,
                        inner_offset=cumulative_thickness_orb,
                        upper_layer_thickness=upper_layer_thickness,
                        lower_layer_thickness=lower_layer_thickness,
                        inner_layer_thickness=inner_layer_thickness,
                        outer_layer_thickness=outer_layer_thickness,
                        cumulative_thickness_uvb=cumulative_thickness_uvb,
                        cumulative_thickness_lvb=cumulative_thickness_lvb,
                        cumulative_thickness_irb=cumulative_thickness_irb,
                        cumulative_thickness_orb=cumulative_thickness_orb,
                    )
                )
        cumulative_thickness_orb += outer_layer_thickness
        cumulative_thickness_irb += inner_layer_thickness
        cumulative_thickness_uvb += upper_layer_thickness
        cumulative_thickness_lvb += lower_layer_thickness
        # build inner layer

        # union layers

    return layers


def tokamak_from_plasma(
    radial_build: Sequence[Tuple[LayerType, float] | Tuple[LayerType, float, str]],
    poloidal_build=None,
    elongation: float = 2.0,
    triangularity: float = 0.55,
    rotation_angle: float = 180.0,
    extra_cut_shapes: Sequence[cq.Workplane] = None,
    extra_intersect_shapes: Sequence[cq.Workplane] = None,
    colors: dict = None,
) -> Assembly:
    """
    Creates a tokamak fusion reactor from a radial build and plasma parameters.

    Args:
        radial_build: sequence of tuples containing the radial build of the
            reactor. Each tuple should contain a LayerType, a float and a string.
        poloidal_build: optional sequence of poloidal segment definitions, one
            per radial pair after the plasma entry. Entries corresponding to
            GAP layers must be `None` in the corresponding slot of
            `poloidal_build`. Segment lengths are measured along the layer
            inner surface (at the cumulative outer radial offset); the inner
            arc is not automatically added.
        elongation: The elongation of the plasma. Defaults to 2.0.
        triangularity: The triangularity of the plasma. Defaults to 0.55.
        rotation_angle: The rotation angle of the plasma. Defaults to 180.0.
        extra_cut_shapes: A list of extra shapes to cut the reactor with. Defaults to [].
        extra_intersect_shapes: A list of extra shapes to intersect the reactor with. Defaults to [].
        colors (dict, optional): the colors to assign to the assembly parts. Defaults to {}.
            Each dictionary entry should be a key that matches the assembly part name
            (e.g. 'plasma', or 'layer_1') and a tuple of 3 or 4 floats between 0 and 1
            representing the RGB or RGBA values.

    Returns:
        CadQuery.Assembly: A CadQuery Assembly object representing the tokamak fusion reactor.
    """

    if extra_cut_shapes is None:
        extra_cut_shapes = []
    if extra_intersect_shapes is None:
        extra_intersect_shapes = []
    if colors is None:
        colors = {}

    inner_equatorial_point = sum_up_to_plasma(radial_build)
    plasma_radial_thickness = get_plasma_value(radial_build)
    outer_equatorial_point = inner_equatorial_point + plasma_radial_thickness

    # sets major radius and minor radius from equatorial_points to allow a
    # radial build. This helps avoid the plasma overlapping the center
    # column and other components
    major_radius = (outer_equatorial_point + inner_equatorial_point) / 2
    minor_radius = major_radius - inner_equatorial_point

    # make vertical build from inner radial build
    pi = get_plasma_index(radial_build)
    rbi = len(radial_build) - 1 - pi  # number of unique entries in outer or inner radial build
    # drop any layer names, they are only supported in radial_build not vertical_build
    upper_vertical_build = [(item[0], item[1]) for item in radial_build[pi - rbi : pi][::-1]]  # get the inner radial build

    plasma_height = 2 * minor_radius * elongation
    # slice operation reverses the list and removes the last value to avoid two plasmas
    vertical_build = upper_vertical_build[::-1] + [(LayerType.PLASMA, plasma_height)] + upper_vertical_build

    return tokamak(
        radial_build=radial_build,
        poloidal_build=poloidal_build,
        vertical_build=vertical_build,
        triangularity=triangularity,
        rotation_angle=rotation_angle,
        extra_cut_shapes=extra_cut_shapes,
        extra_intersect_shapes=extra_intersect_shapes,
        colors=colors
    )


def tokamak(
    radial_build: Sequence[Tuple[str, float] | Tuple[str, float, str]],
    vertical_build: Sequence[Tuple[str, float] | Tuple[str, float, str]],
    poloidal_build=None,
    triangularity: float = 0.55,
    rotation_angle: float = 180.0,
    extra_cut_shapes: Sequence[cq.Workplane] = None,
    extra_intersect_shapes: Sequence[cq.Workplane] = None,
    colors: dict = None,
) -> Assembly:
    """
    Creates a tokamak fusion reactor from a radial and vertical build.

    Args:
        radial_build: sequence of tuples containing the radial build of the
            reactor. Each tuple should contain a LayerType, a float and the string is optional.
        vertical_build: sequence of tuples containing the vertical build of the
            reactor. Each tuple should contain a LayerType, a float and the string is optional.
        poloidal_build: optional sequence of poloidal segment definitions, one per radial pair after
            the plasma entry. Entries corresponding to GAP layers must be `None` in the corresponding slot of `poloidal_build`. 
            Segment lengths are measured along the layer inner surface (at the cumulative outer radial offset);
            the inner arc is not automatically added.
        triangularity: The triangularity of the plasma. Defaults to 0.55.
        rotation_angle: The rotation angle of the plasma. Defaults to 180.0.
        extra_cut_shapes: A list of extra shapes to cut the reactor with. Defaults to [].
        extra_intersect_shapes: A list of extra shapes to intersect the reactor with. Defaults to [].
        colors (dict, optional): the colors to assign to the assembly parts. Defaults to {}.
            Each dictionary entry should be a key that matches the assembly part name
            (e.g. 'plasma', or 'layer_1') and a tuple of 3 or 4 floats between 0 and 1
            representing the RGB or RGBA values.

    Returns:
        CadQuery.Assembly: A CadQuery Assembly object representing the tokamak fusion reactor.
    """

    if extra_cut_shapes is None:
        extra_cut_shapes = []
    if extra_intersect_shapes is None:
        extra_intersect_shapes = []
    if colors is None:
        colors = {}

    validate_vertical_build_names(vertical_build, "tokamak()")

    inner_equatorial_point = sum_up_to_plasma(radial_build)
    plasma_radial_thickness = get_plasma_value(radial_build)
    plasma_vertical_thickness = get_plasma_value(vertical_build)
    outer_equatorial_point = inner_equatorial_point + plasma_radial_thickness

    major_radius = (outer_equatorial_point + inner_equatorial_point) / 2
    minor_radius = major_radius - inner_equatorial_point

    elongation = (plasma_vertical_thickness / 2) / minor_radius
    blanket_rear_wall_end_height = sum([item[1] for item in vertical_build])

    plasma = plasma_simplified(
        major_radius=major_radius,
        minor_radius=minor_radius,
        elongation=elongation,
        triangularity=triangularity,
        rotation_angle=rotation_angle,
    )

    inner_radial_build = create_center_column_shield_cylinders(
        radial_build, rotation_angle, blanket_rear_wall_end_height
    )

    blanket_cutting_cylinder = inner_radial_build[0] if inner_radial_build else None

    blanket_layers = create_layers_from_plasma(
        radial_build=radial_build,
        vertical_build=vertical_build,
        minor_radius=minor_radius,
        major_radius=major_radius,
        triangularity=triangularity,
        elongation=elongation,
        rotation_angle=rotation_angle,
        center_column=blanket_cutting_cylinder,
        poloidal_build=poloidal_build,
        layer_count=len(inner_radial_build)
    )

    cut_names, intersect_names, layer_names = get_assembly_names(
        extra_cut_shapes, extra_intersect_shapes, inner_radial_build, blanket_layers
    )

    validate_unique_assembly_names([*cut_names, *intersect_names, *layer_names, "plasma"], "tokamak()")

    my_assembly = Assembly()

    for entry, name in zip(extra_cut_shapes, cut_names):
        if not isinstance(entry, cq.Workplane):
            raise ValueError(f"extra_cut_shapes should only contain cadquery Workplanes, not {type(entry)}")
        my_assembly.add(entry, name=name, color=cq.Color(*colors.get(name, (0.5,0.5,0.5))))

    # builds up the intersect shapes
    if len(extra_intersect_shapes) > 0:
        # makes a union of the the radial build to use as a base for the intersect shapes
        if inner_radial_build:
            reactor_compound = inner_radial_build[0]
            compound_entries = inner_radial_build[1:] + blanket_layers
        elif blanket_layers:
            reactor_compound = blanket_layers[0]
            compound_entries = blanket_layers[1:]
        else:
            raise ValueError("tokamak() requires at least one solid layer to build a reactor compound.")

        for entry in compound_entries:
            reactor_compound = reactor_compound.union(entry)

        # adds the extra intersect shapes to the assembly
        for entry, name in zip(extra_intersect_shapes, intersect_names):
            reactor_entry_intersection = entry.intersect(reactor_compound)
            my_assembly.add(reactor_entry_intersection, name=name, color=cq.Color(*colors.get(name, (0.5,0.5,0.5))))

    # cut the core layers with any extra shapes (a no-op when there are none)
    cutters = extra_cut_shapes + extra_intersect_shapes
    for entry, name in zip(inner_radial_build + blanket_layers, layer_names):
        # TODO track the names of shapes, even when extra shapes are made due to splitting
        for cutter in cutters:
            entry = entry.cut(cutter)
        my_assembly.add(entry, name=name, color=cq.Color(*colors.get(name, (0.5,0.5,0.5))))

    my_assembly.add(plasma, name="plasma", color=cq.Color(*colors.get("plasma", (0.5,0.5,0.5))))

    my_assembly.elongation = elongation
    my_assembly.triangularity = triangularity
    my_assembly.major_radius = major_radius
    my_assembly.minor_radius = minor_radius

    return my_assembly
