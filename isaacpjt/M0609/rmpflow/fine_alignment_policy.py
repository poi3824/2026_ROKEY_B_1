"""ROS/Isaac-free helpers for closed-loop fine-alignment motion steps."""

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple


BUSBAR_HOLE_LOCAL_CENTERS_M = (
    (-0.24163777, -0.13888114, 0.003),
    (0.24163777, 0.13888114, 0.003),
)
BUSBAR_HOLE_LOCAL_BOTTOM_CENTERS_M = tuple(
    (x_m, y_m, -z_m)
    for x_m, y_m, z_m in BUSBAR_HOLE_LOCAL_CENTERS_M
)
MAX_BOLT_AXIS_TILT_RAD = math.radians(3.0)

Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]


@dataclass(frozen=True)
class BusbarHoleAlignment:
    """Best planar rigid alignment from busbar holes to two bolt axes."""

    pairing: str
    swapped: bool
    hole_world_points_m: Tuple[Point3, Point3]
    paired_bolt_axes_xy_m: Tuple[Point2, Point2]
    hole_midpoint_xy_m: Point2
    bolt_midpoint_xy_m: Point2
    desired_rigid_xy_correction_m: Point2
    midpoint_residual_m: float
    yaw_residual_rad: float
    endpoint_residuals_m: Tuple[float, float]
    max_endpoint_residual_m: float


@dataclass(frozen=True)
class WorldAxis:
    """One finite local-Z line represented in world coordinates."""

    origin_m: Point3
    direction: Point3
    tilt_rad: float


@dataclass(frozen=True)
class BusbarBoreAlignment:
    """Top/bottom hole observations against two paired live bolt axes."""

    alignment: BusbarHoleAlignment
    bottom_hole_world_points_m: Tuple[Point3, Point3]
    paired_axis_indices: Tuple[int, int]
    bolt_axes: Tuple[WorldAxis, WorldAxis]
    top_axis_points_m: Tuple[Point3, Point3]
    bottom_axis_points_m: Tuple[Point3, Point3]
    top_endpoint_residuals_m: Tuple[float, float]
    bottom_endpoint_residuals_m: Tuple[float, float]
    bore_endpoint_residuals_m: Tuple[float, float]
    max_bore_endpoint_residual_m: float


@dataclass(frozen=True)
class FineStepStatus:
    """Result of one executor-side settle observation."""

    stamp_ns: Optional[int]
    settle_count: int
    settled: bool


@dataclass(frozen=True)
class PlanarCommandLeadStatus:
    """Bounded XY controller lead retained around one fine-alignment step."""

    x_m: float
    y_m: float
    magnitude_m: float
    grew: bool
    saturated: bool


def _finite_matrix4_rows(transform) -> Tuple[Tuple[float, ...], ...]:
    """Copy a duck-typed Gf/sequence matrix into four finite rows."""
    try:
        rows = tuple(
            tuple(float(transform[row][column]) for column in range(4))
            for row in range(4)
        )
    except (
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:
        raise ValueError(
            "mesh_world_transform must be a finite 4x4 matrix"
        ) from exc
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError("mesh_world_transform must be a finite 4x4 matrix")
    return rows


def _transform_usd_row_point(
    matrix: Tuple[Tuple[float, ...], ...],
    point: Point3,
) -> Point3:
    """Apply a USD/Gf row-vector 4x4 transform to one local point."""
    homogeneous = (point[0], point[1], point[2], 1.0)
    transformed = tuple(
        sum(homogeneous[row] * matrix[row][column] for row in range(4))
        for column in range(4)
    )
    divisor = transformed[3]
    if not math.isfinite(divisor) or abs(divisor) <= 1.0e-12:
        raise ValueError("mesh_world_transform produced an invalid point")
    world = tuple(value / divisor for value in transformed[:3])
    if not all(math.isfinite(value) for value in world):
        raise ValueError("mesh_world_transform produced an invalid point")
    return world


def extract_local_z_world_axis(
    transform,
    *,
    maximum_tilt_rad: float = MAX_BOLT_AXIS_TILT_RAD,
) -> WorldAxis:
    """Extract a normalized, upward local-Z line from a full USD affine matrix."""

    maximum_tilt_rad = float(maximum_tilt_rad)
    if (
        not math.isfinite(maximum_tilt_rad)
        or maximum_tilt_rad < 0.0
        or maximum_tilt_rad >= 0.5 * math.pi
    ):
        raise ValueError("maximum_tilt_rad must be finite and below 90 degrees")

    matrix = _finite_matrix4_rows(transform)
    origin = _transform_usd_row_point(matrix, (0.0, 0.0, 0.0))
    local_z_point = _transform_usd_row_point(matrix, (0.0, 0.0, 1.0))
    raw_direction = tuple(
        local_z_point[index] - origin[index]
        for index in range(3)
    )
    length = math.sqrt(sum(value * value for value in raw_direction))
    if not math.isfinite(length) or length <= 1.0e-12:
        raise ValueError("local-Z world axis is degenerate")

    direction = tuple(value / length for value in raw_direction)
    if direction[2] < 0.0:
        direction = tuple(-value for value in direction)
    if direction[2] <= 1.0e-9:
        raise ValueError("local-Z world axis is near-horizontal")

    tilt_rad = math.acos(max(-1.0, min(1.0, direction[2])))
    if tilt_rad > maximum_tilt_rad + 1.0e-12:
        raise ValueError(
            "local-Z world axis tilt exceeds limit: "
            f"{math.degrees(tilt_rad):.6f}deg > "
            f"{math.degrees(maximum_tilt_rad):.6f}deg"
        )
    return WorldAxis(
        origin_m=origin,
        direction=direction,
        tilt_rad=tilt_rad,
    )


def project_world_axis_to_z(axis: WorldAxis, world_z_m: float) -> Point3:
    """Intersect a finite non-horizontal world axis with one world-Z plane."""

    if not isinstance(axis, WorldAxis):
        raise TypeError("axis must be a WorldAxis")
    world_z_m = float(world_z_m)
    values = (*axis.origin_m, *axis.direction, axis.tilt_rad, world_z_m)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("world axis projection inputs must be finite")
    direction_length = math.sqrt(
        sum(value * value for value in axis.direction)
    )
    if (
        abs(direction_length - 1.0) > 1.0e-9
        or abs(axis.direction[2]) <= 1.0e-9
    ):
        raise ValueError("world axis direction must be normalized and vertical")

    distance = (
        world_z_m - axis.origin_m[2]
    ) / axis.direction[2]
    projected = (
        axis.origin_m[0] + distance * axis.direction[0],
        axis.origin_m[1] + distance * axis.direction[1],
        world_z_m,
    )
    if not all(math.isfinite(value) for value in projected):
        raise ValueError("world axis projection produced a non-finite point")
    return projected


def transformed_usd_points_bounds(
    transform,
    local_points: Sequence[Sequence[float]],
) -> Tuple[Point3, Point3]:
    """Transform actual authored points and return their exact world AABB."""

    matrix = _finite_matrix4_rows(transform)
    try:
        points = tuple(
            tuple(float(value) for value in point)
            for point in local_points
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("local_points must contain finite 3-vectors") from exc
    if not points or any(len(point) != 3 for point in points):
        raise ValueError("local_points must contain finite 3-vectors")
    if not all(math.isfinite(value) for point in points for value in point):
        raise ValueError("local_points must contain finite 3-vectors")

    world_points = tuple(
        _transform_usd_row_point(matrix, point)
        for point in points
    )
    return (
        tuple(min(point[index] for point in world_points) for index in range(3)),
        tuple(max(point[index] for point in world_points) for index in range(3)),
    )


def busbar_bore_unavailable_gate(
    top_endpoint_residuals_m: Sequence[float],
    bottom_endpoint_residuals_m: Sequence[float],
    previous_count: int,
    *,
    alignment_tolerance_m: float,
    required_steps: int,
    observation_stable: bool,
) -> Tuple[int, bool]:
    """
    Bound a planar alignment that cannot close both ends of a tilted bore.

    The executor only commands XY and world-Z yaw during hole alignment.  Once
    the top endpoints and the robot pose are stable inside tolerance, a bottom
    endpoint that remains outside the same tolerance cannot be improved by
    repeatedly issuing the already-satisfied top-plane command.  Requiring
    consecutive stable observations avoids treating contact jitter as an
    authored/live geometry failure.
    """

    try:
        top = tuple(float(value) for value in top_endpoint_residuals_m)
        bottom = tuple(
            float(value) for value in bottom_endpoint_residuals_m
        )
        previous_count = int(previous_count)
        alignment_tolerance_m = float(alignment_tolerance_m)
        required_steps = int(required_steps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("bore liveness inputs must be finite") from exc
    if (
        len(top) != 2
        or len(bottom) != 2
        or not all(math.isfinite(value) for value in (*top, *bottom))
        or not math.isfinite(alignment_tolerance_m)
        or alignment_tolerance_m < 0.0
        or previous_count < 0
        or required_steps <= 0
    ):
        raise ValueError("bore liveness inputs must be finite and bounded")

    unavailable_sample = (
        bool(observation_stable)
        and max(top) <= alignment_tolerance_m
        and max(bottom) > alignment_tolerance_m
    )
    next_count = previous_count + 1 if unavailable_sample else 0
    return next_count, next_count >= required_steps


def _parse_bolt_axes_xy(
    bolt_axes_xy: Sequence[Sequence[float]],
) -> Tuple[Point2, Point2]:
    try:
        axes = tuple(tuple(float(value) for value in point)
                     for point in bolt_axes_xy)
    except (TypeError, ValueError) as exc:
        raise ValueError("bolt_axes_xy must contain two finite points") from exc
    if len(axes) != 2 or any(len(point) < 2 for point in axes):
        raise ValueError("bolt_axes_xy must contain exactly two XY points")
    if not all(math.isfinite(value) for point in axes for value in point):
        raise ValueError("bolt_axes_xy must contain two finite points")
    return (
        (axes[0][0], axes[0][1]),
        (axes[1][0], axes[1][1]),
    )


def _squared_planar_distance(left: Point2, right: Point2) -> float:
    return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2


def _wrapped_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _build_busbar_hole_alignment(
    holes: Tuple[Point3, Point3],
    paired_bolts: Tuple[Point2, Point2],
    *,
    swapped: bool,
) -> BusbarHoleAlignment:
    """Build one alignment after an association has already been selected."""

    hole_xy = tuple((point[0], point[1]) for point in holes)
    hole_midpoint = (
        0.5 * (hole_xy[0][0] + hole_xy[1][0]),
        0.5 * (hole_xy[0][1] + hole_xy[1][1]),
    )
    bolt_midpoint = (
        0.5 * (paired_bolts[0][0] + paired_bolts[1][0]),
        0.5 * (paired_bolts[0][1] + paired_bolts[1][1]),
    )
    correction = (
        bolt_midpoint[0] - hole_midpoint[0],
        bolt_midpoint[1] - hole_midpoint[1],
    )

    hole_vector = (
        hole_xy[1][0] - hole_xy[0][0],
        hole_xy[1][1] - hole_xy[0][1],
    )
    bolt_vector = (
        paired_bolts[1][0] - paired_bolts[0][0],
        paired_bolts[1][1] - paired_bolts[0][1],
    )
    if (
        math.hypot(*hole_vector) <= 1.0e-12
        or math.hypot(*bolt_vector) <= 1.0e-12
    ):
        raise ValueError("hole and bolt pairs need non-zero planar spans")

    yaw_residual = _wrapped_angle(
        math.atan2(bolt_vector[1], bolt_vector[0])
        - math.atan2(hole_vector[1], hole_vector[0])
    )
    cos_yaw = math.cos(yaw_residual)
    sin_yaw = math.sin(yaw_residual)
    endpoint_residuals = []
    for hole, bolt in zip(hole_xy, paired_bolts):
        centered_x = hole[0] - hole_midpoint[0]
        centered_y = hole[1] - hole_midpoint[1]
        corrected = (
            bolt_midpoint[0]
            + cos_yaw * centered_x
            - sin_yaw * centered_y,
            bolt_midpoint[1]
            + sin_yaw * centered_x
            + cos_yaw * centered_y,
        )
        endpoint_residuals.append(math.sqrt(
            _squared_planar_distance(corrected, bolt)
        ))

    midpoint_residual = math.hypot(*correction)
    residuals = (endpoint_residuals[0], endpoint_residuals[1])
    return BusbarHoleAlignment(
        pairing="swapped" if swapped else "direct",
        swapped=swapped,
        hole_world_points_m=holes,
        paired_bolt_axes_xy_m=paired_bolts,
        hole_midpoint_xy_m=hole_midpoint,
        bolt_midpoint_xy_m=bolt_midpoint,
        desired_rigid_xy_correction_m=correction,
        midpoint_residual_m=midpoint_residual,
        yaw_residual_rad=yaw_residual,
        endpoint_residuals_m=residuals,
        max_endpoint_residual_m=max(residuals),
    )


def solve_busbar_hole_alignment(
    mesh_world_transform,
    bolt_axes_xy: Sequence[Sequence[float]],
) -> BusbarHoleAlignment:
    """
    Fit the two transformed busbar holes to two unordered bolt axes.

    ``mesh_world_transform`` follows the USD/Gf row-vector convention, where
    translation occupies row 3.  Applying the complete 4x4 matrix preserves
    authored scale and rotation instead of treating the Mesh pose as a rigid
    translation/quaternion pair.

    Direct and swapped endpoint associations are compared by their planar
    least-squares cost.  The returned XY correction moves the hole midpoint to
    the bolt midpoint.  ``yaw_residual_rad`` then rotates around that midpoint,
    and ``max_endpoint_residual_m`` is the remaining endpoint mismatch after
    applying both rigid corrections (for example, a bolt/hole span mismatch).
    """
    matrix = _finite_matrix4_rows(mesh_world_transform)
    holes = tuple(
        _transform_usd_row_point(matrix, local_point)
        for local_point in BUSBAR_HOLE_LOCAL_CENTERS_M
    )
    hole_xy = tuple((point[0], point[1]) for point in holes)
    bolts = _parse_bolt_axes_xy(bolt_axes_xy)

    direct_cost = sum(
        _squared_planar_distance(hole_xy[index], bolts[index])
        for index in range(2)
    )
    swapped_cost = sum(
        _squared_planar_distance(hole_xy[index], bolts[1 - index])
        for index in range(2)
    )
    swapped = swapped_cost < direct_cost
    paired_bolts = (
        (bolts[1], bolts[0])
        if swapped
        else bolts
    )
    return _build_busbar_hole_alignment(
        (holes[0], holes[1]),
        paired_bolts,
        swapped=swapped,
    )


def solve_busbar_bore_alignment(
    mesh_world_transform,
    bolt_world_transforms: Sequence[Sequence[Sequence[float]]],
    *,
    maximum_axis_tilt_rad: float = MAX_BOLT_AXIS_TILT_RAD,
) -> BusbarBoreAlignment:
    """Fit top holes to live bolt lines and verify both bore endpoints."""

    try:
        bolt_transforms = tuple(bolt_world_transforms)
    except TypeError as exc:
        raise ValueError(
            "bolt_world_transforms must contain exactly two transforms"
        ) from exc
    if len(bolt_transforms) != 2:
        raise ValueError(
            "bolt_world_transforms must contain exactly two transforms"
        )

    matrix = _finite_matrix4_rows(mesh_world_transform)
    top_holes = tuple(
        _transform_usd_row_point(matrix, local_point)
        for local_point in BUSBAR_HOLE_LOCAL_CENTERS_M
    )
    bottom_holes = tuple(
        _transform_usd_row_point(matrix, local_point)
        for local_point in BUSBAR_HOLE_LOCAL_BOTTOM_CENTERS_M
    )
    bolt_axes = tuple(
        extract_local_z_world_axis(
            transform,
            maximum_tilt_rad=maximum_axis_tilt_rad,
        )
        for transform in bolt_transforms
    )
    top_projection = tuple(
        tuple(
            project_world_axis_to_z(axis, hole[2])
            for axis in bolt_axes
        )
        for hole in top_holes
    )

    direct_cost = sum(
        _squared_planar_distance(
            top_holes[index][:2],
            top_projection[index][index][:2],
        )
        for index in range(2)
    )
    swapped_cost = sum(
        _squared_planar_distance(
            top_holes[index][:2],
            top_projection[index][1 - index][:2],
        )
        for index in range(2)
    )
    swapped = swapped_cost < direct_cost
    paired_axis_indices = (1, 0) if swapped else (0, 1)
    paired_top_axes = tuple(
        top_projection[index][paired_axis_indices[index]]
        for index in range(2)
    )
    alignment = _build_busbar_hole_alignment(
        (top_holes[0], top_holes[1]),
        tuple(point[:2] for point in paired_top_axes),
        swapped=swapped,
    )
    paired_bottom_axes = tuple(
        project_world_axis_to_z(
            bolt_axes[paired_axis_indices[index]],
            bottom_holes[index][2],
        )
        for index in range(2)
    )
    top_residuals = tuple(
        math.hypot(
            top_holes[index][0] - paired_top_axes[index][0],
            top_holes[index][1] - paired_top_axes[index][1],
        )
        for index in range(2)
    )
    bottom_residuals = tuple(
        math.hypot(
            bottom_holes[index][0] - paired_bottom_axes[index][0],
            bottom_holes[index][1] - paired_bottom_axes[index][1],
        )
        for index in range(2)
    )
    bore_residuals = tuple(
        max(top_residuals[index], bottom_residuals[index])
        for index in range(2)
    )
    return BusbarBoreAlignment(
        alignment=alignment,
        bottom_hole_world_points_m=(
            bottom_holes[0],
            bottom_holes[1],
        ),
        paired_axis_indices=paired_axis_indices,
        bolt_axes=(bolt_axes[0], bolt_axes[1]),
        top_axis_points_m=(paired_top_axes[0], paired_top_axes[1]),
        bottom_axis_points_m=(
            paired_bottom_axes[0],
            paired_bottom_axes[1],
        ),
        top_endpoint_residuals_m=(
            top_residuals[0],
            top_residuals[1],
        ),
        bottom_endpoint_residuals_m=(
            bottom_residuals[0],
            bottom_residuals[1],
        ),
        bore_endpoint_residuals_m=(
            bore_residuals[0],
            bore_residuals[1],
        ),
        max_bore_endpoint_residual_m=max(bore_residuals),
    )


class BoundedPlanarCommandLead:
    """
    Add a bounded controller-only XY lead without changing the logical target.

    RMPFlow can settle just short of a submillimetre target.  While the
    measured command-direction response remains below the gate's real response
    requirement, this helper grows a small lead along the correction direction.
    It holds while the response is sufficient and resumes bounded growth if the
    response falls below the requirement before the real settle ACK.
    """

    def __init__(
        self,
        *,
        growth_step_m: float = 0.00005,
        max_lead_m: float = 0.001,
    ):
        values = (float(growth_step_m), float(max_lead_m))
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError(
                "command lead growth and maximum must be finite and positive"
            )
        if values[0] > values[1]:
            raise ValueError("command lead growth cannot exceed its maximum")
        self.growth_step_m = values[0]
        self.max_lead_m = values[1]
        self.reset()

    def reset(self):
        """Clear both retained lead and the active correction direction."""
        self._lead_x_m = 0.0
        self._lead_y_m = 0.0
        self._active_direction = None

    def begin(self, command_x_m: float, command_y_m: float):
        """Start one step, retaining only same-direction prior lead."""
        command_x_m = float(command_x_m)
        command_y_m = float(command_y_m)
        if not all(
            math.isfinite(value)
            for value in (command_x_m, command_y_m)
        ):
            raise ValueError("command lead direction must be finite")

        command_norm = math.hypot(command_x_m, command_y_m)
        if command_norm <= 1.0e-12:
            # A yaw-only step must not invent planar motion.  Keep the
            # previously settled controller offset so the camera observation
            # remains stationary, but do not grow it.
            self._active_direction = None
            return self.status()

        direction = (
            command_x_m / command_norm,
            command_y_m / command_norm,
        )
        retained_m = max(
            0.0,
            self._lead_x_m * direction[0]
            + self._lead_y_m * direction[1],
        )
        retained_m = min(retained_m, self.max_lead_m)
        self._lead_x_m = direction[0] * retained_m
        self._lead_y_m = direction[1] * retained_m
        self._active_direction = direction
        return self.status()

    def advance(
        self,
        *,
        observed_response_m: float,
        required_response_m: float,
    ):
        """Grow once when real response is short, otherwise freeze the lead."""
        observed_response_m = float(observed_response_m)
        required_response_m = float(required_response_m)
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (observed_response_m, required_response_m)
        ):
            raise ValueError(
                "command lead responses must be finite and non-negative"
            )

        previous_m = math.hypot(self._lead_x_m, self._lead_y_m)
        next_m = previous_m
        if (
            self._active_direction is not None
            and required_response_m > 0.0
            and observed_response_m < required_response_m
        ):
            next_m = min(
                self.max_lead_m,
                previous_m + self.growth_step_m,
            )
            self._lead_x_m = self._active_direction[0] * next_m
            self._lead_y_m = self._active_direction[1] * next_m

        return PlanarCommandLeadStatus(
            x_m=self._lead_x_m,
            y_m=self._lead_y_m,
            magnitude_m=next_m,
            grew=next_m > previous_m,
            saturated=next_m >= self.max_lead_m,
        )

    def finish_step(self):
        """Stop growth while retaining the settled controller offset."""
        self._active_direction = None
        return self.status()

    def status(self):
        """Return the current lead without changing it."""
        magnitude_m = math.hypot(self._lead_x_m, self._lead_y_m)
        return PlanarCommandLeadStatus(
            x_m=self._lead_x_m,
            y_m=self._lead_y_m,
            magnitude_m=magnitude_m,
            grew=False,
            saturated=magnitude_m >= self.max_lead_m,
        )

    def command_xy(self, logical_x_m: float, logical_y_m: float):
        """Return a controller XY target while leaving its logical input intact."""
        logical_x_m = float(logical_x_m)
        logical_y_m = float(logical_y_m)
        if not all(
            math.isfinite(value)
            for value in (logical_x_m, logical_y_m)
        ):
            raise ValueError("logical controller target must be finite")
        return (
            logical_x_m + self._lead_x_m,
            logical_y_m + self._lead_y_m,
        )

    def commit_xy(self, logical_x_m: float, logical_y_m: float):
        """Fold the current lead into the canonical XY without a target jump."""

        committed_xy = self.command_xy(logical_x_m, logical_y_m)
        self.reset()
        return committed_xy


class FineAlignmentStepGate:
    """Acknowledge one incremental command only after the arm has settled."""

    def __init__(
        self,
        *,
        position_tolerance_m: float = 0.005,
        orientation_tolerance_rad: float = math.radians(3.0),
        motion_tolerance_m: float = 0.0001,
        orientation_motion_tolerance_rad: float = math.radians(0.005),
        unobservable_motion_tolerance_m: Optional[float] = None,
        unobservable_orientation_motion_tolerance_rad: Optional[float] = None,
        position_response_noise_margin_m: float = 0.0,
        orientation_response_noise_margin_rad: float = 0.0,
        settle_steps: int = 6,
    ):
        # A sub-quantized camera correction has no observable response proof
        # after its noise margin is removed.  It still needs the full
        # position/orientation and consecutive-motion gate, but Isaac's
        # quaternion/controller quantization is noisier than the strict
        # incremental-move stationary threshold.  Keep that broader profile
        # opt-in and use it only for the proof-waived case in ``begin``.
        if unobservable_motion_tolerance_m is None:
            unobservable_motion_tolerance_m = motion_tolerance_m
        if unobservable_orientation_motion_tolerance_rad is None:
            unobservable_orientation_motion_tolerance_rad = (
                orientation_motion_tolerance_rad
            )
        values = (
            position_tolerance_m,
            orientation_tolerance_rad,
            motion_tolerance_m,
            orientation_motion_tolerance_rad,
            unobservable_motion_tolerance_m,
            unobservable_orientation_motion_tolerance_rad,
            position_response_noise_margin_m,
            orientation_response_noise_margin_rad,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("settle tolerances must be finite and non-negative")
        if not isinstance(settle_steps, int) or settle_steps <= 0:
            raise ValueError("settle_steps must be a positive integer")

        self.position_tolerance_m = float(position_tolerance_m)
        self.orientation_tolerance_rad = float(
            orientation_tolerance_rad
        )
        self.motion_tolerance_m = float(motion_tolerance_m)
        self.orientation_motion_tolerance_rad = float(
            orientation_motion_tolerance_rad
        )
        self.unobservable_motion_tolerance_m = float(
            unobservable_motion_tolerance_m
        )
        self.unobservable_orientation_motion_tolerance_rad = float(
            unobservable_orientation_motion_tolerance_rad
        )
        self.position_response_noise_margin_m = float(
            position_response_noise_margin_m
        )
        self.orientation_response_noise_margin_rad = float(
            orientation_response_noise_margin_rad
        )
        self.settle_steps = settle_steps
        self.reset()

    def reset(self):
        self.active_stamp_ns: Optional[int] = None
        self.settle_count = 0
        self.required_position_response_m = 0.0
        self.required_orientation_response_rad = 0.0
        self.using_unobservable_motion_tolerance = False
        self.active_motion_tolerance_m = self.motion_tolerance_m
        self.active_orientation_motion_tolerance_rad = (
            self.orientation_motion_tolerance_rad
        )

    def begin(
        self,
        stamp_ns: int,
        *,
        required_position_response_m: float = 0.0,
        required_orientation_response_rad: float = 0.0,
    ):
        stamp_ns = int(stamp_ns)
        if stamp_ns <= 0:
            raise ValueError("fine-alignment command stamp must be positive")
        if self.active_stamp_ns is not None:
            raise RuntimeError("a fine-alignment step is already active")
        responses = (
            float(required_position_response_m),
            float(required_orientation_response_rad),
        )
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in responses
        ):
            raise ValueError(
                "required step responses must be finite and non-negative"
            )

        self.active_stamp_ns = stamp_ns
        self.settle_count = 0
        position_response_requirement_m = max(
            0.0,
            responses[0] - self.position_response_noise_margin_m,
        )
        # A correction can be non-zero while its net, noise-adjusted response
        # is smaller than the per-tick pose-settle quantum.  Requiring a
        # measured displacement in that interval deadlocks RMPFlow after it
        # has settled (the live 0.2 mm correction reduces to a 0.06 mm proof
        # requirement with the configured 0.1 mm margin).  Such a step still
        # needs every position/orientation and configured stationary check
        # below;
        # only the unobservable response proof is waived.  Any response above
        # that quantum remains strictly non-zero validated.
        if (
            0.0 < position_response_requirement_m
            <= self.motion_tolerance_m
        ):
            position_response_requirement_m = 0.0
        self.required_position_response_m = (
            position_response_requirement_m
        )
        self.required_orientation_response_rad = max(
            0.0,
            responses[1] - self.orientation_response_noise_margin_rad,
        )
        self.using_unobservable_motion_tolerance = (
            position_response_requirement_m <= 0.0
            and self.required_orientation_response_rad <= 0.0
        )
        if self.using_unobservable_motion_tolerance:
            self.active_motion_tolerance_m = (
                self.unobservable_motion_tolerance_m
            )
            self.active_orientation_motion_tolerance_rad = (
                self.unobservable_orientation_motion_tolerance_rad
            )
        else:
            self.active_motion_tolerance_m = self.motion_tolerance_m
            self.active_orientation_motion_tolerance_rad = (
                self.orientation_motion_tolerance_rad
            )

    def update(
        self,
        *,
        position_error_m: float,
        orientation_error_rad: float,
        motion_m: float,
        orientation_motion_rad: float,
        position_response_m: float,
        orientation_response_rad: float,
    ) -> FineStepStatus:
        values = (
            float(position_error_m),
            float(orientation_error_rad),
            float(motion_m),
            float(orientation_motion_rad),
            float(position_response_m),
            float(orientation_response_rad),
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("settle observations must be finite and non-negative")
        if self.active_stamp_ns is None:
            return FineStepStatus(None, 0, False)

        within_tolerance = (
            values[0] <= self.position_tolerance_m
            and values[1] <= self.orientation_tolerance_rad
            and values[2] <= self.active_motion_tolerance_m
            and values[3] <= self.active_orientation_motion_tolerance_rad
            and values[4] >= self.required_position_response_m
            and values[5] >= self.required_orientation_response_rad
        )
        if within_tolerance:
            self.settle_count += 1
        else:
            self.settle_count = 0

        stamp_ns = self.active_stamp_ns
        settled = self.settle_count >= self.settle_steps
        status = FineStepStatus(stamp_ns, self.settle_count, settled)
        if settled:
            self.reset()
        return status


def planar_command_response(
    *,
    start_x: float,
    start_y: float,
    current_x: float,
    current_y: float,
    command_x: float,
    command_y: float,
) -> float:
    """Return net EE displacement projected onto the XY command direction."""
    values = tuple(
        float(value)
        for value in (
            start_x,
            start_y,
            current_x,
            current_y,
            command_x,
            command_y,
        )
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("planar response inputs must be finite")

    start_x, start_y, current_x, current_y, command_x, command_y = values
    command_norm = math.hypot(command_x, command_y)
    if command_norm <= 1.0e-12:
        return 0.0
    response = (
        (current_x - start_x) * command_x
        + (current_y - start_y) * command_y
    ) / command_norm
    return max(0.0, response)


def planar_yaw_delta(
    *,
    quaternion_x: float,
    quaternion_y: float,
    quaternion_z: float,
    quaternion_w: float,
    max_abs_rad: float = math.radians(1.0),
) -> float:
    """Validate a planar delta quaternion and return its wrapped yaw."""
    values = tuple(
        float(value)
        for value in (
            quaternion_x,
            quaternion_y,
            quaternion_z,
            quaternion_w,
        )
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("fine-alignment quaternion contains NaN or inf")
    if not math.isfinite(max_abs_rad) or max_abs_rad <= 0.0:
        raise ValueError("max_abs_rad must be finite and positive")

    x, y, z, w = values
    if abs(x) > 1.0e-9 or abs(y) > 1.0e-9:
        raise ValueError("fine-alignment delta must be a planar yaw")
    norm = math.hypot(z, w)
    if norm <= 1.0e-12:
        raise ValueError("fine-alignment quaternion has zero norm")
    z /= norm
    w /= norm
    yaw = 2.0 * math.atan2(z, w)
    yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
    if abs(yaw) > max_abs_rad:
        raise ValueError("fine-alignment yaw delta exceeds the safety limit")
    return yaw
