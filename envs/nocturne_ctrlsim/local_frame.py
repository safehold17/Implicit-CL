import math


def angle_of_rotation(yaw: float) -> float:
    return (math.pi / 2.0) - float(yaw)


def angle_sub(current_angle: float, target_angle: float) -> float:
    diff = (target_angle - current_angle) % (2 * math.pi)
    if diff > math.pi:
        diff = -(2 * math.pi - diff)
    return diff


def to_local(dx: float, dy: float, angle: float) -> tuple[float, float]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    local_x = dx * cos_a + dy * sin_a
    local_y = -dx * sin_a + dy * cos_a
    return local_x, local_y
