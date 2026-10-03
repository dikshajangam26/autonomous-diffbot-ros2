"""Pure differential-drive maths, with no ROS imports so it can be unit-tested anywhere."""
import math


def meters_per_tick(wheel_radius, ticks_per_revolution):
    """Distance one wheel rolls for one encoder tick (m)."""
    return 2.0 * math.pi * wheel_radius / ticks_per_revolution


def wrap_angle(a):
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def yaw_to_quaternion(yaw):
    """Heading about z -> quaternion (x, y, z, w)."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def integrate(x, y, theta, d_left, d_right, wheel_base):
    """Advance the pose by one step.

    d_left / d_right are the distances (m) each wheel rolled since the last step.
    Returns (x, y, theta, d_center, d_theta). The heading in the middle of the step
    is used for the translation, which is more accurate than the heading at its start.
    """
    d_center = (d_left + d_right) / 2.0
    d_theta = (d_right - d_left) / wheel_base
    x += d_center * math.cos(theta + d_theta / 2.0)
    y += d_center * math.sin(theta + d_theta / 2.0)
    theta = wrap_angle(theta + d_theta)
    return x, y, theta, d_center, d_theta
