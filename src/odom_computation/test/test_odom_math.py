import math

from odom_computation.odom_math import integrate, meters_per_tick, wrap_angle, yaw_to_quaternion

R, B, TPR = 0.065, 0.21, 512


def test_one_revolution_is_the_circumference():
    assert math.isclose(meters_per_tick(R, TPR) * TPR, 2 * math.pi * R)


def test_straight_line_has_no_turn():
    d = 100 * meters_per_tick(R, TPR)
    x, y, th, dc, dth = integrate(0.0, 0.0, 0.0, d, d, B)
    assert math.isclose(x, d) and y == 0.0 and th == 0.0 and dth == 0.0


def test_spin_in_place_turns_without_moving():
    d = 50 * meters_per_tick(R, TPR)
    x, y, th, dc, dth = integrate(0.0, 0.0, 0.0, -d, d, B)
    assert x == 0.0 and y == 0.0
    assert math.isclose(th, 2 * d / B)


def test_heading_is_wrapped():
    assert math.isclose(wrap_angle(3 * math.pi / 2), -math.pi / 2)


def test_quaternion_is_unit_and_matches_yaw():
    qx, qy, qz, qw = yaw_to_quaternion(math.pi / 2)
    assert math.isclose(qx * qx + qy * qy + qz * qz + qw * qw, 1.0)
    assert math.isclose(2 * math.atan2(qz, qw), math.pi / 2)


def test_forward_drive_at_0_2_ms_for_10_s():
    # what the provided tick publisher simulates: 0.2 m/s, 10 Hz -> x grows, y and theta stay ~0
    x = y = th = 0.0
    mpt = meters_per_tick(R, TPR)
    carry = 0.0
    ticks_per_step = 0.2 * 0.1 / mpt
    for _ in range(100):
        carry += ticks_per_step
        n = math.trunc(carry)
        carry -= n
        x, y, th, _, _ = integrate(x, y, th, n * mpt, n * mpt, B)
    assert abs(x - 2.0) < 0.01 and abs(y) < 1e-9 and abs(th) < 1e-9
