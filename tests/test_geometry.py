import numpy as np

from sightline.geometry import compute_homography, pixel_to_plan, plan_distance


def test_homography_identity_ish():
    pixel = [[0, 0], [100, 0], [100, 100], [0, 100]]
    plan = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    H = compute_homography(pixel, plan)
    x, y = pixel_to_plan(H, 50, 50)
    assert abs(x - 5.0) < 0.2
    assert abs(y - 5.0) < 0.2
    assert plan_distance((0, 0), (3, 4)) == 5.0
