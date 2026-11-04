import math

import numpy as np

from sightline_pipeline.localize import (Circle, box_to_circle, circle_box_iou, circle_iou, circles_from_map,
                                         contains, yolo_to_xyxy)


def test_peak_maps_to_correct_frame_location():
    m = np.zeros((12, 21), np.float32)
    m[3, 15] = 1.0                       # row 3 of 12, col 15 of 21
    c = circles_from_map(m, 1920, 1080)[0]
    assert abs(c.cx / 1920 - 15.5 / 21) < 0.03 and abs(c.cy / 1080 - 3.5 / 12) < 0.03
    n = c.norm()
    assert 0 <= n["cx"] <= 1 and abs(n["rx"] * 1920 - n["ry"] * 1080) < 1


def test_radius_clamped_and_grows_with_region():
    small = np.zeros((12, 21)); small[6, 10] = 1
    big = np.zeros((12, 21)); big[3:9, 6:15] = 1
    cs = circles_from_map(small, 1000, 1000, min_radius_frac=0.04)[0]
    cb = circles_from_map(big, 1000, 1000)[0]
    assert cs.r >= 0.04 * math.hypot(1000, 1000) - 1e-6
    assert cb.r > cs.r
    flat = circles_from_map(np.ones((12, 21)), 100, 100)
    assert flat == []


def test_max_circles_two_peaks():
    m = np.zeros((12, 21)); m[2, 2] = 1.0; m[9, 18] = 0.95
    assert len(circles_from_map(m, 2100, 1200, max_circles=2)) == 2
    assert len(circles_from_map(m, 2100, 1200, max_circles=1)) == 1


def test_geometry():
    c = Circle(50, 50, 10, 100, 100)
    assert contains(c, 55, 55) and not contains(c, 61, 50)
    assert circle_box_iou(c, (40, 40, 60, 60), 400) > 0.6
    assert circle_box_iou(c, (80, 80, 90, 90)) == 0.0
    e = box_to_circle(40, 40, 60, 60, 100, 100)
    assert e.cx == 50 and abs(e.r - math.hypot(20, 20) / 2) < 1e-9
    assert abs(circle_iou(c, c) - 1.0) < 1e-9 and circle_iou(c, Circle(90, 90, 5, 100, 100)) == 0.0
    assert yolo_to_xyxy((0.5, 0.5, 0.2, 0.4), 100, 50) == (40.0, 15.0, 60.0, 35.0)
    s = c.scaled(200, 200)
    assert s.cx == 100 and s.r == 20
