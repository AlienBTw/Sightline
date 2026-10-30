from sightline.geometry.calibrate import load_cameras_file, validate_camera_calibration


def test_demo_camera_calibration():
    data = load_cameras_file()
    cam = data["cameras"][0]
    summary = validate_camera_calibration(cam)
    assert summary["homography_ok"] is True
    assert summary["scale_ok"] is True
    assert abs(summary["plan_segment_m"] - 5.5) < 1e-6
