import numpy as np


def orientation_sign_from_quad(quad):
    """Return the handedness of TL/TR/BR/BL image corners of a pitch.

    The four-edge calibration already validates that these corners form a
    non-degenerate convex quadrilateral. Both signs are legitimate camera
    views; the sign is a property of the calibrated mapping, not a fixed
    requirement that the camera be on one side of the field.
    """
    corners = np.asarray(quad, dtype=np.float64)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("Invalid four-edge pitch corners")
    top = corners[1] - corners[0]
    left = corners[3] - corners[0]
    signed_area = top[0] * left[1] - top[1] * left[0]
    if not np.isfinite(signed_area) or abs(signed_area) <= 1e-12:
        raise ValueError("Degenerate four-edge pitch corners")
    return 1 if signed_area > 0 else -1


def is_valid(points, pitch_to_pixel, pixel_to_pitch, w, h, orientation_sign=1):
    """
    Checks the validity of a set of points based on coordinate transformations and geometric constraints.
    Parameters
    ----------
    points : array-like, shape (N, 2)
        List or array of points in pitch (field) coordinates.
    pitch_to_pixel : callable
        Function that converts pitch coordinates to pixel coordinates.
    pixel_to_pitch : callable
        Function that converts pixel coordinates back to pitch coordinates.
    w : int or float
        Width of the image (in pixels).
    h : int or float
        Height of the image (in pixels).
    orientation_sign : int, optional
        Expected local pitch-to-image orientation (+1 or -1). The legacy
        calibration keeps its historical +1 default; four-edge calibration
        passes the sign established by its validated quadrilateral.
    Returns
    -------
    mask : np.ndarray, shape (N,)
        Boolean array indicating which points are valid according to the following criteria:
        - The pixel coordinates are within the image boundaries.
        - The round-trip transformation error (pitch -> pixel -> pitch) is less than 5 units.
        - The local transformation has the expected, non-zero orientation.
    """
    import os
    import yaml
    root_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(root_path, "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    if orientation_sign not in (-1, 1):
        raise ValueError("orientation_sign must be +1 or -1")

    # 将点从球场坐标转换为像素坐标
    points = np.array(points, dtype=np.float32).reshape(-1, 2)  # 确保输入是正确的形状
    N = points.shape[0]
    points_x = points.copy()
    points_x[:, 0] += 0.1

    points_y = points.copy()
    points_y[:, 1] += 0.1

    points_tot = np.vstack([points, points_x, points_y])
    pixel_points = pitch_to_pixel(points_tot)
    
    # 将像素坐标转换回球场坐标
    pitch_points = pixel_to_pitch(pixel_points[:N, :])
    
    mask = np.zeros(points.shape[0], dtype=bool)
    for i, (pixel_pt, pixel_pt_x, pixel_pt_y,pitch_pt, pt) in enumerate(zip(pixel_points[:N, :],
                                                     pixel_points[N:2*N, :],
                                                     pixel_points[2 * N:, :],
                                                     pitch_points,
                                                     points)):
        local_area = ((pixel_pt_x[0] - pixel_pt[0]) * (pixel_pt_y[1] - pixel_pt[1])
                      - (pixel_pt_x[1] - pixel_pt[1]) * (pixel_pt_y[0] - pixel_pt[0]))
        if 0 <= pixel_pt[0] < w and 0 <= pixel_pt[1] < h \
            and (np.linalg.norm(pitch_pt - pt) < config_dict['reprojection_eps']) \
            and ((not config_dict['enable_orientation']) or
                 (np.isfinite(local_area) and orientation_sign * local_area > 0)):
            mask[i] = True

    return mask
