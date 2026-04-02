"""
image_to_spline.py
Converts a black-and-white sharpie line drawing into an ordered list of 2D points
representing the centerline (medial axis) of the drawn shape.
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist


def load_and_preprocess(image_path: str) -> np.ndarray:
    """Load image, threshold to clean binary, return bool array (True = ink)."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Slight blur to reduce noise from sharpie texture
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Otsu threshold — works well for sharpie on white paper
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological close to fill any gaps in the sharpie stroke
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closed.astype(bool)


def get_skeleton_points(binary: np.ndarray) -> np.ndarray:
    """Skeletonize the binary image and return (N, 2) array of [x, y] points."""
    skeleton = skeletonize(binary)
    yx_points = np.column_stack(np.where(skeleton))  # (row, col) = (y, x)
    xy_points = yx_points[:, ::-1]  # flip to (x, y)
    return xy_points


def order_points_by_nearest_neighbor(points: np.ndarray) -> np.ndarray:
    """
    Walk the skeleton points in order using nearest-neighbor traversal.
    Finds the best starting point (an endpoint or extremal point) and walks from there.
    Returns ordered (N, 2) array.
    """
    if len(points) == 0:
        raise ValueError("No skeleton points found. Check your image.")

    ordered = []
    remaining = points.tolist()

    # Start from the topmost-leftmost point as a heuristic
    start_idx = np.lexsort((points[:, 0], points[:, 1])).tolist()[0]
    current = remaining.pop(start_idx)
    ordered.append(current)

    while remaining:
        remaining_arr = np.array(remaining)
        dists = cdist([current], remaining_arr)[0]
        nearest_idx = np.argmin(dists)

        # If the nearest point is very far, we have a disconnected skeleton —
        # stop here (handles open curves gracefully)
        if dists[nearest_idx] > 20:
            break

        current = remaining.pop(nearest_idx)
        ordered.append(current)

    return np.array(ordered)


def fit_spline(ordered_points: np.ndarray, num_output_points: int = 500, smoothing: float = None):
    """
    Fit a parametric B-spline through the ordered skeleton points.
    Returns (N, 2) array of smoothed [x, y] points.
    
    smoothing: None = auto (scales with number of points), 0 = interpolate exactly
    """
    x = ordered_points[:, 0].astype(float)
    y = ordered_points[:, 1].astype(float)

    if smoothing is None:
        smoothing = len(x) * 2.0  # lenient smoothing by default

    # Detect if the shape is closed (start and end are close together)
    dist_start_end = np.hypot(x[-1] - x[0], y[-1] - y[0])
    is_closed = dist_start_end < 15  # pixels

    try:
        tck, u = splprep([x, y], s=smoothing, per=is_closed, k=3)
    except Exception:
        # Fallback: less smoothing
        tck, u = splprep([x, y], s=smoothing * 0.1, per=is_closed, k=3)

    u_new = np.linspace(0, 1, num_output_points)
    x_smooth, y_smooth = splev(u_new, tck)

    return np.column_stack([x_smooth, y_smooth]), is_closed


def image_to_spline(image_path: str, smoothing: float = None, num_points: int = 500):
    """
    Full pipeline: image path → smoothed 2D spline points + closed flag.
    
    Returns:
        points: (N, 2) float array in pixel coordinates (x right, y down)
        is_closed: bool — whether the shape appears to be a closed loop
    """
    print(f"[1/4] Loading and preprocessing {image_path}...")
    binary = load_and_preprocess(image_path)

    print("[2/4] Skeletonizing...")
    skel_pts = get_skeleton_points(binary)
    print(f"      Found {len(skel_pts)} skeleton points")

    print("[3/4] Ordering skeleton points...")
    ordered = order_points_by_nearest_neighbor(skel_pts)

    print("[4/4] Fitting spline...")
    spline_pts, is_closed = fit_spline(ordered, num_output_points=num_points, smoothing=smoothing)
    print(f"      Shape {'is' if is_closed else 'is NOT'} closed")

    return spline_pts, is_closed


if __name__ == "__main__":
    # Quick test with a generated circle image
    import matplotlib.pyplot as plt

    # Create a test image: a circle drawn in "sharpie" (thick ring)
    test_img = np.ones((400, 400), dtype=np.uint8) * 255
    cv2.circle(test_img, (200, 200), 120, 0, 12)
    cv2.imwrite("/tmp/test_circle.png", test_img)

    pts, closed = image_to_spline("/tmp/test_circle.png")
    print(f"Output: {len(pts)} points, closed={closed}")

    plt.figure(figsize=(6, 6))
    plt.plot(pts[:, 0], -pts[:, 1])  # flip Y for display
    plt.axis("equal")
    plt.title(f"Extracted spline ({'closed' if closed else 'open'})")
    plt.savefig("/tmp/test_spline.png")
    print("Saved preview to /tmp/test_spline.png")
