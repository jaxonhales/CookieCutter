"""
spline_to_stl.py
Takes 2D spline points (from image_to_spline.py) and builds a cookie-cutter STL:

Geometry:
  - Thin wall (default 2mm) extruded to full height (default 50mm)
  - Inner base flange (default 5mm wide, 2.5mm tall) for stability
  - Optional: slightly tapered top edge for sharpness

                Side view:
    ┌──┐  ← wall top (full height, 50mm)
    │  │
    │  │
    │  └──────┐  ← inner flange starts (5mm inward, 2.5mm tall)
    └─────────┘  ← base Z=0
"""

import cadquery as cq
import numpy as np


def pixel_pts_to_mm(points: np.ndarray, pixels_per_mm: float) -> np.ndarray:
    """Convert pixel-space points to mm, flipping Y axis (image Y is down)."""
    mm = points / pixels_per_mm
    mm[:, 1] = -mm[:, 1]  # flip Y so shape isn't mirrored
    # Center around origin
    mm[:, 0] -= mm[:, 0].mean()
    mm[:, 1] -= mm[:, 1].mean()
    return mm


def build_cookie_cutter(
    spline_pts_mm: np.ndarray,
    is_closed: bool,
    wall_height: float = 50.0,
    wall_thickness: float = 2.0,
    base_flange_width: float = 5.0,
    base_flange_height: float = 2.5,
    taper_angle_deg: float = 0.0,  # set >0 (e.g. 5) for a sharper cutting edge
) -> cq.Workplane:
    """
    Build a cookie-cutter solid from 2D mm spline points.

    Returns a CadQuery Workplane containing the solid.
    """

    # Convert numpy array to list of (x, y) tuples for CadQuery
    pts = [(float(p[0]), float(p[1])) for p in spline_pts_mm]

    # Close the loop if needed by appending the first point
    if is_closed and pts[0] != pts[-1]:
        pts.append(pts[0])

    # ── Build the wall profile ──────────────────────────────────────────────
    # We'll use CadQuery's offset wire approach:
    # 1. Create the centerline wire from spline points
    # 2. Offset outward by half wall_thickness, inward by half wall_thickness
    # 3. Loft between them (or just extrude a 2D shell profile)

    half = wall_thickness / 2.0

    # Build a 2D "thick wall" cross-section by working with the spline as a
    # closed wire, then shelling. CadQuery makes this straightforward:
    # - Make a face from the closed spline
    # - Extrude it (solid)
    # - Shell it inward to wall_thickness (removes the fill, leaves just walls)
    # - Add the flange separately

    wp = cq.Workplane("XY")

    if is_closed:
        # Make a filled face, extrude to full height, then shell
        face = wp.spline(pts).close().extrude(wall_height)

        # Shell: remove top face, keep wall_thickness walls
        # CadQuery shell with positive offset = removes inner material
        try:
            walled = face.shell(-wall_thickness, kind="intersection")
        except Exception:
            # Fallback: shell with arc kind
            walled = face.shell(-wall_thickness)

        # ── Base flange ──────────────────────────────────────────────────────
        # Inner offset of the centerline by (half + flange_width), make a filled
        # face, extrude flange_height, then subtract the hole (inner offset by half)

        # Build outer flange boundary (inner wall edge = center offset inward by half)
        # Build inner flange boundary (center offset inward by half + flange_width)
        # We do this by making two spline polygons and using a 2D boolean

        flange_outer_pts = pts  # the original spline = outer edge of the flange
        # We rely on CadQuery's offset for the inner boundary
        # Simpler: make a full disc at base height, subtract the inner hole

        flange_disk = (
            cq.Workplane("XY")
            .spline(flange_outer_pts).close()
            .extrude(base_flange_height)
        )

        # Inner hole = the shape inset by (wall_thickness + base_flange_width)
        # We approximate this by shelling the disk very aggressively
        # Actually: shell the full solid by (wall_thickness + base_flange_width)
        try:
            flange_shell = flange_disk.shell(-(wall_thickness + base_flange_width))
        except Exception:
            flange_shell = flange_disk.shell(-(wall_thickness + base_flange_width), kind="arc")

        # Combine wall + flange
        result = walled.union(flange_shell)

    else:
        # Open curve: extrude a thick wall directly
        # Make a rectangle profile (wall cross-section) swept along the path
        # CadQuery sweep: make a rect profile, sweep along the spline
        profile = cq.Workplane("YZ").rect(wall_thickness, wall_height)
        path_wire = cq.Workplane("XY").spline(pts)
        result = profile.sweep(path_wire)

        # No flange for open curves (no enclosed area)
        print("  Note: open curve detected — no base flange added.")

    return result


def build_cookie_cutter_v2(
    spline_pts_mm: np.ndarray,
    is_closed: bool,
    wall_height: float = 50.0,
    wall_thickness: float = 2.0,
    base_flange_width: float = 5.0,
    base_flange_height: float = 2.5,
) -> cq.Workplane:
    """
    More robust approach using explicit polygon offsetting (shapely) 
    to build the wall and flange cross-sections, then extrude each.
    Shapely offset is much more reliable than CadQuery shell for complex splines.
    """
    from shapely.geometry import LinearRing, LineString, Polygon
    from shapely.ops import unary_union

    pts_list = [(float(p[0]), float(p[1])) for p in spline_pts_mm]

    if is_closed:
        if pts_list[0] != pts_list[-1]:
            pts_list.append(pts_list[0])
        ring = LinearRing(pts_list)
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)  # fix self-intersections

        # ── Wall: band between outer and inner offsets ──────────────────────
        outer = poly.buffer(wall_thickness / 2)
        inner_wall = poly.buffer(-wall_thickness / 2)
        wall_ring = outer.difference(inner_wall)

        # ── Base flange: inner_wall minus deep_inner ────────────────────────
        deep_inner = poly.buffer(-(wall_thickness / 2 + base_flange_width))
        flange_ring = inner_wall.difference(deep_inner)

        def shapely_to_cq_wire(shapely_poly, workplane="XY"):
            """Convert a shapely Polygon to a CadQuery face."""
            coords = list(shapely_poly.exterior.coords)
            pts = [(float(x), float(y)) for x, y in coords]
            return cq.Workplane(workplane).polyline(pts).close()

        def extrude_shapely_poly(shapely_poly, height, z_offset=0.0):
            """Extrude a (possibly multi) shapely polygon to a CQ solid."""
            from shapely.geometry import MultiPolygon
            if isinstance(shapely_poly, MultiPolygon):
                geoms = list(shapely_poly.geoms)
            else:
                geoms = [shapely_poly]

            solids = []
            for geom in geoms:
                coords = list(geom.exterior.coords)
                cq_pts = [(float(x), float(y)) for x, y in coords]
                solid = (
                    cq.Workplane("XY")
                    .workplane(offset=z_offset)
                    .polyline(cq_pts).close()
                    .extrude(height)
                )
                # Subtract holes (e.g. for wall_ring the interior is the open space)
                for hole in geom.interiors:
                    hole_pts = [(float(x), float(y)) for x, y in hole.coords]
                    hole_solid = (
                        cq.Workplane("XY")
                        .workplane(offset=z_offset)
                        .polyline(hole_pts).close()
                        .extrude(height)
                    )
                    solid = solid.cut(hole_solid)
                solids.append(solid)

            if len(solids) == 1:
                return solids[0]
            result = solids[0]
            for s in solids[1:]:
                result = result.union(s)
            return result

        print("  Building wall solid...")
        wall_solid = extrude_shapely_poly(wall_ring, wall_height)

        print("  Building base flange solid...")
        flange_solid = extrude_shapely_poly(flange_ring, base_flange_height)

        print("  Combining wall + flange...")
        result = wall_solid.union(flange_solid)

    else:
        # Open path: sweep a rectangular profile
        print("  Open curve: sweeping rectangular profile...")
        pts_cq = [(float(p[0]), float(p[1])) for p in spline_pts_mm]
        path = cq.Workplane("XY").spline(pts_cq)
        profile = cq.Workplane("YZ").rect(wall_thickness, wall_height)
        result = profile.sweep(path)

    return result


def save_stl(workplane: cq.Workplane, output_path: str, tolerance: float = 0.1):
    """Export the CadQuery solid to an STL file."""
    print(f"  Exporting to {output_path}...")
    cq.exporters.export(workplane.val(), output_path, exportType="STL", tolerance=tolerance)
    print(f"  Done! STL saved to {output_path}")


if __name__ == "__main__":
    # Quick test: circle cookie cutter
    theta = np.linspace(0, 2 * np.pi, 200)
    r = 40  # 40mm radius
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    print("Building test circle cookie cutter...")
    result = build_cookie_cutter_v2(pts, is_closed=True)
    save_stl(result, "/tmp/test_cookie_cutter.stl")
