"""
app.py — Cookie Cutter Generator Web App
Flask backend handling image processing, preview, and STL generation.
"""

import os
import uuid
import json
import tempfile
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB max upload

# Temp directory for STL files (cleaned up periodically)
TEMP_DIR = Path(tempfile.gettempdir()) / "cookie_cutter_outputs"
TEMP_DIR.mkdir(exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/preview", methods=["POST"])
def preview():
    """
    Step 1: Receive uploaded image, run skeletonization, return spline points as JSON.
    The frontend draws these on a canvas so the user can verify the shape.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        import numpy as np
        import cv2
        from image_to_spline import image_to_spline

        # Save upload to temp file
        suffix = Path(file.filename).suffix or ".png"
        tmp_path = TEMP_DIR / f"upload_{uuid.uuid4().hex}{suffix}"
        file.save(str(tmp_path))

        # Run pipeline
        spline_px, is_closed = image_to_spline(str(tmp_path))

        # Get image dimensions for the canvas
        img = cv2.imread(str(tmp_path), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape

        # Clean up upload
        tmp_path.unlink(missing_ok=True)

        # Normalize points to 0–1 range so frontend can scale to canvas size
        x = spline_px[:, 0]
        y = spline_px[:, 1]
        x_norm = ((x - x.min()) / (x.max() - x.min())).tolist()
        y_norm = ((y - y.min()) / (y.max() - y.min())).tolist()

        return jsonify({
            "points_x": x_norm,
            "points_y": y_norm,
            "is_closed": bool(is_closed),
            "num_points": len(x_norm),
            "image_aspect": float(w) / float(h),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Step 2: Receive image + settings, run full pipeline, return STL file.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        import numpy as np
        from image_to_spline import image_to_spline
        from spline_to_stl import pixel_pts_to_mm, build_cookie_cutter_v2, save_stl

        file = request.files["image"]
        wall_height    = float(request.form.get("wall_height", 50))
        wall_thickness = float(request.form.get("wall_thickness", 2.0))
        flange_width   = float(request.form.get("flange_width", 5.0))
        flange_height  = float(request.form.get("flange_height", 2.5))
        force_size_mm  = request.form.get("force_size", "").strip()
        force_size_mm  = float(force_size_mm) if force_size_mm else None
        dpi            = float(request.form.get("dpi", 96))

        # Save upload
        suffix = Path(file.filename).suffix or ".png"
        tmp_input  = TEMP_DIR / f"input_{uuid.uuid4().hex}{suffix}"
        tmp_output = TEMP_DIR / f"output_{uuid.uuid4().hex}.stl"
        file.save(str(tmp_input))

        # Run pipeline
        spline_px, is_closed = image_to_spline(str(tmp_input))

        pixels_per_mm = dpi / 25.4
        if force_size_mm:
            extent = max(
                spline_px[:, 0].max() - spline_px[:, 0].min(),
                spline_px[:, 1].max() - spline_px[:, 1].min(),
            )
            pixels_per_mm = extent / force_size_mm

        spline_mm = pixel_pts_to_mm(spline_px, pixels_per_mm)

        cutter = build_cookie_cutter_v2(
            spline_mm,
            is_closed=is_closed,
            wall_height=wall_height,
            wall_thickness=wall_thickness,
            base_flange_width=flange_width,
            base_flange_height=flange_height,
        )

        save_stl(cutter, str(tmp_output))
        tmp_input.unlink(missing_ok=True)

        return send_file(
            str(tmp_output),
            as_attachment=True,
            download_name="cookie_cutter.stl",
            mimetype="application/octet-stream",
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
