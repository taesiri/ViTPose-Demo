#!/usr/bin/env python

from __future__ import annotations

import argparse
import pathlib
import tarfile
import base64
import io
import base64
from typing import List
from PIL import Image
import numpy as np
import json
from model import AppDetModel, AppPoseModel
from flask import Flask, request, jsonify
from flask import Flask, current_app

app = Flask(__name__)

DESCRIPTION = "# [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)"


def extract_tar() -> None:
    if pathlib.Path("mmdet_configs/configs").exists():
        return
    with tarfile.open("mmdet_configs/configs.tar") as f:
        f.extractall("mmdet_configs")


extract_tar()

det_model = AppDetModel()
pose_model = AppPoseModel()


def numpy_to_base64(arr: np.ndarray) -> str:
    arr_data = Image.fromarray(arr.astype(np.uint8))
    buffered = io.BytesIO()
    arr_data.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode("utf-8")


def base64_to_numpy(base64_img_string: str) -> np.ndarray:
    img_data = base64.b64decode(base64_img_string)
    img_data = Image.open(io.BytesIO(img_data))
    npimg = np.array(img_data)
    return npimg

def default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Decode image
    input_image = base64.b64decode(data["image"])
    input_image = Image.open(io.BytesIO(input_image))
    input_image = np.array(input_image)

    # Extract model parameters
    detector_name = data.get("detector", det_model.model_name)
    pose_model_name = data.get("pose_model", pose_model.model_name)
    vis_det_score_threshold = data.get("vis_det_score_threshold", 0.5)
    det_score_threshold = data.get("det_score_threshold", 0.5)
    vis_kpt_score_threshold = data.get("vis_kpt_score_threshold", 0.3)
    vis_dot_radius = data.get("vis_dot_radius", 4)
    vis_line_thickness = data.get("vis_line_thickness", 2)

    # Set detection model
    if detector_name != det_model.model_name:
        det_model.set_model(detector_name)

    # Set pose model
    if pose_model_name != pose_model.model_name:
        pose_model.set_model(pose_model_name)

    # Run detection model
    det_preds, detection_visualization, json_detect = det_model.run(
        detector_name,
        input_image,
        vis_det_score_threshold,
    )

    # Run pose model
    pose_preds, pose_visualization, json_pose = pose_model.run(
        pose_model_name,
        input_image,
        det_preds,
        det_score_threshold,
        vis_kpt_score_threshold,
        vis_dot_radius,
        vis_line_thickness,
    )

    # Convert numpy arrays to base64 for JSON serialization
    detection_visualization_base64 = numpy_to_base64(detection_visualization)
    pose_visualization_base64 = numpy_to_base64(pose_visualization)

    return current_app.response_class(
        response=json.dumps({
            'det_preds': det_preds,
            'detection_visualization_img': detection_visualization_base64,
            'json_detect': json_detect,
            'pose_preds': pose_preds,
            'pose_visualization_img': pose_visualization_base64,
            'json_pose': json_pose,
        }, default=default),
        mimetype=current_app.json.mimetype
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=7703)
