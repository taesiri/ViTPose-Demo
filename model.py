from __future__ import annotations

import os
import pathlib
import shlex
import subprocess
import sys
import json

# if os.getenv('SYSTEM') == 'spaces':
#     import mim

#     mim.uninstall('mmcv-full', confirm_yes=True)
#     mim.install('mmcv-full==1.5.0', is_yes=True)

#     subprocess.run(shlex.split('pip uninstall -y opencv-python'))
#     subprocess.run(shlex.split('pip uninstall -y opencv-python-headless'))
#     subprocess.run(shlex.split('pip install opencv-python-headless==4.5.5.64'))

import huggingface_hub
import numpy as np
import torch
import torch.nn as nn

app_dir = pathlib.Path(__file__).parent
submodule_dir = app_dir / "ViTPose"
sys.path.insert(0, submodule_dir.as_posix())

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (
    inference_top_down_pose_model,
    init_pose_model,
    process_mmdet_results,
    vis_pose_result,
)

HF_TOKEN = os.getenv("HF_TOKEN")


class DetModel:
    MODEL_DICT = {
        "YOLOX-tiny": {
            "config": "mmdet_configs/configs/yolox/yolox_tiny_8x8_300e_coco.py",
            "model": "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth",
        },
        "YOLOX-s": {
            "config": "mmdet_configs/configs/yolox/yolox_s_8x8_300e_coco.py",
            "model": "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth",
        },
        "YOLOX-l": {
            "config": "mmdet_configs/configs/yolox/yolox_l_8x8_300e_coco.py",
            "model": "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth",
        },
        "YOLOX-x": {
            "config": "mmdet_configs/configs/yolox/yolox_x_8x8_300e_coco.py",
            "model": "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth",
        },
    }

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._load_all_models_once()
        self.model_name = "YOLOX-l"
        self.model = self._load_model(self.model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        dic = self.MODEL_DICT[name]
        return init_detector(dic["config"], dic["model"], device=self.device)

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def detect_and_visualize(
        self, image: np.ndarray, score_threshold: float
    ) -> tuple[list[np.ndarray], np.ndarray]:
        out, outjson = self.detect(image)
        vis = self.visualize_detection_results(image, out, score_threshold)
        return out, vis, outjson

    def detect(self, image: np.ndarray) -> list[np.ndarray]:
        image = image[:, :, ::-1]  # RGB -> BGR
        out = inference_detector(self.model, image)
        # Convert numpy arrays to lists
        out2 = [arr.tolist() for arr in out]
        # Convert output to JSON
        out_json = json.dumps(out2)
        return out, out_json

    def visualize_detection_results(
        self,
        image: np.ndarray,
        detection_results: list[np.ndarray],
        score_threshold: float = 0.3,
    ) -> np.ndarray:
        person_det = [detection_results[0]] + [np.array([]).reshape(0, 5)] * 79

        image = image[:, :, ::-1]  # RGB -> BGR
        vis = self.model.show_result(
            image,
            person_det,
            score_thr=score_threshold,
            bbox_color=None,
            text_color=(200, 200, 200),
            mask_color=None,
        )
        return vis[:, :, ::-1]  # BGR -> RGB


class AppDetModel(DetModel):
    def run(
        self, model_name: str, image: np.ndarray, score_threshold: float
    ) -> tuple[list[np.ndarray], np.ndarray]:
        self.set_model(model_name)
        return self.detect_and_visualize(image, score_threshold)


class PoseModel:
    MODEL_DICT = {
        "ViTPose-B (single-task train)": {
            "config": "ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py",
            "model": "vitpose-b.pth",
        },
        "ViTPose-L (single-task train)": {
            "config": "ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py",
            "model": "vitpose-l.pth",
        },
        "ViTPose-B (multi-task train, COCO)": {
            "config": "ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py",
            "model": "vitpose-b-multi-coco.pth",
        },
        "ViTPose-L (multi-task train, COCO)": {
            "config": "ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py",
            "model": "vitpose-l-multi-coco.pth",
        },
    }

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_name = "ViTPose-B (multi-task train, COCO)"
        self.model = self._load_model(self.model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        dic = self.MODEL_DICT[name]
        ckpt_path = huggingface_hub.hf_hub_download(
            "taesiri/ViTPose", dic["model"], use_auth_token=HF_TOKEN
        )
        model = init_pose_model(dic["config"], ckpt_path, device=self.device)
        return model

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def predict_pose_and_visualize(
        self,
        image: np.ndarray,
        det_results: list[np.ndarray],
        box_score_threshold: float,
        kpt_score_threshold: float,
        vis_dot_radius: int,
        vis_line_thickness: int,
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
        out, outjson = self.predict_pose(image, det_results, box_score_threshold)
        vis = self.visualize_pose_results(
            image, out, kpt_score_threshold, vis_dot_radius, vis_line_thickness
        )
        return out, vis, outjson

    def predict_pose(
        self,
        image: np.ndarray,
        det_results: list[np.ndarray],
        box_score_threshold: float = 0.5,
    ) -> list[dict[str, np.ndarray]]:
        image = image[:, :, ::-1]  # RGB -> BGR
        person_results = process_mmdet_results(det_results, 1)
        out, _ = inference_top_down_pose_model(
            self.model,
            image,
            person_results=person_results,
            bbox_thr=box_score_threshold,
            format="xyxy",
        )
        # return out
        out_for_json = [
            {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in person.items()
            }
            for person in out
        ]
        # Convert output to JSON
        outjson = json.dumps(out_for_json)

        return out, outjson

    def visualize_pose_results(
        self,
        image: np.ndarray,
        pose_results: list[np.ndarray],
        kpt_score_threshold: float = 0.3,
        vis_dot_radius: int = 4,
        vis_line_thickness: int = 1,
    ) -> np.ndarray:
        image = image[:, :, ::-1]  # RGB -> BGR
        vis = vis_pose_result(
            self.model,
            image,
            pose_results,
            kpt_score_thr=kpt_score_threshold,
            radius=vis_dot_radius,
            thickness=vis_line_thickness,
        )
        return vis[:, :, ::-1]  # BGR -> RGB


class AppPoseModel(PoseModel):
    def run(
        self,
        model_name: str,
        image: np.ndarray,
        det_results: list[np.ndarray],
        box_score_threshold: float,
        kpt_score_threshold: float,
        vis_dot_radius: int,
        vis_line_thickness: int,
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
        self.set_model(model_name)
        return self.predict_pose_and_visualize(
            image,
            det_results,
            box_score_threshold,
            kpt_score_threshold,
            vis_dot_radius,
            vis_line_thickness,
        )
