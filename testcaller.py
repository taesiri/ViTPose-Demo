import requests
import json
import base64


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
    
# Read an image file and encode it as base64
with open("/home/user/app/images/pexels-cottonbro-5770445.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# Specify the model parameters you want to use
data = {
  "image": encoded_string,
  "detector": "YOLOX-l",
  "pose_model": "ViTPose-B (multi-task train, COCO)",
  "vis_det_score_threshold": 0.5,
  "det_score_threshold": 0.5,
  "vis_kpt_score_threshold": 0.3,
  "vis_dot_radius": 4,
  "vis_line_thickness": 2
}

# Make the POST request
response = requests.post('http://localhost:7703/predict', json=data)

# Save the response to a file
with open('response-test.json', 'w') as f:
    json.dump(response.json(), f)
