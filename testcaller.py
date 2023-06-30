import os
import numpy as np
from PIL import Image
import base64
import io
import requests
import json

# Define helper functions
def base64_to_numpy(base64_img_string: str) -> np.ndarray:
    img_data = base64.b64decode(base64_img_string)
    img_data = Image.open(io.BytesIO(img_data))
    npimg = np.array(img_data)
    return npimg

def save_image_from_np_array(np_array, filename):
    image = Image.fromarray(np_array)
    image.save(filename)

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

# Save the images to a folder
response_data = response.json()

# Define the output directory
output_dir = './outputs'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through each key-value pair in the JSON response
for key, base64_image in response_data.items():
    if key.endswith('_img'):  # Ensure we're working with an image
        np_array = base64_to_numpy(base64_image)
        output_filename = os.path.join(output_dir, f'{key}.jpg')
        save_image_from_np_array(np_array, output_filename)
