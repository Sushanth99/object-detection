
import os

import cv2
import cvlib
# from cvlib.object_detection import draw_bbox

import io
import pathlib
import uvicorn
import numpy as np
# import nest_asyncio
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

# print('I am here 1')
def draw_boxes(filename, model="yolov3-tiny", confidence=0.5):    
    # print('I am here 2')
    img = cv2.imread(filename)
    bbox, label, conf = cvlib.detect_common_objects(img, confidence=confidence, model=model)
    
    for l, c in zip(label, conf):
        print(f"Detected object: {l} with confidence level of {c}\n")
    
    output_image = cvlib.object_detection.draw_bbox(img, bbox, label, conf, write_conf=True)
    cv2.imwrite(f'images_{model}/{filename.split("/")[-1]}', output_image)

## Test for draw_boxes function
# for image_file in os.listdir('images'):
#     draw_boxes(f'images/{image_file}')


app = FastAPI(title='Object Detection')

class Model(str, Enum):
    yolov3tiny = "yolov3-tiny"
    yolov3 = "yolov3"
    yolov4 = "yolov4"
    yolov4tiny = "yolov4-tiny"


@app.get("/")
def home():
    return "Please go to http://localhost:8000/docs to try out the API"


@app.post("/object-detection") 
def prediction(model: Model, file: UploadFile = File(...)):

    filename = file.filename
    file_extension = pathlib.Path(filename).suffix in (".jpg", ".jpeg", ".png")
    print(file_extension)
    if not file_extension:
        raise HTTPException(status_code=415, detail="Unsupported file. Only images of JPEG/JPG and PNG are allowed.")
    
    image_stream = io.BytesIO(file.file.read())
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    bbox, label, conf = cvlib.detect_common_objects(image, model=model)
    output_image = cvlib.object_detection.draw_bbox(image, bbox, label, conf, write_conf=True)

    output_folder = f"images_{model}"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    cv2.imwrite(f'{output_folder}/{filename}', output_image)

    file_image = open(f'{output_folder}/{filename}', mode="rb")

    return StreamingResponse(file_image, media_type="image/jpeg")

