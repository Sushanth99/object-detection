# Object detection using YOLOv3 and YOLOv4

## Usage

```
curl -X 'POST' \
  '<HOST_NAME>/object-detection?model=<MODEL>' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@<FILE_PATH>;type=image/jpeg'
  --output <OUTPUT_FILENAME>
```

Example
```
curl -X 'POST' \
  'http://127.0.0.1:8000/object-detection?model=yolov4-tiny' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@images/pexels-mareefe-672101.jpeg;type=image/jpeg'
  --output images_yolov4-tiny/pexels-mareefe-672101.jpeg
```

Models available: yolov3-tiny, yolov3, yolov4-tiny, yolov4

## References
The API is based on Coursera's Introduction to Machine Learning in Production course by Deeplearning.ai.
