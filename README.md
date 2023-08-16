# YOLO Detector 

Built for the [LFX Mentorship 2023 03-Sep-Nov Challenge - #2702](https://github.com/WasmEdge/WasmEdge/discussions/2702)

## Required Packages



## Python Functions to be ported to Rust
### Preprocessing Functions:

`cv2.normalize` - To normalize images  
    - Candidate 

`cv2.resize` - To Resize images


### Post Processing functions:
TODO

`draw_detections(image, boxes, scores, class_ids)`

`rescale_boxes(boxes)`

`extract_boxes(predictions)`

`cv2.dnn.NMSBoxes``