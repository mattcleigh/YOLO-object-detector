# YOLO Detector 

Built for the [LFX Mentorship 2023 03-Sep-Nov Challenge - #2702](https://github.com/WasmEdge/WasmEdge/discussions/2702)

## Required Packages



## Python Functions to be ported to Rust
### Preprocessing:

`cv2.normalize` - To normalize images  
    - potentially exists [opencvmini](https://github.com/second-state/opencvmini/blob/main/src/lib.rs#L39C20-L39C20) 

`cv2.resize` - To Resize images
    - to be added to opencv mini ? 

`cv2.dnn.readNet` -  loads networks


### Post Processing:

`draw_detections(image, boxes, scores, class_ids)`

`rescale_boxes(boxes)`

`extract_boxes(predictions)`

`cv2.dnn.NMSBoxes`

`cv2.getTextSize`

`cv2.putText`

Maybe it makes sense to have a single function to take the output of the forward pass of a YOLO net output, 
and return the bounding boxes and classnames. 
It could be comprised of all the functions above 


### Helper Functions: 
`cv2.imshow`
`cv2.waitKey`
`cv2.destroyAllWindows`
We may want to decouple this to the actual Yolo crate, as these functions will require peripherals and some idea of a windowing system. 
So depending on the design of the YOLO Crate, it may make sense to put these elsewhere.