import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def prepare_image(image: np.ndarray, shape: tuple) -> np.ndarray:
    """Prepare the image for the network.

    Currently only supports the following
    - Resize
    - Normalise
    """
    # Normalize for Object Detection
    image = cv2.normalize(
        src=image,
        dst=None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8UC1,
    )

    # Resize input image to match net size
    image = cv2.resize(image, shape)

    return image


def forward_pass(
    image: np.ndarray, network: cv2.dnn.Net, output_names: tuple
) -> np.ndarray:
    """Pass a pre-processed image through the network and extract the outputs.

    Will also apply 1/255 scaling to the input image
    """
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0)
    network.setInput(blob)
    return network.forward(output_names)[0]


def draw_detections(image, boxes, scores, class_ids):
    for box, score, class_id in zip(boxes, scores, class_ids):
        x, y, w, h = box.astype(int)
        color = colors[class_id]

        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=2)
        label = class_names[class_id]
        label = f"{label} {int(score * 100)}%"
        cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(
            image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2
        )
    return image


def rescale_boxes(boxes):
    input_shape = np.array([net_width, net_height, net_width, net_height])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    # img_width, img_height are global values
    boxes *= np.array([img_width, img_height, img_width, img_height])
    return boxes


def extract_boxes(predictions):
    # Extract boxes from predictions
    boxes = predictions[:, :4]

    # Scale boxes to original image dimensions
    boxes = rescale_boxes(boxes)

    # Convert boxes to xywh format
    boxes_ = np.copy(boxes)
    boxes_[..., 0] = boxes[..., 0] - boxes[..., 2] * 0.5
    boxes_[..., 1] = boxes[..., 1] - boxes[..., 3] * 0.5
    return boxes_


def process_output(
    outputs: np.ndarray,
    conf_threshold: float = 0.68,
    iou_threshold: float = 0.5,
)->tuple:
    """Process the output of the yolo model to get a collection of bounding boxes.

    Full Yolo model output is typicaly:
    - Grid X = Number of X grid cells (60)
    - Grid Y = Number of Y grid cells (40)
    - Num_Anchors = Number of anchor boxes per gird cell (7)
    - 85 = Box features
        - x location of center of object with cell coords
        - y location of center of object with cell coords
        - height of box ratio to cell height
        - width of box ratio to cell width
        - probability that anchor box contains an object center
        - ...: class probabilities for 80 classes

    First 3 dimensions are often merged
    """

    # Pop off the additional batch dimension
    outputs = np.squeeze(outputs)

    # Get the anchors the have a high confidence that they contain an object
    obj_conf = outputs[:, 4]
    confidence_mask =  obj_conf > conf_threshold
    outputs = outputs[confidence_mask]
    obj_conf = obj_conf[confidence_mask]

    # Multiply class confidence with bounding box confidence
    outputs[:, 5:] *= obj_conf[:, None]

    # Get the maximum class score
    scores = np.max(outputs[:, 5:], axis=-1)

    # Filter out the objects with a low score (ambiguous class)
    valid_mask = scores > conf_threshold
    outputs = outputs[valid_mask]
    scores = scores[valid_mask]

    # Get the class id with the highest confidence
    class_ids = np.argmax(outputs[:, 5:], axis=-1)

    # Get bounding boxes for each object
    boxes = extract_boxes(predictions)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold
    )

    if len(indices) > 0:
        indices = indices.flatten()

    return boxes[indices], scores[indices], class_ids[indices]


def main() -> None:
    """main script"""

    # Initial args
    model_path = "model/yolov7-tiny_480x640.onnx"
    class_name_path = "model/class_names.txt"
    image_folder = "images"
    conf_threshold = 0.68
    iou_threshold = 0.5

    # Load the pretrained network ONNX file and the associated class names
    net = cv2.dnn.readNet(model_path)
    output_names = net.getUnconnectedOutLayersNames()
    class_names = [x.strip() for x in open(class_name_path, "r").readlines()]

    # Get the expected image dimensions based on the model name
    inpt_shape = tuple(int(x) for x in Path(model_path).stem.split("_")[-1].split("x"))

    # Get colors to represent each class
    colors = np.random.default_rng(3).uniform(0, 255, size=(len(class_names), 3))

    # Get a list of images to run over
    for img_path in tqdm(Path(image_folder).glob("*"), "Drawing boxes on images"):
        image = cv2.imread(str(img_path))
        normalized_image = prepare_image(image, inpt_shape)
        output = forward_pass(normalized_image, net, output_names)
        boxes, scores, class_ids = process_output(output, conf_threshold, iou_threshold)

        # 5. Draw Bounding boxes on images
        output_img = draw_detections(image, boxes, scores, class_ids)

    cv2.imwrite("./image_out.jpg", output_img)
    print("image_out.jpg written, exit")


if __name__ == "__main__":
    main()
