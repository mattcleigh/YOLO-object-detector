"""Executable script to apply bounding boxes to images."""

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


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_names: list,
    class_colors: np.ndarray,
) -> np.ndarray:
    """Draw the bounding boxes onto the pixel values of the image."""
    for box, score, class_id in zip(boxes, scores, class_ids):
        # Get the pixes coordinates and the appropriate colour and name
        x, y, w, h = box.astype(int)
        color = class_colors[class_id]
        label = f"{class_names[class_id]} {int(score * 100)}%"

        # Draw rectangle onto the image
        lw = int(0.005 * min(image.shape[:2]))
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=lw)
        # cv2.getTextSize(
        #     label,
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     lw//5,
        #     lw//2
        # )
        cv2.putText(
            image,
            label,
            org=(x, y - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=np.round(np.sqrt(lw)),
            color=color,
            thickness=lw // 2,
        )
    return image


def extract_boxes(predictions: np.ndarray, size_ratio: np.ndarray) -> np.ndarray:
    """Pull out the bounding boxes from the predictions and resize to original."""

    # Extract boxes from predictions (x_center, y_center, width, height)
    boxes = predictions[:, :4]

    # Scale boxes to original image dimensions
    boxes *= np.array([*size_ratio, *size_ratio])

    # Change the format to (x_left, y_bottom, width, height)
    boxes[..., 0] -= boxes[..., 2] * 0.5
    boxes[..., 1] -= boxes[..., 3] * 0.5
    return boxes


def process_output(
    outputs: np.ndarray,
    size_ratio: np.ndarray,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> tuple[np.ndarray]:
    """Process the output of the yolo model to get a collection of bounding boxes.

    Full Yolo model output is typicaly:
    - Grid X = Number of X grid cells (60)
    - Grid Y = Number of Y grid cells (40)
    - Num_Anchors = Number of anchor boxes per gird cell (7)
    - 85 = Box features (x1, y1, width, height, prob_object, *prob_classes)
    First 3 dimensions are often merged
    """

    # Pop off the additional batch dimension
    predictions = np.squeeze(outputs)

    # Get the anchors the have a high confidence that they contain an object
    obj_conf = predictions[:, 4]
    confidence_mask = obj_conf > conf_threshold
    predictions = predictions[confidence_mask]
    obj_conf = obj_conf[confidence_mask]

    # Multiply class confidence with bounding box confidence
    predictions[:, 5:] *= obj_conf[:, None]  # Is this necc?

    # Get the maximum class score
    max_scores = np.max(predictions[:, 5:], axis=-1)

    # Filter out boxes where the class is ambiguous (low max score)
    valid_mask = max_scores > conf_threshold
    predictions = predictions[valid_mask]
    max_scores = max_scores[valid_mask]

    # Get the class id corresponding to the max score
    class_ids = np.argmax(predictions[:, 5:], axis=-1)

    # Get bounding boxes for each object
    boxes = extract_boxes(predictions, size_ratio)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), max_scores.tolist(), conf_threshold, iou_threshold
    )

    # Return the surviving boxes, their scores and class ids
    return boxes[indices], max_scores[indices], class_ids[indices]


def main() -> None:
    """Run script."""

    # Initial args
    model_path = "model/yolov7-tiny_480x640.onnx"
    class_name_path = "model/class_names.txt"
    image_folder = "images"
    output_folder = "outputs"
    conf_threshold = 0.5
    iou_threshold = 0.5

    # Load the pretrained network ONNX file and the associated class names
    net = cv2.dnn.readNet(model_path)
    output_names = net.getUnconnectedOutLayersNames()

    # Get the expected image dimensions based on the model name (width x height)
    inpt_shape = tuple(int(x) for x in Path(model_path).stem.split("_")[-1].split("x"))
    inpt_shape = inpt_shape[::-1]

    # Get names and colors to represent each class
    class_names = [x.strip() for x in open(class_name_path).readlines()]
    class_colors = np.random.default_rng(3).uniform(0, 255, size=(len(class_names), 3))

    # Get a list of images to run over
    file_list = list(Path(image_folder).glob("*"))
    for img_path in tqdm(file_list, "Drawing boxes on images"):
        image = cv2.imread(str(img_path))

        # Get the size ratio (width x height) to reposition the boxes
        size_ratio = np.divide(image.shape[1::-1], inpt_shape)

        # Prepare the image and pass through the network
        prepared_image = prepare_image(image, inpt_shape)
        output = forward_pass(prepared_image, net, output_names)

        # Get a list of bounding boxes in the original img dimensions
        boxes, scores, class_ids = process_output(
            output, size_ratio, conf_threshold, iou_threshold
        )

        # Draw the bounding boxes onto the original image and save
        output_img = draw_boxes(
            image, boxes, scores, class_ids, class_names, class_colors
        )
        cv2.imwrite(str(Path(output_folder, img_path.name)), output_img)


if __name__ == "__main__":
    main()
