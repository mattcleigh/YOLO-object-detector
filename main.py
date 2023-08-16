import os
import cv2
import numpy as np

# Initialize the NET
model = "yolov7-tiny_480x640.onnx"
net = cv2.dnn.readNet(model)
output_names = net.getUnconnectedOutLayersNames()

# Get Net Width + Height
input_shape = os.path.splitext(os.path.basename(model))[
    0].split('_')[-1].split('x')
net_height = int(input_shape[0])
net_width = int(input_shape[1])

# Load Class names
class_names = list(
    map(lambda x: x.strip(), open('class.names', 'r').readlines()))

conf_threshold = 0.7
iou_threshold = 0.5
colors = np.random.default_rng(3).uniform(
    0, 255, size=(len(class_names), 3))


def prepare_image(image):

    # Normalize for Object Detection
    normalized_image = cv2.normalize(
        image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
    )

    # Resize input image to match net size
    input_img = cv2.resize(
        normalized_image, (net_width, net_height)
    )

    return input_img


def detect(processed_image):

    blob = cv2.dnn.blobFromImage(processed_image, 1 / 255.0)

    # Perform inference on the image
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outputs = net.forward(output_names)

    return outputs


def draw_detections(image, boxes, scores, class_ids):
    for box, score, class_id in zip(boxes, scores, class_ids):
        x, y, w, h = box.astype(int)
        color = colors[class_id]

        # Draw rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness=2)
        label = class_names[class_id]
        label = f'{label} {int(score * 100)}%'
        cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)
    return image


def rescale_boxes(boxes):
    input_shape = np.array(
        [net_width, net_height, net_width, net_height])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([img_width, img_height,
                       img_width, img_height])
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


def process_output(output):

    predictions = np.squeeze(output[0])

    # Filter out object confidence scores below threshold
    obj_conf = predictions[:, 4]

    predictions = predictions[obj_conf > conf_threshold]
    obj_conf = obj_conf[obj_conf > conf_threshold]

    # Multiply class confidence with bounding box confidence
    predictions[:, 5:] *= obj_conf[:, np.newaxis]

    # Get the scores
    scores = np.max(predictions[:, 5:], axis=1)

    # Filter out the objects with a low score
    valid_scores = scores > conf_threshold
    predictions = predictions[valid_scores]
    scores = scores[valid_scores]

    # Get the class with the highest confidence
    class_ids = np.argmax(predictions[:, 5:], axis=1)

    # Get bounding boxes for each object
    boxes = extract_boxes(predictions)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)
    if len(indices) > 0:
        indices = indices.flatten()

    return boxes[indices], scores[indices], class_ids[indices]


if __name__ == "__main__":

    # 1. Load image
    image = cv2.imread("./image.jpg")
    img_height, img_width = image.shape[:2]  # [h,w,d]

    # 2. Prepare Image
    normalized_image = prepare_image(image)

    # 3. Forward Pass through Net
    net_output = detect(normalized_image)

    # 4. Process Net outputs
    boxes, scores, class_ids = process_output(net_output)

    # 5. Draw Bounding boxes on images
    output_img = draw_detections(image, boxes, scores, class_ids)

    cv2.imwrite("./image_out.jpg", output_img)

    # cv2.imshow("Output image", output_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
