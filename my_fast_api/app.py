import logging
from PIL import Image
import io
import supervision as sv
from ultralytics import YOLO
import numpy as np
import cv2 
from typing import List
#annotators
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_image_from_path_or_bytes(image_source) -> Image:
    """
    Load an image from either a file path or image data in bytes.

    Args:
        image_source (str or bytes): Either a file path or image data in bytes.

    Returns:
        PIL.Image: The loaded image.

    Raises:
        ValueError: If the input type is unsupported.
    """
    if isinstance(image_source, str):
        input_image = Image.open(image_source).convert('RGB')
    elif isinstance(image_source, bytes):
        input_image = Image.open(io.BytesIO(image_source)).convert('RGB')
    else:
        raise ValueError("Unsupported input type. Please provide a file path or image data in bytes.")

    return input_image

def get_model_prediction(model: YOLO, input_image: Image):
    """
    Perform car damage detection on the input image using the specified YOLO model.

    Args:
        model (YOLO): The YOLO model for car damage detection.
        input_image (PIL.Image): The input image.

    Returns:
        list: A list of dictionaries containing the detected damage categories and their severity.
    """
    try:
        results = model(input_image)[0]
        detections = sv.Detections.from_ultralytics(results)
        class_ids = detections.class_id
        areas = detections.box_area
        class_names = ['dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat']

        result_list = [{str(class_ids[i]): areas[i]} for i in range(min(len(class_ids), len(areas)))]
        minor_threshold = 10000
        moderate_threshold = 10000000

        result_with_names = []

        for item in result_list:
            key, value = list(item.items())[0]
            area = int(value)

            # Determine the threshold category
            if area < minor_threshold:
                category = 'minor'
            elif area < moderate_threshold:
                category = 'moderate'
            else:
                category = 'severe'
            name = class_names[int(key)]
            result_with_names.append({name: category})

        return result_with_names

    except Exception as e:
        logger.exception("An error occurred during model prediction: %s", str(e))
        raise RuntimeError("An error occurred during model prediction")

# try:
#     model = YOLO('artifacts/run_hd27bn94_model:v0/best.pt')
#     input_image = get_image_from_path_or_bytes('test/000047.jpg')
#     result_dict = get_model_prediction(model=model, input_image=input_image)
#     print(result_dict)
# except Exception as e:
#     logger.exception("An error occurred: %s", str(e))
#     print("An error occurred. Please check the logs for details.")


def get_frames_from_video(video_source: str) -> List[Image.Image]:
    """
    Extract frames from a video file.

    Args:
        video_source (str): Path to the video file.

    Returns:
        List[PIL.Image.Image]: List of frames as PIL Image objects.
    """
    cap = cv2.VideoCapture(video_source)  # Ensure video_source is a string representing the file path
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to PIL Image and append to the list
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames

def get_model_prediction_with_tracking(model: YOLO, tracker: sv.ByteTrack, input_image: Image) -> np.ndarray:
    """
    Perform car damage detection with object tracking on the input image using the specified YOLO model and tracker.

    Args:
        model (YOLO): The YOLO model for car damage detection.
        tracker (sv.ByteTrack): The object tracker.
        input_image (PIL.Image): The input image.

    Returns:
        np.ndarray: Annotated image with detected objects and labels.
    """
    frame_array = np.array(input_image)  # Convert PIL image to numpy array
    results = model(frame_array)[0]  # Perform object detection
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(scene=frame_array.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)

    return annotated_frame