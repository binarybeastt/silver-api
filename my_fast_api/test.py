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
model = YOLO('best.pt')
tracker = sv.ByteTrack()

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

def predict():
    frames_generator = sv.get_video_frames_generator(source_path='test\sample1.mp4')
    video_info = sv.VideoInfo.from_video_path('test\sample1.mp4')
    with sv.VideoSink(target_path='output/target2.mp4', video_info=video_info) as sink:
        # Process each frame and write annotated frames to VideoSink
        for frame in frames_generator:
            annotated_frame = get_model_prediction_with_tracking(model, tracker, frame)
            sink.write_frame(frame=annotated_frame)

predict()