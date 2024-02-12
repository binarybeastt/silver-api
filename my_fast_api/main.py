import cv2
import os
import wandb
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import StreamingResponse
import logging
from app import get_image_from_path_or_bytes, get_model_prediction, get_model_prediction_with_tracking, get_frames_from_video
from ultralytics import YOLO
import supervision as sv

app = FastAPI(
    title='Car Damage Detection API',
    description=''
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
api_key = "2831da270e9cce1dfdee82637fd19c84adae37c8"
os.environ["WANDB_API_KEY"] = api_key
wandb.login(key=api_key)
run = wandb.init()
output_video_path = ''
tracker = sv.ByteTrack()

@app.post('/car_damage_detection')
async def cars_damage_detection(file: bytes = File(...)):
    """
    Endpoint to perform car damage detection on the input image.

    Args:
        file (bytes): The input image file.

    Returns:
        dict: Predicted car damage categories and their severity.
    """
    try:
        artifact = run.use_artifact('ibolarinwa606/model-registry/object detection:v0', type='model')
        artifact_dir = artifact.download()
        model = YOLO(os.path.join(artifact_dir, 'best.pt'))  # Initialize the model after downloading
        
        input_image = get_image_from_path_or_bytes(file)
        predict = get_model_prediction(input_image=input_image, model=model)
        return predict
    except Exception as e:
        logger.exception("An error occurred during car damage detection: %s", str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred during car damage detection")

def save_annotated_video(frames, output_path):
    """
    Save annotated frames to a video file.

    Args:
        frames (List[np.ndarray]): List of annotated frames.
        output_path (str): Path to save the output video file.
    """
    # Get the height and width of the frames
    height, width, _ = frames[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec for your system
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    # Write each annotated frame to the video file
    for frame in frames:
        out.write(frame)

    # Release the VideoWriter object
    out.release()

@app.post('/car_damage_detection_video')
async def cars_damage_detection_video(video_file: UploadFile = File(...)):
    """
    Endpoint to perform car damage detection with object tracking on a video file.

    Args:
        video_file (UploadFile): Video file uploaded by the user.

    Returns:
        FileResponse: Annotated video file with detected objects and labels.
    """
    try:
        artifact = run.use_artifact('ibolarinwa606/model-registry/object detection:v0', type='model')
        artifact_dir = artifact.download()
        model = YOLO(os.path.join(artifact_dir, 'best.pt'))  # Initialize the model after downloading
        
        # Get video frames generator
        frames_generator = sv.get_video_frames_generator(source_path=f'test/{video_file.filename}')
        
        # Get video information
        video_info = sv.VideoInfo.from_video_path(f'test/{video_file.filename}')
        
        # Create a VideoSink to save annotated video
        global output_video_path
        output_video_path = f'output/annotated_{video_file.filename}'
        with sv.VideoSink(target_path=output_video_path, video_info=video_info) as sink:
            # Process each frame and write annotated frames to VideoSink
            for frame in frames_generator:
                annotated_frame = get_model_prediction_with_tracking(model, tracker, frame)
                sink.write_frame(frame=annotated_frame)

    except Exception as e:
        # Handle exceptions
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get('/display_annotated_video')
async def display_annotated_video():
    """
    Endpoint to display the annotated video.

    Returns:
        StreamingResponse: Annotated video file with detected objects and labels.
    """
    global output_video_path  # Use the global keyword to access the variable

    if not output_video_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Annotated video not available")

    def generate():
        with open(output_video_path, 'rb') as video_file:
            while True:
                chunk = video_file.read(1024 * 8)  # Read 8KB chunks of the video file
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(generate(), media_type='video/mp4')

@app.get('/health')
async def health_check():
    """
    Endpoint to check the health status of the API.

    Returns:
        dict: Health status.
    """
    return {'status': 'healthy'}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
