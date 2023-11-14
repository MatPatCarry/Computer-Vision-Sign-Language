import mediapipe as mp
import cv2
import numpy as np
import logging
import tensorflow as tf
import sqlite3
import numpy as np
from PIL import Image
from IPython.display import display
from mediapipe.tasks.python.vision import PoseLandmarkerResult, PoseLandmarker

logger = logging.getLogger(__name__)

mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils

IMAGE_SIZE = (256, 256)
SEQUENCE_LENGTH = 25
BG_COLOR = (255, 255, 255)

 # Drawing utilities
def mediapipe_detection(
        image: np.array, 
        model: mp_holistic.Holistic):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False   

    results = model.process(image)                 # Make prediction
    image.flags.writeable = True  

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):

    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# def draw_styled_landmarks(image, results):
#     # Draw face connections
#     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
#                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
#                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
#                              ) 
#     # Draw pose connections
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
#                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
#                              ) 
#     # Draw left hand connections
#     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
#                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
#                              ) 
#     # Draw right hand connections  
#     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
#                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
#                              )
    
def extract_keypoints(
        results: PoseLandmarkerResult,
        pose_indexes: tuple = (11, 25),
        use_visibility: bool = True):
    
    pose_start_idx, pose_end_idx = pose_indexes
    landmark_num_vals = 4 if use_visibility else 3

    if not results.pose_landmarks:
        return np.zeros(shape=(pose_end_idx - pose_start_idx, landmark_num_vals))
    
    return np.array(
        [
            [res.x, res.y, res.z, res.visibility] 
            if use_visibility 
            else [res.x, res.y, res.z]
            for res in results.pose_landmarks[0][pose_start_idx: pose_end_idx]
        ]
    )

def begin_and_end(number: int):

    if number == 1:
        return 1, 0
    
    if number % 2 == 0:
        from_beginning, from_ending = number / 2, number / 2
    else:
        from_beginning, from_ending = number // 2 + 1, number // 2

    return int(from_beginning), int(from_ending)

def extract_video_frames(
        capture: cv2.VideoCapture,
        sequence_length: int = 25) -> list[np.ndarray]:

    video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(round(video_length / sequence_length, 0), 1)

    video_frames = []
    frame_nr, counter = 1, 1
    success, frame = capture.read()

    while success:
        success, frame = capture.read()

        if success and counter % step == 0: 
            
            video_frames.append(frame)
            frame_nr += 1

        counter += 1

    return video_frames

def load_mp_image(
    img_numpy: np.ndarray,
    img_size: tuple = (256, 256)) -> mp.Image:

    img_numpy = cv2.resize(img_numpy, img_size)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=img_numpy)

def adjust_video_frames(
        video_frames: list[np.ndarray],
        sequence_length: int = 25,
        image_size: tuple[int, int] = (256, 256),
        img_channels: int = 3) -> list[np.ndarray]:
    
    video_frames_length = len(video_frames)

    if video_frames_length == sequence_length:
        return video_frames
    
    if video_frames_length > sequence_length:

        excess = video_frames_length - sequence_length
        from_beginning, from_ending = begin_and_end(excess)
        video_frames = video_frames[from_beginning:-from_ending]

    else:

        shortage = sequence_length - video_frames_length

        black_image = np.zeros(
            (image_size[0], image_size[1], img_channels), 
            dtype=np.uint8
        )

        from_beginning, from_ending = begin_and_end(shortage)
        video_frames = [black_image] * from_beginning \
            + video_frames \
            + [black_image] * from_ending
        
    return video_frames

def refactor_video_into_input(
        video_path: str,
        key_points_detector: PoseLandmarker,
        sequence_length: int = 25,
        image_size: tuple = (256, 256),
        pose_indexes: tuple = (11, 25),
        use_visibility: bool = True) -> np.ndarray:

    capture = cv2.VideoCapture(video_path)

    video_frames = extract_video_frames(
        capture=capture,
        sequence_length=sequence_length
    )

    capture.release()
    
    video_frames = adjust_video_frames(
        video_frames=video_frames,
        sequence_length=sequence_length,
        image_size=image_size,
        img_channels=3
    )
        
    keypoints_for_whole_video = []

    for frame in video_frames:
    
        landmarker_detection = key_points_detector.detect(
            load_mp_image(
                img_numpy=frame,
                img_size=image_size
            )
        )

        keypoints_for_whole_video.append(
            extract_keypoints(
                results=landmarker_detection,
                pose_indexes=pose_indexes,
                use_visibility=use_visibility
            )
        )

    keypoints_for_whole_video = np.array(
        keypoints_for_whole_video, 
        dtype=np.float16
    )

    keypoint_diff = np.diff(
        a=keypoints_for_whole_video,
        axis=0
    )

    if use_visibility:

        keypoint_diff[:, :, -1] = np.round(
            keypoints_for_whole_video[1:, :, -1], 3
        )

    logger.info(f"Length: {len(keypoints_for_whole_video)}")

    return keypoint_diff

def connect_to_db(
    db: str) -> sqlite3.Connection or None:

    try:
        conn = sqlite3.connect(db)
    except Exception as e:
        logger.error(f'Error when trying to connect to db: {e}')
        return None
    else:
        return conn

def execute_sql(
    query: str,
    db_name: str,
    only_select: bool = True,
    query_params: tuple = None) -> tuple or None:

    conn = connect_to_db(db_name)

    if not isinstance(conn, sqlite3.Connection):

        error_message = f'Error when trying to connect to DB'
        logger.error(error_message)
        return error_message
    
    with conn:

        cur = conn.cursor()

        try:
            cur.execute(query) if query_params is None else cur.execute(query, query_params)
        except Exception as e:

            logger.error(f'Error when trying to execute query: {e}')
            return f'Error when trying to execute query: {e}'
        
        else:

            logger.debug(f'Query: {query} executed successfully')

            if only_select:
                return cur.fetchall()
            
            conn.commit()
            




