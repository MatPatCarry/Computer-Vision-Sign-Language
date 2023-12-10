import mediapipe as mp
import cv2
import numpy as np
import logging
import sqlite3
import numpy as np
from mediapipe.tasks.python.vision import PoseLandmarkerResult
from mediapipe.tasks.python import vision
from flask import make_response, jsonify

logger = logging.getLogger(__name__)

IMAGE_SIZE = (256, 256)
SEQUENCE_LENGTH = 25
BG_COLOR = (255, 255, 255)

def extract_keypoints(
        results: PoseLandmarkerResult,
        pose_indexes: tuple = (11, 25),
        use_visibility: bool = True):
    
    pose_start_idx, pose_end_idx = pose_indexes
    landmark_num_vals = 4 if use_visibility else 3

    if not results:
        return np.zeros(shape=(pose_end_idx - pose_start_idx, landmark_num_vals))

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
    step = int(max(round(video_length / sequence_length, 0), 1))

    video_frames = []
    timestamps = []
    frame_nr, counter = 1, 1
    success, frame = capture.read()

    while success:
        success, frame = capture.read()

        if success and counter % step == 0: 

            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)
            video_frames.append(frame)
            timestamps.append(timestamp)
            frame_nr += 1

        counter += 1

    return video_frames, timestamps

def load_mp_image(
    img_numpy: np.ndarray,
    img_size: tuple = (256, 256)) -> mp.Image:

    img_numpy = cv2.resize(img_numpy, img_size)
    img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=img_numpy)

def adjust_video_frames(
        video_frames: list[np.ndarray],
        timestamps: list[float],
        sequence_length: int = 25,
        image_size: tuple[int, int] = (256, 256),
        img_channels: int = 3) -> list[np.ndarray]:
    
    video_frames_length = len(video_frames)

    if video_frames_length == sequence_length:
        return video_frames, timestamps
    
    if video_frames_length > sequence_length:

        excess = video_frames_length - sequence_length
        from_beginning, from_ending = begin_and_end(excess)

        video_frames = video_frames[from_beginning: -from_ending if from_ending else None]
        timestamps = timestamps[from_beginning: -from_ending if from_ending else None]

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
        
        timestamp_diff = timestamps[1] - timestamps[0]

        timestamps = [timestamps[0] - timestamp_diff * i for i in range(1, from_beginning + 1)][::-1] \
            + timestamps \
            + [timestamps[-1] + timestamp_diff * i for i in range(1, from_ending + 1)]

    return video_frames, timestamps

def refactor_video_into_input(
        video_path: str,
        sequence_length: int = 25,
        image_size: tuple = (256, 256),
        pose_indexes: tuple = (11, 25),
        use_visibility: bool = True,
        video_task: bool = False,
        detector_options: vision.PoseLandmarkerOptions = None):

    capture = cv2.VideoCapture(video_path)

    video_frames, timestamps = extract_video_frames(
        capture=capture,
        sequence_length=sequence_length
    )

    capture.release()
    
    video_frames, timestamps = adjust_video_frames(
        video_frames=video_frames,
        timestamps=timestamps,
        sequence_length=sequence_length,
        image_size=image_size,
        img_channels=3
    )
        
    keypoints_for_whole_video = []

    if timestamps is not None:
        timestamps = [int(x) for x in timestamps]

    with vision.PoseLandmarker.create_from_options(detector_options) as key_points_detector:

        for i, frame in enumerate(video_frames):

            landmarker_detection = []

            if not np.all(frame == 0):

                landmarker_detection = key_points_detector.detect(

                    load_mp_image(
                        img_numpy=frame,
                        img_size=image_size
                    )

                ) if not video_task else key_points_detector.detect_for_video(
                    load_mp_image(
                        img_numpy=frame,
                        img_size=image_size
                    ),
                    timestamps[i]
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

    steps, n_keypoints, n_cords = keypoints_for_whole_video.shape

    if steps != sequence_length:

        print(f"Steps: {steps} - applying padding")

        shortage = sequence_length - steps
        from_beginning, from_ending = begin_and_end(shortage)

        keypoints_for_whole_video = np.concatenate([
            np.tile(keypoints_for_whole_video[0], (from_beginning, 1)) \
                .reshape(from_beginning, -1, n_cords),
            keypoints_for_whole_video,
        ])

        if from_ending:
            keypoints_for_whole_video = np.concatenate([
                keypoints_for_whole_video,
                np.tile(keypoints_for_whole_video[-1], (from_ending, 1)) \
                    .reshape(from_ending, -1, n_cords)
            ])


    assert keypoints_for_whole_video.shape[0] == sequence_length

    max_x, max_y, max_z = np.max(keypoints_for_whole_video, axis=(0, 1))
    min_x, min_y, min_z = np.min(keypoints_for_whole_video, axis=(0, 1))

    max_x = 1 if max_x < 1 else max_x
    max_y = 1 if max_y < 1 else max_y

    min_x = 0 if min_x > 0 else min_x
    min_y = 0 if min_y > 0 else min_y

    max_z = max(max_z, abs(min_z))
    max_z = 1 if max_z < 1 else max_z

    keypoints_for_whole_video[:, :, 0] = (keypoints_for_whole_video[:, :, 0] - min_x) / (max_x - min_x)
    keypoints_for_whole_video[:, :, 1] = (keypoints_for_whole_video[:, :, 1] - min_y) / (max_y - min_y)

    keypoints_for_whole_video[:, :, 2] = \
        (keypoints_for_whole_video[:, :, 2] + max_z) / (max_z * 2)

    logger.info(f"Length: {len(keypoints_for_whole_video)}")

    return keypoints_for_whole_video

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

def prepare_cors_resp(response_dict: dict):

    resp = make_response(jsonify(response_dict), 200)
    resp.headers['Content-Type'] = 'application/json'
    resp.headers['Access-Control-Allow-Origin'] = '*'

    return resp


            




