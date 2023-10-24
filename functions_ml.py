import mediapipe as mp
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils

IMAGE_SIZE = (256, 256)
SEQUENCE_LENGTH = 25
BG_COLOR = (255, 255, 255)

 # Drawing utilities
def mediapipe_detection(image, model):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

# def draw_landmarks(image, results):

#     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
#     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
#     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
     # Draw right hand connections
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
    
def extract_keypoints(results):

    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, lh, rh])

def begin_and_end(number: int):

    if number == 1:
        return 1, 0
    
    if number % 2 == 0:
        from_beginning, from_ending = number / 2, number / 2
    else:
        from_beginning, from_ending = number // 2 + 1, number // 2

    return int(from_beginning), int(from_ending)

def refactor_video_into_input(attached_video):

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        capture = cv2.VideoCapture(attached_video)
        video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        step = max(round(video_length / SEQUENCE_LENGTH, 0), 1)

        video_frames = []
        success, frame = capture.read()
        frame_nr = 1
        counter = 1

        while success:
            success, frame = capture.read()

            if success and counter % step == 0: 
                
                video_frames.append(frame)
                frame_nr += 1

            counter += 1

        if len(video_frames) != 25:
            
            if len(video_frames) > 25:
                excess = len(video_frames) - 25
                from_beginning, from_ending = begin_and_end(excess)
                video_frames = video_frames[from_beginning:-from_ending]

            else:
                shortage = 25 - len(video_frames)
                black_image = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
                from_beginning, from_ending = begin_and_end(shortage)
                video_frames = [black_image] * from_beginning + video_frames + [black_image] * from_ending

        keypoints_for_whole_video = []

        for frame_index, frame in enumerate(video_frames, start=1):
     
                # frame = remove_background(mp_selfie_segmentation, frame, BG_COLOR)
            resized_frame = cv2.resize(frame, IMAGE_SIZE)
            image, results = mediapipe_detection(resized_frame, holistic)
            # draw_styled_landmarks(image, results)
            keypoints_for_whole_video.append(extract_keypoints(results))
    
        capture.release()

    logger.info(f"Length: {len(keypoints_for_whole_video)}")

    keypoints_for_whole_video = np.asarray(keypoints_for_whole_video).reshape(1, 25, -1)
    # keypoints_for_whole_video = tf.cast(keypoints_for_whole_video, tf.float32)

    logger.info(f"Shape: {keypoints_for_whole_video.shape}")

    return keypoints_for_whole_video


