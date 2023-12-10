from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from src.utils import refactor_video_into_input, execute_sql, prepare_cors_resp
import os
import numpy as np
import logging
import plotly
import plotly.express as px
import json
import yaml
from datetime import datetime
import pytz
from flask_cors import CORS

CONFIG_DIR = 'config'
CONFIG_FILE = 'config.yaml'
ALLOWED_EXT = {'mp4', 'webm'}
VIDEO_TIMESTAMP_FORMAT = "%Y_%m_%d_%H_%M_%S"
TIME_ZONE = pytz.timezone('Europe/Warsaw')
SEQ_LENGTH = 15
POSE_INDEXES = (11, 23)
IMAGE_SIZE = (256, 256)

with open(os.path.join(CONFIG_DIR, CONFIG_FILE), 'r') as yaml_conf:
    CONFIG = yaml.load(yaml_conf, Loader=yaml.SafeLoader)

LOGGER_SETTINGS = CONFIG.get('LOGGING')
DIRS = CONFIG.get('DIRS')
FILES = CONFIG.get('FILES')
ML_MODELS = CONFIG.get('ML_MODELS')
DB_SETTINGS = CONFIG.get('DB')

logging.basicConfig(
    level=LOGGER_SETTINGS.get('LOGGER_LEVEL'),
    format=LOGGER_SETTINGS.get('LOGGING_FORMAT')
)

logger = logging.getLogger()
logger.debug('Logger initialized successfully')

DETECTOR_PATH = os.path.join(
    DIRS.get('MODELS_DIR'), 
    DIRS.get('KP_DETECTORS_DIR'),
    ML_MODELS.get('KP_DETECTOR')
)

with open(DETECTOR_PATH, 'rb') as det_bytes_model:
    model_data = det_bytes_model.read()

base_options = python.BaseOptions(model_asset_buffer=model_data)
running_mode = vision.RunningMode

detector_options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    running_mode=running_mode.VIDEO)

logger.debug('Mediapipe models loaded successfully')

SLR_MODEL = tf.keras.models.load_model(
    os.path.join(
        DIRS.get('MODELS_DIR'), 
        DIRS.get('CLASSIFIERS_DIR'),
        ML_MODELS.get('MAIN_SLR')
    ), compile=False
)

MODELS = {
    "Keypoints_extraction_LSTM_model": SLR_MODEL
}

logger.debug('SLR models loaded successfully')

with open(os.path.join(CONFIG_DIR, FILES.get('CLASSES_MAPPING')), 'r', encoding='utf-8') as json_mapping:
    CLASS_MAPPING = json.load(json_mapping)

app = Flask(__name__)
CORS(app, resources={
    r"/get_prediction":{
        "origins":"*"
    }
})

@app.route("/home")
def home():

    msg = request.args.get('message', None)
    return render_template("home.html", message=msg, models=MODELS.keys())

@app.route("/predict", methods=["POST"])
def predict():

    try:

        uploaded_video = request.files.get('file')
        chosen_model = request.form.get('model')
        save_video = request.form.get('save_video')

        logger.debug(f'Request args: {request.args}')

        try:
            save_video = int(save_video)
        except ValueError:

            message = 'Invalid form argument'
            logger.error(message)
            return redirect(url_for('home', message=message))                              

        if not uploaded_video:

            message = 'Video not uploaded'
            logger.error(message)
            return redirect(url_for('home', message=message))
        
        video_filename = secure_filename(uploaded_video.filename)
        video_timestamp = datetime.now(TIME_ZONE).strftime(VIDEO_TIMESTAMP_FORMAT)

        video_ext = video_filename.split('.')[-1]

        if video_ext not in ALLOWED_EXT:

            message = 'Invalid extension of loaded file'
            logger.error(message)
            return redirect(url_for('home', message=message))
        
        video_filename = f"{video_timestamp}__SLR.{video_ext}"
        video_saving_path = os.path.join(
            DIRS.get('LOADED_VIDEOS'), 
            video_filename
        )

        try:
            uploaded_video.save(video_saving_path)

        except OSError:

            message = 'Can not successfully save uploaded file'
            logger.error(message)
            return redirect(url_for('home', message=message))
        
        model = MODELS.get(chosen_model)

        if model is None:

            message = f'Chosen model {chosen_model} can not be loaded and properly used'
            logger.error(message)
            return redirect(url_for('home', message=message))

        logger.debug(f"Using model: {chosen_model} for SLR")

        try:

            keypoints = refactor_video_into_input(
                video_path=video_saving_path, 
                sequence_length=SEQ_LENGTH,
                image_size=IMAGE_SIZE,
                pose_indexes=POSE_INDEXES,
                use_visibility=False,
                video_task=True,
                detector_options=detector_options
            )

            predictions = model(tf.expand_dims(keypoints.reshape(SEQ_LENGTH, -1), 0))[0].numpy()

        except ValueError:
            return redirect(url_for('error'))
        
        else:

            if save_video:

                sql_query = f"""
                    INSERT INTO {DB_SETTINGS.get('VIDEOS_TABLE', 'sign_videos')} (name, video_array)
                    VALUES (?, ?)
                """

                query_res = execute_sql(
                    query=sql_query,
                    db_name=DB_SETTINGS.get('DB_NAME'),
                    only_select=False,
                    query_params=(video_filename, keypoints.tobytes())
                )

                if isinstance(query_res, str):

                    message = f'Can not save needed data in database'
                    logger.error(message)

            predictions = [str(round(number, 3)) for number in predictions]
            logger.debug(f"Prediction result for {video_filename}: {predictions}")

        return redirect(
            url_for(
                endpoint='result', 
                predictions=predictions,
                video_filename=video_filename
                )
            )
    
    except Exception as e:

        logger.error(f"Error occured: {e}")
        return redirect(url_for('error'))

@app.route("/result")
def result():

    predictions = request.args.getlist('predictions')
    video_filename = request.args.get('video_filename')
    logger.debug(f"{predictions = }, {video_filename = }")

    labels = list(CLASS_MAPPING.values())[:-1]
    predictions = np.array([float(pred) for pred in predictions])
    class_pred_number = np.argmax(predictions)

    logger.info(predictions)
    
    fig = px.bar(
        x=labels, 
        y=predictions, 
        labels={"x": "Class", "y": "Probability"}
    )

    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        "prediction.html", 
        figure=fig_json, 
        video_filename=video_filename,
        classes=CLASS_MAPPING, 
        prediction=CLASS_MAPPING.get(str(class_pred_number))
    )

@app.route('/loaded_videos/<filename>')
def get_video(filename):
    return send_from_directory('loaded_videos', filename)

@app.route("/error")
def error():
    return render_template("error.html")

@app.route('/submit_prediction', methods=['POST'])
def submit_prediction():

    selected_prediction = request.form.get('prediction')
    video_filename = request.form.get('video_filename')

    logger.debug(f"{selected_prediction = }, {video_filename = }")

    sql_query = f"""
        UPDATE {DB_SETTINGS.get('VIDEOS_TABLE', 'sign_videos')}
        SET verified = ?, true_label = ?
        WHERE name = ?
    """

    query_res = execute_sql(
        query=sql_query,
        db_name=DB_SETTINGS.get('DB_NAME'),
        only_select=False,
        query_params=(1, selected_prediction, video_filename)
    )

    if isinstance(query_res, str):

        message = f'Can not save needed data in database'
        logger.error(message)
        return jsonify({'message': message}), 500

    return redirect(url_for('home', message='Prediction submitted successfully'))

@app.route('/get_prediction', methods=['POST'])
def get_prediction():

    uploaded_video = request.files.get('file')

    if not uploaded_video:

        message = 'Video not uploaded'
        logger.error(message)
        return jsonify({'message': message}), 500
    
    logger.debug('Checking if file passed')
    
    video_filename = secure_filename(uploaded_video.filename)
    video_timestamp = datetime.now(TIME_ZONE).strftime(VIDEO_TIMESTAMP_FORMAT)
    video_ext = video_filename.split('.')[-1]

    if video_ext not in ALLOWED_EXT:

        message = 'Invalid extension of loaded file'
        logger.error(message)
        return jsonify({'message': message}), 500
    
    video_filename = f"{video_timestamp}__SLR.{video_ext}"
    video_saving_path = os.path.join(
        DIRS.get('LOADED_VIDEOS'), 
        video_filename
    )

    try:
        uploaded_video.save(video_saving_path)

    except OSError:

        message = 'Can not successfully save uploaded file'
        logger.error(message)

        return jsonify({'message': message}), 500
    
    try:

        keypoints = refactor_video_into_input(
            video_path=video_saving_path, 
            sequence_length=SEQ_LENGTH,
            image_size=IMAGE_SIZE,
            pose_indexes=POSE_INDEXES,
            use_visibility=False,
            video_task=True,
            detector_options=detector_options
        ).reshape(SEQ_LENGTH, -1)

        predictions = SLR_MODEL(tf.expand_dims(keypoints, 0))[0].numpy()

    except ValueError:
        return jsonify({'message': 'Can not get proper prediction'}), 500
    
    logger.debug(f"Prediction result for {video_filename}: {predictions}")

    class_pred_number = np.argmax(predictions)
    pred_class = CLASS_MAPPING.get(str(class_pred_number))

    resp_dict = {'prob': f"{np.max(predictions):.3f}", 'prediction': pred_class}
    logger.debug(f"Returning dict: {resp_dict = }")

    return prepare_cors_resp(resp_dict=resp_dict)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
