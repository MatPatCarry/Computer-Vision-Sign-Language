from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import mediapipe as mp
from functions_ml import refactor_video_into_input
import os
import numpy as np
import logging
import plotly
import plotly.express as px
import json
import pandas as pd
import yaml

CONFIG_FILE = 'config.yaml'
ALLOWED_EXT = {'mp4', 'webm'}

with open(CONFIG_FILE, 'r') as yaml_conf:
    CONFIG = yaml.load(yaml_conf, Loader=yaml.SafeLoader)

LOGGER_SETTINGS = CONFIG.get('LOGGING')
DIRS = CONFIG.get('DIRS')
FILES = CONFIG.get('FILES')
ML_MODELS = CONFIG.get('ML_MODELS')

logging.basicConfig(
    level=LOGGER_SETTINGS.get('LOGGER_LEVEL'),
    format=LOGGER_SETTINGS.get('LOGGING_FORMAT')
)

logger = logging.getLogger()

logger.debug('Logger initialized successfully')

mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils

logger.debug('Mediapipe models loaded successfully')

SLR_MODEL = tf.keras.models.load_model(
    os.path.join(
        DIRS.get('MODELS_DIR'), 
        ML_MODELS.get('MAIN_SLR')
    )
)

MODELS = {
    "model_keypoints_extraction": SLR_MODEL
}

logger.debug('SLR models loaded successfully')

with open(FILES.get('CLASSES_MAPPING'), 'r', encoding='utf-8') as json_mapping:
    CLASS_MAPPING = json.load(json_mapping)

app = Flask(__name__)


@app.route("/")
def home():

    msg = request.args.get('message', '')

    return render_template("home.html", message=msg)

@app.route("/predict", methods=["POST"])
def predict():

    uploaded_video = request.files.get("file")
    chosen_model = request.form.get('model')

    logger.debug(f'Request args: {request.args}')

    if not uploaded_video:

        message = 'Video not uploaded'
        logger.error(message)

        return redirect(url_for('home', message=message))
    
    video_filename = secure_filename(uploaded_video.filename)

    if video_filename.split('.')[-1] not in ALLOWED_EXT:

        message = 'Invalid extension of loaded file'
        logger.error(message)

        return redirect(url_for('home', message=message))

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
        predictions = model.predict(
            refactor_video_into_input(video_saving_path))[0]
        
    except ValueError:
        return redirect(url_for('error'))
    
    else:
        predictions = [str(round(number, 3)) for number in predictions]
        logger.debug(f"Prediction result for {video_filename}: {predictions}")

    return redirect(
        url_for(
            endpoint='result', 
            predictions=predictions,
            video_filename=video_filename
            )
        )

@app.route("/video")
def result():

    predictions = request.args.getlist('predictions')
    video_filename = request.args.get('video_filename')

    logger.debug(f"{predictions = }, {video_filename = }")

    labels = list(CLASS_MAPPING.values())
    
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
        prediction=CLASS_MAPPING.get(str(class_pred_number))
    )

@app.route('/loaded_videos/<filename>')
def get_video(filename):
    return send_from_directory('loaded_videos', filename)

@app.route("/error")
def error():
    return render_template("error.html")

@app.route('/save_video', methods=['POST'])
def save_video():
    try:
        recorded_video = request.files['recordedVideo']
        if recorded_video:
            save_path = os.path.join('records', 'recorded_video.mp4')
            recorded_video.save(save_path)
            return jsonify({'message': 'Video saved successfully.'})
        else:
            return jsonify({'message': 'No video data received.'})
    except Exception as e:
        return jsonify({'message': 'Error saving video: ' + str(e)})

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == "__main__":
    app.run(debug=True)
