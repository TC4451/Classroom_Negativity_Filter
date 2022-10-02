import sys

import cv2
from keras.models import load_model
import numpy as np
import pandas as pd
import os
import pdb

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input

# variables to create csv
file_names = []
faceX = []
faceY = []
face_width = []
face_height = []
angry_prob = []
disgust_prob = []
fear_prob = []
happy_prob = []
sad_prob = []
surprise_prob = []
neutral_prob = []

# parameters for loading data and images
#image_path = sys.argv[1]
# Appending file name
#file_names.append(image_path.split('/')[-1])

detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
gender_offsets = (30, 60)
gender_offsets = (10, 10)
emotion_offsets = (20, 40)
emotion_offsets = (0, 0)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]

def detect_emotion (image_path):
    # loading images
    rgb_image = load_image(image_path, color_mode='rgb')
    gray_image = load_image(image_path, color_mode='grayscale')
    gray_image = np.squeeze(gray_image)
    gray_image = gray_image.astype('uint8')

    faces = detect_faces(face_detection, gray_image)
    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
        rgb_face = rgb_image[y1:y2, x1:x2]

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        #print("(x1, x2, y1, y2): ", x1, x2, y1, y2)
        gray_face = gray_image[y1:y2, x1:x2]

        try:
            rgb_face = cv2.resize(rgb_face, (gender_target_size))
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        # After we know that the processing has succeeded, we can safely add the data to the parallel arrays
        file_names.append(fn)
        faceX.append(x1)
        faceY.append(y1)
        face_width.append(x2-x1)
        face_height.append(y2-y1)

        rgb_face = preprocess_input(rgb_face, False)
        rgb_face = np.expand_dims(rgb_face, 0)
        gender_prediction = gender_classifier.predict(rgb_face)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        #print(emotion_classifier.predict(gray_face))
        prob_list = emotion_classifier.predict(gray_face)[0]
        angry_prob.append(prob_list[0])
        disgust_prob.append(prob_list[1])
        fear_prob.append(prob_list[2])
        happy_prob.append(prob_list[3])
        sad_prob.append(prob_list[4])
        surprise_prob.append(prob_list[5])
        neutral_prob.append(prob_list[6])
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion_text = emotion_labels[emotion_label_arg]

        if gender_text == gender_labels[0]:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, gender_text, color, 0, -20, 1, 2)
        draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -50, 1, 2)

    #bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    #cv2.imwrite('../images/predicted_test_image.png', bgr_image)

# parameters for loading path
fn = ""
for file in sys.stdin:
    fn = file.split('/')[-1]
    path = file.split('\n')[0]
    detect_emotion(path)

df_file_names = pd.DataFrame(file_names, columns=['file_name'])
df_faceX = pd.DataFrame(faceX, columns = ['faceX'])
df_faceY = pd.DataFrame(faceY, columns = ['faceY'])
df_face_width = pd.DataFrame(face_width, columns = ['face_width'])
df_face_height = pd.DataFrame(face_height, columns = ['face_height'])
df_angry_prob = pd.DataFrame(angry_prob, columns = ['angry'])
df_disgust_prob = pd.DataFrame(disgust_prob, columns = ['disgust'])
df_fear_prob = pd.DataFrame(fear_prob, columns = ['fear'])
df_happy_prob = pd.DataFrame(happy_prob, columns = ['happy'])
df_sad_prob = pd.DataFrame(sad_prob, columns = ['sad'])
df_surprise_prob = pd.DataFrame(surprise_prob, columns = ['surprise'])
df_neutral_prob = pd.DataFrame(neutral_prob, columns = ['neutral'])
#df_angry_prob['angry'].replace('', np.nan, inplace=True)
all_df = pd.concat([df_file_names, df_faceX, df_faceY, df_face_width, df_face_height, df_angry_prob, df_disgust_prob, df_fear_prob, df_happy_prob, df_sad_prob, df_surprise_prob, df_neutral_prob], axis=1)
#all_df.dropna(subset=['angry'], inplace=True)
output_path = sys.argv[1]
all_df.to_csv(output_path)

