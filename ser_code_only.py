from traceback import clear_frames
import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow import keras
# import librosa
# import librosa.display
import sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder, normalize
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import soundfile
import pickle
from joblib import dump, load
import statistics
import sys
import matplotlib. pyplot as plt
import seaborn as sns

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# sample rate of each audio
sample_rate = 22050

# model path
model = tf.keras.models.load_model('saved_model')

#testing_path = sys.argv[1]

# Data Augmentation
def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

# # reformat testing data to make them more flexible
# def stretch(data, rate=0.8):
#     return librosa.effects.time_stretch(data, rate)

# def shift(data):
#     shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
#     return np.roll(data, shift_range)

# def pitch(data, sampling_rate, pitch_factor=0.7):
#     return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

# # Feature Extraction
# def extract_features(data):
#     # print(type(data))
#     # ZCR
#     result = np.array([])
#     zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
#     result=np.hstack((result, zcr)) # stacking horizontally

#     # Chroma_stft
#     stft = np.abs(librosa.stft(data))
#     chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
#     result = np.hstack((result, chroma_stft)) # stacking horizontally

#     # MFCC
#     mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
#     result = np.hstack((result, mfcc)) # stacking horizontally

#     # Root Mean Square Value
#     rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
#     result = np.hstack((result, rms)) # stacking horizontally

#     # MelSpectogram
#     mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
#     result = np.hstack((result, mel)) # stacking horizontally
    
#     return result

# def get_features(path):
#     # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
#     data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
#     # without augmentation
#     res1 = extract_features(data)
#     result = np.array(res1)
    
#     # data with noise
#     noise_data = noise(data)
#     res2 = extract_features(noise_data)
#     result = np.vstack((result, res2)) # stacking vertically
    
#     # data with stretching and pitching
#     new_data = stretch(data)
#     data_stretch_pitch = pitch(new_data, sample_rate)
#     res3 = extract_features(data_stretch_pitch)
#     result = np.vstack((result, res3)) # stacking vertically
    
#     return result

# # Testing Feature Extraction
# def get_testing_features(path):
#     data, sample_rate = librosa.load(path, duration = 2.5, offset = 0.6)

#     res1 = extract_features(data)
#     result = np.array(res1)

#     return result

# Data Loading
X= np.load("data_X.npy")
Y= np.load("data_Y.npy")
Features = pd.DataFrame(X)
Features['labels'] = Y

X = Features.iloc[: ,:-1].values
Y = Features['labels'].values

# # count the number of each emotion in the data set
# num_happy = Features['labels'].value_counts()['happy']
# num_calm = Features['labels'].value_counts()['calm']
# num_neutral = Features['labels'].value_counts()['neutral']
# num_disgust = Features['labels'].value_counts()['disgust']
# num_sad = Features['labels'].value_counts()['sad']
# num_angry = Features['labels'].value_counts()['angry']
# num_fearful = Features['labels'].value_counts()['fear']
# num_surprised = Features['labels'].value_counts()['surprise']

# load and fit the encoder
encoder = load('enc.joblib')
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

# split the testing and training data
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)

# print(type(y_test))
# # count the number of each emotion in the data set
# num_happy = y_test['labels'].value_counts()['happy']
# num_calm = y_test['labels'].value_counts()['calm']
# num_neutral = y_test['labels'].value_counts()['neutral']
# num_disgust = y_test['labels'].value_counts()['disgust']
# num_sad = y_test['labels'].value_counts()['sad']
# num_angry = y_test['labels'].value_counts()['angry']
# num_fearful = y_test['labels'].value_counts()['fear']
# num_surprised = y_test['labels'].value_counts()['surprise']

# load and fit the scaler
scaler = load('sca.joblib')
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# # split the audio into 5 second intervals and get its feature
# def split (path):
#     # load file from path
#     data, sample_rate = librosa.load(path)
#     print(sample_rate)
#     # 5 sec interval
#     buffer = 5 * sample_rate

#     # variables to help with the loop
#     samples_total = len(data)
#     samples_wrote = 0
#     counter = 1

#     while samples_wrote < samples_total:

#         #check if the buffer is not exceeding total samples 
#         if buffer > (samples_total - samples_wrote):
#             buffer = samples_total - samples_wrote

#         # each block represents the 5 second interval
#         block = data[samples_wrote : (samples_wrote + buffer)]
#         block = np.float64(block)

#         list_features = []
#         list_features.append(extract_features(block))

#         split_feature = pd.DataFrame(list_features) 
#         features = split_feature.iloc[: ,:].values

#         features = scaler.transform(features)
#         features = np.expand_dims(features, axis=2)

#         # print probabilities of each emotion
#         print(model.predict(features))
#         pred = encoder.inverse_transform(model.predict(features))
#         print(pred.astype(str))
#         print(str((counter-1) * 5) + " sec till " + str((counter) * 5) + " sec" + '\n')

#         counter += 1
#         samples_wrote += buffer

#split(testing_path)

# confusion matrix
pred_test = model.predict(x_test)
y_pred = encoder.inverse_transform(pred_test)
y_test = encoder.inverse_transform(y_test)
df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df['Predicted Labels'] = y_pred.flatten()
df['Actual Labels'] = y_test.flatten()

num_happy = df['Actual Labels'].value_counts()['happy']
num_calm = df['Actual Labels'].value_counts()['calm']
num_neutral = df['Actual Labels'].value_counts()['neutral']
num_disgust = df['Actual Labels'].value_counts()['disgust']
num_sad = df['Actual Labels'].value_counts()['sad']
num_angry = df['Actual Labels'].value_counts()['angry']
num_fearful = df['Actual Labels'].value_counts()['fear']
num_surprised = df['Actual Labels'].value_counts()['surprise']



cm = confusion_matrix(y_test, y_pred)
cm = normalize(cm)
cm = np.round(cm, 4)
# plt.figure(figsize = (12, 10))
# cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
# sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
# plt.title('Confusion Matrix', size=20)
# plt.xlabel('Predicted Labels', size=14)
# plt.ylabel('Actual Labels', size=14)
# plt.show()

cm_ref = np.array([[967,2,117,46,210,30,9,15],
                    [0,122,1,0,3,5,10,1],
                    [73,16,701,82,188,198,188,15],
                    [62,3,80,738,203,89,253,15],
                    [123,7,147,97,898,108,53,17],
                    [6,19,133,60,113,724,208,2],
                    [5,25,102,115,55,166,999,3],
                    [6,4,14,25,34,4,16,392]])
cm_ref = normalize(cm_ref)
cm_ref = np.round(cm_ref, 4)
plt.figure(figsize = (12, 10))
cm_ref = pd.DataFrame(cm_ref , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
sns.heatmap(cm_ref, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.show()

# plt.figure(figsize = (12, 10))
# cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
# sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
# plt.title('Confusion Matrix', size=20)
# plt.xlabel('Predicted Labels', size=14)
# plt.ylabel('Actual Labels', size=14)
# plt.show()

# def audio_sr():
#     self_testing = "C:/Users/daizi/Desktop/WPI/2021-2022/2022Summer/SpeechRecognitionResearch/speech-emotion-recognition/Test_Emotion_Self/"
#     self_testing_directory_list = os.listdir(self_testing)
#     for file in self_testing_directory_list:
#         data, sample_rate = librosa.load(self_testing + file)
#         print(sample_rate)

# def split_from_path(path):
#     directory_list = os.listdir(path)
#     split_file_path = []
#     for file in directory_list:
#         split_file_path.append(path + file)
#     for p in split_file_path:
#         print(p)
#         split(p)

# def emoToBin (list_emotion_str):
#     score = []
#     for e in list_emotion_str:
#         score.append(int(e == "angry" or e == "sad" or e == "fear" or e == "disgust"))
#     return score

# def emoToBin_hpy (list_emotion_str):
#     score = []
#     for e in list_emotion_str:
#         score.append(int(e="happy"))
#     return score

# def emoToBin_float (list_emotion_str):
#     score = []
#     for e in list_emotion_str:
#         score.append(float(e == "angry" or e == "sad" or e == "fear" or e == "disgust"))
#     return score

# def emoToBin_hpy_float(list):
#     score = []
#     for e in list:
#         score.append(float(e=="happy"))
#     return score

# def emoToBin_ref_matrix(ref_list):
#     score = [0, 0, 0, 0, 0, 0, 0, 0]
#     final_list = []
#     for e in ref_list:
#         if e == "angry":
#             score[0] = 1
#         elif e == "calm":
#             score[1] = 1
#         elif e == "disgust":
#             score[2] = 1
#         elif e == "fear":
#             score[3] = 1
#         elif e == "happy":
#             score[4] = 1
#         elif e == "neutral":
#             score[5] = 1
#         elif e == "sad":
#             score[6] = 1
#         else:
#             score[7] = 1
#         final_list.append(score)
#         score = [0, 0, 0, 0, 0, 0, 0, 0]
#     return final_list

# def compare (ref_list, pred_list):
#     result = sklearn.metrics.roc_auc_score(ref_list, pred_list)
#     return result

# def compare_matrix(ref_list, pred_list):
#     result = []
#     for i in range(0, len(ref_list)-1):
#         result.append(compare(ref_list[i], pred_list[i]))
#     return result
    
# def arrToList (list):
#     final_list = []
#     for l in list:
#         l = l.tolist()
#         final_list.append(l)
#     return final_list


# def to_individual_emotion(emotion, emo_list):
#     i = 0
#     if emotion == "angry":
#         i = 0
#     elif emotion == "calm":
#         i = 1
#     elif emotion == "disgust":
#         i = 2
#     elif emotion == "fear":
#         i = 3
#     elif emotion == "happy":
#         i = 4
#     elif emotion == "neutral":
#         i = 5
#     elif emotion == "sad":
#         i = 6
#     else:
#         i = 7
#     single_list = []
#     for l in emo_list:
#         single_list.append(l[i])
#     return single_list


# def to_npfloat (num_list):
#     npfloat_list = []
#     for n in num_list:
#         npfloat_list.append(np.float(n))
#     return npfloat_list


# emo_list = y_pred.astype(str)

# list1 = emoToBin_hpy_float(emo_list)
# print(list1)
# list2 = emoToBin_hpy(file_emotion)
# print(list2)
# result = compare(list2, list1)
# print(result)

# print(emoToBin_ref_matrix(file_emotion))
# ref_list = emoToBin_ref_matrix(file_emotion)
# pred_list = model.predict(testing_X)
# pred_list = arrToList(pred_list)

# single_ref_list = to_individual_emotion("angry",ref_list)
# single_pred_list = to_individual_emotion("angry", pred_list)
# result = compare(single_ref_list, single_pred_list)
# print(result)


##### self testing #####
# ref_list = emoToBin_float(self_file_emotion)
# print(ref_list)
# pred_list = emoToBin(self_list)
# print(pred_list)
# print(compare_hpy(ref_list, pred_list))