import pandas as pd
import numpy as np
import sys
import pdb

# create a csv containing all data from ser, sentiment analysis, and fer
def concatenate_csv(ser_path, text_path, face_path):
    audio_angry = []
    audio_calm = []
    audio_disgust = []
    audio_fear = []
    audio_happy = []
    audio_neutral = []
    audio_sad = []
    audio_surprise = []

    # find start and end time of current file
    df_text = pd.read_csv(ser_path)
    START_TIME = df_text.iloc[0, 0]
    END_TIME = df_text.iloc[-1, 1]

    # convert SER to ten second blocks
    def ser_to_ten_sec (path):
        df = pd.read_csv(path)
        num_lines = len(df)
        find_emotion_values(df, num_lines)

    def find_emotion_values (df, num_lines):
        all_emo_values = df[['happy', 'angry', 'calm', 'disgust','fear','neutral','sad','surprise']]
        for x in range(0, num_lines, 2):
            emo_list = all_emo_values.iloc[x:x+2].mean(axis=0).tolist()
            audio_happy.append(emo_list[0])
            audio_angry.append(emo_list[1])
            audio_calm.append(emo_list[2])
            audio_disgust.append(emo_list[3])
            audio_fear.append(emo_list[4])
            audio_neutral.append(emo_list[5])
            audio_sad.append(emo_list[6])
            audio_surprise.append(emo_list[7])

    ser_to_ten_sec(ser_path)

    audio_angry_df = pd.DataFrame(audio_angry, columns=['audioAngry'])
    audio_calm_df = pd.DataFrame(audio_calm, columns=['audioCalm'])
    audio_disgust_df = pd.DataFrame(audio_disgust, columns=['audioDisgust'])
    audio_fear_df = pd.DataFrame(audio_fear, columns=['audioFear'])
    audio_happy_df = pd.DataFrame(audio_happy, columns=['audioHappy'])
    audio_neutral_df = pd.DataFrame(audio_neutral, columns=['audioNeutral'])
    audio_sad_df = pd.DataFrame(audio_sad, columns=['audioSad'])
    audio_surprise_df = pd.DataFrame(audio_surprise, columns=['audioSurprise'])


    face_angry = []
    face_disgust = []
    face_fear = []
    face_happy = []
    face_neutral = []
    face_sad = []
    face_surprise = []

    # convert face recognition to ten second blocks
    def fer_to_ten_sec (path):
        df = pd.read_csv(path)  
        rowcount = 0
        for x in np.arange(START_TIME, END_TIME-5, 10):
            end = x+10
            counter = 0
            for i in range (0,9):
                if (rowcount+counter) >= len(df):
                    break
                location = df.iloc[rowcount+counter,1]
                time_value = location.split("/")[-1]
                time_value = time_value.split(".")[0]
                time_value = time_value.replace("frame","")
                time_value = START_TIME + int(time_value) - 1
                if time_value < end:
                    counter += 1
                    i += 1
                else:
                    break
            all_emo_values = df[['happy', 'angry', 'disgust','fear','neutral','sad','surprise']]
            emo_list = all_emo_values.iloc[rowcount:rowcount+counter].mean(axis=0).tolist()
            face_happy.append(emo_list[0])
            face_angry.append(emo_list[1])
            face_disgust.append(emo_list[2])
            face_fear.append(emo_list[3])
            face_neutral.append(emo_list[4])
            face_sad.append(emo_list[5])
            face_surprise.append(emo_list[6])
            rowcount += counter

    fer_to_ten_sec(face_path)

    face_angry_df = pd.DataFrame(face_angry, columns=['faceAngry'])
    face_disgust_df = pd.DataFrame(face_disgust, columns=['faceDisgust'])
    face_fear_df = pd.DataFrame(face_fear, columns=['faceFear'])
    face_happy_df = pd.DataFrame(face_happy, columns=['faceHappy'])
    face_neutral_df = pd.DataFrame(face_neutral, columns=['faceNeutral'])
    face_sad_df = pd.DataFrame(face_sad, columns=['faceSad'])
    face_surprise_df = pd.DataFrame(face_surprise, columns=['faceSurprise'])


    # Find the current file name
    def search_for_file_name(audio_path):
        file_name = audio_path.split("/", -1)
        return file_name[-1]

    file_name = search_for_file_name(ser_path)


    # Concatenate all data frames
    df = pd.read_csv(text_path)
    with_audio_df = pd.concat([df, audio_angry_df, audio_calm_df, audio_disgust_df, audio_fear_df, audio_happy_df, audio_neutral_df, audio_sad_df, audio_surprise_df], axis=1)
    with_face_df = pd.concat([with_audio_df, face_angry_df, face_disgust_df, face_fear_df, face_happy_df, face_neutral_df, face_sad_df, face_surprise_df], axis=1)
    with_face_df = with_face_df.iloc[: , 1:]
    with_face_df.to_csv("../results_corrected/" + file_name)

# txt containing all file names
# "../data/file_names.txt"
fn_path = sys.argv[1]
# get all file paths
with open(fn_path, 'r') as f:
    for line in f:
        f_name = line.rstrip()
        print(f_name)
        ser = "../data/all_ser_csv/" + f_name + ".csv"
        text = "../data/all_sentiments_csv/sentiment_" + f_name + ".csv"
        face = "../data/all_faces_csv/" + f_name + ".txt"
        concatenate_csv(ser, text, face)
