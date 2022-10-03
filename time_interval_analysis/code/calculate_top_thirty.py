import pandas as pd
import numpy as np
import statistics
import os

audio_pos_std = []
audio_neg_std = []
text_pos_std = []
text_neg_std = []
face_pos_std = []
face_neg_std = []
audio_and_face_pos_std = []
audio_and_face_neg_std = []
overall_pos_std = []
overall_neg_std = []

# find std of all values in a file
def find_std(path):
    df = pd.read_csv(path)
    audio_positive_std = df['audioHappy'].std()
    df_audio_negative = df['audioAngry'] + df['audioDisgust']
    audio_negative_std = (df_audio_negative).std()
    df_text_positive = df['score'].apply(process_text_pos_data)
    text_positive_std = df_text_positive.std()
    df_text_negative = df_text_positive.apply(process_text_neg_data)
    text_negative_std = df_text_negative.std()
    face_positive_std = df['faceHappy'].std()
    df_face_negative = df['faceAngry'] + df['faceDisgust']
    face_negative_std = (df_face_negative).std()
    audio_face_positive_std = (df['audioHappy'] * df['faceHappy']).std()
    audio_face_negative_std = (df_audio_negative * df_face_negative).std()
    overall_positive_std = (df['audioHappy'] * df_text_positive * df['faceHappy']).std()
    overall_negative_std = (df_audio_negative * df_text_negative * df_face_negative).std()
    return audio_positive_std, audio_negative_std, text_positive_std, text_negative_std, face_positive_std, face_negative_std, audio_face_positive_std, audio_face_negative_std, overall_positive_std, overall_negative_std

# change text score to 0~1
def process_text_pos_data(x):
    return x/2+0.5

# change text score to opposite
def process_text_neg_data(x):
    return 1-x


directory_path = "../results_corrected/"

dir_list = os.listdir(directory_path)
file_name = []
for file in dir_list:
    file_name.append(file)
    path = directory_path + file
    std_list = find_std(path)
    audio_pos_std.append(std_list[0])
    audio_neg_std.append(std_list[1])
    text_pos_std.append(std_list[2])
    text_neg_std.append(std_list[3])
    face_pos_std.append(std_list[4])
    face_neg_std.append(std_list[5])
    audio_and_face_pos_std.append(std_list[6])
    audio_and_face_neg_std.append(std_list[7])
    overall_pos_std.append(std_list[8])
    overall_neg_std.append(std_list[9])

file_name = pd.DataFrame(file_name, columns = ['file_name'])

# find the top thirty file with largest standard deviation
def find_thirty_max_std(list_std): 
    df_std = pd.DataFrame(list_std, columns = ['std'])
    df = pd.concat([file_name, df_std], axis=1)
    df_max_thirty_row = df.nlargest(n=30, columns=['std'])
    max_thirty = df_max_thirty_row['std'].tolist()
    max_thirty_fn = df_max_thirty_row['file_name'].tolist()
    #return type = array
    return max_thirty, max_thirty_fn

# find more information about the individual files
def find_more_info(fn, method):
    max_val = 0
    min_val = 0
    max_time = 0
    min_time = 0
    df = pd.read_csv("../results_corrected/" + fn)
    df['audioNeg'] = df['audioAngry'] + df['audioDisgust']
    df['posText'] = df['score'].apply(process_text_pos_data)
    df['negText'] = df['posText'].apply(process_text_neg_data)
    df['faceNeg'] = df['faceAngry'] + df['faceDisgust']
    df['serferPos'] = df['audioHappy'] * df['faceHappy']
    df['serferNeg'] = df['audioNeg'] * df['faceNeg']
    df['overallPos'] = df['audioHappy'] * df['faceHappy'] * df['posText']
    df['overallNeg'] = df['audioNeg'] * df['faceNeg'] * df['negText']
    max_row = df.nlargest(n=1, columns=[method])
    min_row = df.nsmallest(n=1, columns=[method])
    max_val = max_row.iloc[0][method]
    min_val = min_row.iloc[0][method]
    max_time = max_row.iloc[0]['startTime']
    min_time = min_row.iloc[0]['startTime']
    return max_val, min_val, max_time, min_time

# create a csv for thirty files based on different types of recognition method
# method: 'audio' = speech emotion recognition, 'text' = sentiment analysis, 'face' = face emotion recognition
#           'serfer' = combination of 'ser' and 'fer', 'overall' = all three methods
def find_data(std_list, method, pos):
    max_val_list = []
    min_val_list = []
    max_time_list = []
    min_time_list = []
    max_thirty_std, max_thirty_fn = find_thirty_max_std(std_list)
    df_max_thirty_fn = pd.DataFrame(max_thirty_fn, columns = ['file_name'])
    df_max_thirty_std = pd.DataFrame(max_thirty_std, columns = ['std'])
    for fn in max_thirty_fn:
        max_val, min_val, max_time, min_time = find_more_info(fn, method)
        max_val_list.append(max_val)
        min_val_list.append(min_val)
        max_time_list.append(max_time)
        min_time_list.append(min_time)
    df_max_val = pd.DataFrame(max_val_list, columns = ['max_' + method])
    df_min_val = pd.DataFrame(min_val_list, columns = ['min_' + method])
    df_max_time = pd.DataFrame(max_time_list, columns = ['max_time'])
    df_min_time = pd.DataFrame(min_time_list, columns = ['min_time'])
    df_all = pd.concat([df_max_thirty_fn, df_max_thirty_std, df_max_val, df_min_val, df_max_time, df_min_time], axis=1)
    if pos == 1:
        df_all.to_csv("../test_filtered_results_corrected/" + method + "_positive.csv")
    if pos == 0:
        df_all.to_csv("../test_filtered_results_corrected/" + method + "_negative.csv")

find_data(audio_pos_std, 'audioHappy', 1)
find_data(face_pos_std, 'faceHappy', 1)
find_data(text_pos_std, 'posText', 1)
find_data(text_neg_std, 'negText', 0)
find_data(audio_neg_std, 'audioNeg', 0)
find_data(face_neg_std, 'faceNeg', 0)
find_data(audio_and_face_pos_std, 'serferPos', 1)
find_data(overall_pos_std, 'overallPos', 1)
find_data(audio_and_face_neg_std, 'serferNeg', 0)
find_data(overall_neg_std, 'overallNeg', 0) 
