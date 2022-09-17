import numpy as np
import pandas as pd
import glob
import sys
import os
import csv
from csv import reader
from csv import writer
import sklearn
from sklearn.metrics import confusion_matrix, classification_report

# top_happy = []
# top_sad = []
# top_fear = []
# top_disgust = []
# top_angry = []
# top_neutral = []
# top_calm = []
# top_surprise= []

# # add a new column in csv containing the file name
# def add_column_in_csv(input_file, output_file, transform_row):
#     """ Append a column in existing csv using csv.reader / csv.writer classes"""
#     # Open the input_file in read mode and output_file in write mode
#     with open(input_file, 'r') as read_obj, \
#             open(output_file, 'w', newline='') as write_obj:
#         # Create a csv.reader object from the input file object
#         csv_reader = reader(read_obj)
#         # Create a csv.writer object from the output file object
#         csv_writer = writer(write_obj)
#         # Read each row of the input csv file as list
#         for row in csv_reader:
#             # Pass the list / row in the transform function to add column text for this row
#             transform_row(row, csv_reader.line_num)
#             # Write the updated row / list to the output file
#             csv_writer.writerow(row)

# directory_path = "C:/Users/daizi/Downloads/all/"
# directory_list = os.listdir(directory_path)
# file_paths = []
# for file in directory_list:
#     p = directory_path + file
#     header_of_new_col = 'file_name'
#     # Add the column in csv file with header
#     add_column_in_csv(p, directory_path + "_" + file,
#                     lambda row, line_num: row.append(header_of_new_col) if line_num == 1 else row.append(
#                       file))

# # concatenate all csv of emotions into a single csv
# def concat_all(directory_path):
#     files = os.path.join(directory_path, "_*.csv")
#     files = glob.glob(files)
#     df = pd.concat(map(pd.read_csv, files), ignore_index=True)
#     df.to_csv("all.csv")

# concat_all('C:/Users/daizi/Downloads/all/')

# # find the top ten moments of each emotion
# def find_top_ten(path):
#     df = pd.read_csv(path)
#     hpy = df.nlargest(n=10, columns=['surprise'], keep='all')
#     hpy.to_csv("surprise_top_ten.csv")

# find_top_ten('C:/Users/daizi/Desktop/WPI/2021-2022/2022Summer/SpeechRecognitionResearch/data-analysis/all.csv')

# # find the bottom ten moments of each emotion
# def find_bot_ten(path):
#     df = pd.read_csv(path)
#     emo = df.nsmallest(n=10, columns=['disgust'], keep='all')
#     emo.to_csv("disgust_bot_ten.csv")

# find_bot_ten('C:/Users/daizi/Desktop/WPI/2021-2022/2022Summer/SpeechRecognitionResearch/data-analysis/all.csv')

# # find the percentage of positive emotion in all videos
# def percent_positive_emotions(path):
#     pos = 0
#     df = pd.read_csv(path)
#     list_of_emo = df['predictedEmotion'].tolist()
#     for emo in list_of_emo:
#         if emo == "happy" or emo == "calm" or emo == "neutral" or emo == "surprise":
#             pos += 1
#     result = pos/len(list_of_emo)
#     print(result)

# percent_positive_emotions('C:/Users/daizi/Desktop/WPI/2021-2022/2022Summer/SpeechRecognitionResearch/data-analysis/all.csv')

# # find the section with most extreme emotions in a video
# def append_most_emotions(path):
#     df = pd.read_csv(path)
#     happiest=df['happy'].max()
#     top_happy.append(df.loc[df['happy'] == happiest])

#     saddest=df['sad'].max()
#     top_sad.append(df.loc[df['sad'] == saddest])

#     most_angry=df['angry'].max()
#     top_angry.append(df.loc[df['angry'] == most_angry])

#     most_disgust=df['disgust'].max()
#     top_disgust.append(df.loc[df['disgust'] == most_disgust])

#     calmest=df['calm'].max()
#     top_calm.append(df.loc[df['calm'] == calmest])

#     most_surprise=df['surprise'].max()
#     top_surprise.append(df.loc[df['surprise'] == most_surprise])

#     most_neutral=df['neutral'].max()
#     top_neutral.append(df.loc[df['neutral'] == most_neutral])

#     most_afraid=df['fear'].max()
#     top_fear.append(df.loc[df['fear'] == most_afraid])


# avg_happy_list = []
# avg_sad_list = []
# avg_angry_list = []
# avg_disgust_list = []
# avg_fear_list = []
# avg_calm_list = []
# avg_neutral_list = []
# avg_surprise_list = []

# # find the average probability of each emotion in a video
# def find_avg_prob(path):
#     data = pd.read_csv(path)
#     happy = data['happy'].mean()
#     sad = data['sad'].mean()
#     angry = data['angry'].mean()
#     disgust = data['disgust'].mean()
#     fear = data['fear'].mean()
#     calm = data['calm'].mean()
#     neutral = data['neutral'].mean()
#     surprise = data['surprise'].mean()
#     avg_happy_list.append(happy)
#     avg_sad_list.append(sad)
#     avg_angry_list.append(angry)
#     avg_disgust_list.append(disgust)
#     avg_fear_list.append(fear)
#     avg_calm_list.append(calm)
#     avg_neutral_list.append(neutral)
#     avg_surprise_list.append(surprise)


# directory_path = "C:/Users/daizi/Downloads/all/"
# directory_list = os.listdir(directory_path)
# file_name_list = []
# for file in directory_list:
#     p = directory_path + file
#     find_avg_prob(p)
#     file_name_list.append(file)

# file_name_df = pd.DataFrame(file_name_list, columns=['file_name'])
# happy_df = pd.DataFrame(avg_happy_list, columns=['happy'])
# sad_df = pd.DataFrame(avg_sad_list, columns=['sad'])
# angry_df = pd.DataFrame(avg_angry_list, columns=['angry'])
# disgust_df = pd.DataFrame(avg_disgust_list, columns=['disgust'])
# fear_df = pd.DataFrame(avg_fear_list, columns=['fear'])
# calm_df = pd.DataFrame(avg_calm_list, columns=['calm'])
# neutral_df = pd.DataFrame(avg_neutral_list, columns=['neutral'])
# surprise_df = pd.DataFrame(avg_surprise_list, columns=['surprise'])
# all_avg_df = pd.concat([file_name_df, happy_df, sad_df, angry_df, disgust_df, fear_df, calm_df, neutral_df, surprise_df], axis=1)
# all_avg_df.to_csv("all_avg.csv")

# def find_most_top_emotions(path):
#     df = pd.read_csv(path)
#     happiest=df['happy'].max()
#     print("happiest: ")
#     print(df.loc[df['happy'] == happiest])
#     print('\n')

#     saddest=df['sad'].max()
#     print("saddest: ")
#     print(df.loc[df['sad'] == saddest])
#     print('\n')

#     most_angry=df['angry'].max()
#     print("most angry: ")
#     print(df.loc[df['angry'] == most_angry])
#     print('\n')

#     most_disgust=df['disgust'].max()
#     print("most disgusted: ")
#     print(df.loc[df['disgust'] == most_disgust])
#     print('\n')

#     calmest=df['calm'].max()
#     print("calmest: ")
#     print(df.loc[df['calm'] == calmest])
#     print('\n')

#     most_surprise=df['surprise'].max()
#     print("most surprised: ")
#     print(df.loc[df['surprise'] == most_surprise])
#     print('\n')

#     most_neutral=df['neutral'].max()
#     print("most neutral: ")
#     print(df.loc[df['neutral'] == most_neutral])
#     print('\n')

#     most_afraid=df['fear'].max()
#     print("most afraid: ")
#     print(df.loc[df['fear'] == most_afraid])
#     print('\n')

# find_most_top_emotions("all_avg.csv")

# def find_most_bot_emotions(path):
#     df = pd.read_csv(path)
#     happiest=df['happy'].min()
#     print("unhappiest: ")
#     print(df.loc[df['happy'] == happiest])
#     print('\n')

#     saddest=df['sad'].min()
#     print("unsaddest: ")
#     print(df.loc[df['sad'] == saddest])
#     print('\n')

#     most_angry=df['angry'].min()
#     print("most not angry: ")
#     print(df.loc[df['angry'] == most_angry])
#     print('\n')

#     most_disgust=df['disgust'].min()
#     print("most not disgusted: ")
#     print(df.loc[df['disgust'] == most_disgust])
#     print('\n')

#     calmest=df['calm'].min()
#     print("uncalmest: ")
#     print(df.loc[df['calm'] == calmest])
#     print('\n')

#     most_surprise=df['surprise'].min()
#     print("most not surprised: ")
#     print(df.loc[df['surprise'] == most_surprise])
#     print('\n')

#     most_neutral=df['neutral'].min()
#     print("most not neutral: ")
#     print(df.loc[df['neutral'] == most_neutral])
#     print('\n')

#     most_afraid=df['fear'].min()
#     print("most not afraid: ")
#     print(df.loc[df['fear'] == most_afraid])
#     print('\n')

# find_most_bot_emotions("all_avg.csv")

# def find_the_most(file_name, emotion):
#     path = "C:/Users/daizi/Downloads/all/" + file_name
#     df = pd.read_csv(path)
#     most_emotion = df[emotion].max()
#     print(df.loc[df[emotion] == most_emotion])


# find_the_most("eefee714.csv", "happy")

# def find_the_least(file_name, emotion):
#     path = "C:/Users/daizi/Downloads/all/" + file_name
#     df = pd.read_csv(path)
#     least_emotion = df[emotion].min()
#     print(df.loc[df[emotion] == least_emotion])

file_name_list = ["b8fcfcd5.csv", "329c42da.csv", "415ea1be.csv", "b1353c3e.csv", "ca037443.csv", "92999976.csv", "33b80e22.csv", "e8269438.csv", "fe6d85bf.csv", "680f6c90.csv", "2e2a2dca.csv", "f0a2ea8c.csv", "c292b62e.csv", "eefee714.csv"]
happy_list = []
def find_happy_score_list(file_list):
    for file in file_list:
        path = "C:/Users/daizi/Downloads/all/" + file
        df = pd.read_csv(path)
        most_emotion = df["happy"].max()
        # print(most_emotion)
        happy_list.append(most_emotion)
        least_emotion = df["happy"].min()
        # print(least_emotion)
        happy_list.append(least_emotion)

my_hpy_estimation = [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
ad_hpy_estimation = [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0]
find_happy_score_list(file_name_list)

# find auc score of own predicted list vs machine predicted list
def compare (ref_list, pred_list):
    result = sklearn.metrics.roc_auc_score(ref_list, pred_list)
    return result

print(compare(ad_hpy_estimation, happy_list))

# def find_the_five_most (file_name, emotion):
#     path = "C:/Users/daizi/Downloads/all/" + file_name
#     df = pd.read_csv(path)
#     print(df.nlargest(n=5, columns=[emotion], keep='all'))

# find_the_five_most("07854a6c.csv", "happy")

# directory_path = "C:/Users/daizi/Downloads/some/"
# directory_list = os.listdir(directory_path)
# file_paths = []
# for file in directory_list:
#     p = directory_path + file
#     print(p)
#     file_paths.append(p)
#     print(find_most_emotions(p))
#     print('\n')


