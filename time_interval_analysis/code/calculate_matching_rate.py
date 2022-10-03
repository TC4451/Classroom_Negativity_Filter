import pandas as pd
import os

def calculate_correct(method, positive):
    method_path = "../test_filtered_results/" + method + "_" + positive + ".csv"
    combined_path = "../data/people_emotion_prediction/combined_" + positive + ".csv"
    df = pd.read_csv(method_path)
    df_combined = pd.read_csv(combined_path)
    correct_count = 0
    non_zero_count = 0
    for x in range (0, len(df)):
        fn = df.iloc[x][1]
        if (fn in set(df_combined['file_name']))== True:
            row_num = df_combined[df_combined['file_name'] == fn].index[0]
            value = 0
            if df_combined.iloc[row_num][-1] != 0:
                non_zero_count += 1
                max_time = df.loc[x]['max_time']
                min_time = df.loc[x]['min_time']
                if max_time > min_time:
                    value = 1
                else:
                    value = -1
                if value == df_combined.iloc[row_num][-1]:
                    correct_count += 1
    result = correct_count / non_zero_count
    print(method + " " + positive + " correct rate: " + str(result))
    print("non zero count: " + str(non_zero_count))

dir_list = os.listdir("../test_filtered_results/")
for f in dir_list:
    method = f.split("_")[0]
    positive = (f.split("_")[1]).split(".")[0]
    calculate_correct(method, positive)
