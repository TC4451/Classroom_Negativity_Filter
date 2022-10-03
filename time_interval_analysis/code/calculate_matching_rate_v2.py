import pandas as pd
import os

VALUE_LOCATION = -3
TIME_ONE_LOCATION = -2
TIME_TWO_LOCATION = -1

# change text score to 0~1
def process_text_pos_data(x):
    return x/2+0.5

# change text score to opposite
def process_text_neg_data(x):
    return 1-x

# find the closest second value
def find_closest(df, time, method):
    dist = (df['startTime'] - time).abs()
    series = df.loc[dist.idxmin()]
    discrepancy = series['startTime'] - time
    if (discrepancy > 1):
        raise Exception("The discrepancy is too large.")
    return series

# calculate the correct rate of estimation
def edited_calculate_correct(df_pos, method):
    correct_count = 0
    non_zero_count = 0
    for x in range (0, len(df_pos)):
        value = 0
        # if the final label is not 0, then proceed
        if df_pos.iloc[x][VALUE_LOCATION] != 0:
            non_zero_count += 1
            fn = df_pos.iloc[x][0]
            df = pd.read_csv("../results_corrected/" + fn)
            df['audioNeg'] = df['audioAngry'] + df['audioDisgust']
            df['posText'] = df['score'].apply(process_text_pos_data)
            df['negText'] = df['posText'].apply(process_text_neg_data)
            df['faceNeg'] = df['faceAngry'] + df['faceDisgust']
            df['serferPos'] = df['audioHappy'] * df['faceHappy']
            df['serferNeg'] = df['audioNeg'] * df['faceNeg']
            df['overallPos'] = df['audioHappy'] * df['faceHappy'] * df['posText']
            df['overallNeg'] = df['audioNeg'] * df['faceNeg'] * df['negText']
            time_one = df_pos.iloc[x][TIME_ONE_LOCATION] 
            time_two = df_pos.iloc[x][TIME_TWO_LOCATION]
            series_one = find_closest(df, time_one, method)
            series_two = find_closest(df, time_two, method)
            val_one = series_one[method]
            val_two = series_two[method]
            #if machine predicted value one is larger, set label to be -1
            #   set it to be one otherwise
            if val_one > val_two:
                value = -1
            else:
                value = 1            
            if value == df_pos.iloc[x][VALUE_LOCATION]:
                correct_count += 1
    result = correct_count / non_zero_count
    print(method + " " + " correct rate: " + str(result))
    print("non zero count: " + str(non_zero_count))

df_pos = pd.read_csv("../data/people_emotion_prediction/combined_positive.csv")
df_neg = pd.read_csv("../data/people_emotion_prediction/combined_negative.csv")
pos_methods = ['audioHappy', 'faceHappy', 'posText', 'serferPos', 'overallPos']
neg_methods = ['audioNeg','faceNeg', 'negText','serferNeg', 'overallNeg']
for m in pos_methods:
    edited_calculate_correct(df_pos, m)
for m in neg_methods:
    edited_calculate_correct(df_neg, m)
