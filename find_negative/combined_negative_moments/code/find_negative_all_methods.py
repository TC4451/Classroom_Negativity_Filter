import pandas as pd

# read in data csv
path_all_method_pos = "../data/people_emotion_prediction/abs_combined_positive.csv"
path_all_method_neg = "../data/people_emotion_prediction/abs_combined_negative.csv"

df_pos = pd.read_csv(path_all_method_pos)
df_neg = pd.read_csv(path_all_method_neg)

df_pe = pd.DataFrame()
df_ne = pd.DataFrame()
# create a new dataframe for collection negative moments
df_comb_neg = pd.DataFrame()

# convert emotion to number
def emo_to_num(df_e, pos):
    global df_pe, df_ne
    #get the rate of negative moment

    w_emotion_columnSeries = df_e['Jake']
    a_emotion_columnSeries = df_e['Andrew']
    z_emotion_columnSeries = df_e['Zilin']

    w_count = []
    a_count = []
    z_count = []

    for i in range(0, len(w_emotion_columnSeries)):
        val = w_emotion_columnSeries[i]
        if val == 'positive':
            w_count.append(1)
        elif val == 'negative':
            w_count.append(-1)
        else:
            w_count.append(0)
        val = a_emotion_columnSeries[i]
        if val == 'positive':
            a_count.append(1)
        elif val == 'negative':
            a_count.append(-1)
        else:
            a_count.append(0)
        val = z_emotion_columnSeries[i]
        if val == 'positive':
            z_count.append(1)
        elif val == 'negative':
            z_count.append(-1)
        else:
            z_count.append(0)

    if pos == 1:
        df_pe['file_name'] = df_e['file_name']
        df_pe['start_time'] = df_e['time_sec']
        df_pe['w_pos_emotion_num'] = w_count
        df_pe['a_pos_emotion_num'] = a_count
        df_pe['z_pos_emotion_num'] = z_count
        df_pe['avg_emotion_num'] = (df_pe['w_pos_emotion_num'] + df_pe['a_pos_emotion_num'] + df_pe['z_pos_emotion_num']) / 3
    else:
        df_ne['file_name'] = df_e['file_name']
        df_ne['start_time'] = df_e['time_sec']
        df_ne['w_neg_emotion_num'] = w_count
        df_ne['a_neg_emotion_num'] = a_count
        df_ne['z_neg_emotion_num'] = z_count
        df_ne['avg_emotion_num'] = (df_ne['w_neg_emotion_num'] + df_ne['a_neg_emotion_num'] + df_ne['z_neg_emotion_num']) / 3

# 1 and 0 are positive or negative, used for column naming
emo_to_num(df_pos, 1)
emo_to_num(df_neg, 0)

# filter the negative moments based on FinalLabel in each csv
def get_neg(df):
    global df_comb_neg
    for i in range (0, len(df)):
        if df.iloc[i]['avg_emotion_num'] < 0:
            df_comb_neg = pd.concat([df_comb_neg, df.iloc[i]], axis=1)

get_neg(df_pe)
get_neg(df_ne)

df_comb_neg = df_comb_neg.T
df_comb_neg = df_comb_neg.iloc[:,:2]
df_comb_neg.to_csv("../data/all_methods_neg_moments.csv")

