import pandas as pd

# path to labeled csv
w_path = "../labeling_data/Whitehill_key_phrase_facial_auditory_labeling.csv"
a_path = "../labeling_data/Andrew_key_phrase_facial_auditory_labeling.csv"
z_path = "../labeling_data/Zilin_key_phrase_facial_auditory_labeling.csv"

df_w = pd.read_csv(w_path);
df_a = pd.read_csv(a_path);
df_z = pd.read_csv(z_path);

df = pd.DataFrame();
df_neg = pd.DataFrame();

#get negative moments
df['file_name'] = df_w['file_name']
df['start_time'] = df_w['time_two']
df['w_semiauto_emotion'] = df_w['overall_emotion_two']
df['a_semiauto_emotion'] = df_a['overall_emotion_two']
df['z_semiauto_emotion'] = df_z['overall_emotion_two']
df['w_corrective'] = df_w['is_two_corrective']
df['a_corrective'] = df_a['is_two_corrective']
df['z_corrective'] = df_z['is_two_corrective']
df['time_in_seconds'] = df_z['time_two_seconds']

w_emotion_columnSeries = df['w_semiauto_emotion']
a_emotion_columnSeries = df['a_semiauto_emotion']
z_emotion_columnSeries = df['z_semiauto_emotion']

w_count = []
a_count = []
z_count = []

def append_emo(val, count_arr):
    if val == 'positive':
        count_arr.append(1)
    elif val == 'negative':
        count_arr.append(-1)
    else:
        count_arr.append(0)

for i in range(0, len(w_emotion_columnSeries)):
    val = w_emotion_columnSeries[i]
    append_emo(val, w_count)
    val = a_emotion_columnSeries[i]
    append_emo(val, a_count)
    val = z_emotion_columnSeries[i]
    append_emo(val, z_count)

df['w_emotion_num'] = w_count
df['a_emotion_num'] = a_count
df['z_emotion_num'] = z_count
df['avg_emotion_num'] = (df['w_emotion_num'] + df['a_emotion_num'] + df['z_emotion_num']) / 3

#get corrective moments
w_corrective_columnSeries = df['w_corrective']
a_corrective_columnSeries = df['a_corrective']
z_corrective_columnSeries = df['z_corrective']

w_count = []
a_count = []
z_count = []
overall_corrective_count = []

for i in range(0, len(w_corrective_columnSeries)):
    count = 0;
    val = w_corrective_columnSeries[i]
    if val == 'yes':
        w_count.append(1)
        count += 1
    else:
        w_count.append(0)
    val = a_corrective_columnSeries[i]
    if val == 'yes':
        a_count.append(1)
        count += 1
    else:
        a_count.append(0)
    val = z_corrective_columnSeries[i]
    if val == 'yes':
        z_count.append(1)
        count += 1
    else:
        z_count.append(0)
    if (count >= 2):
        overall_corrective_count.append(1)
    else:
        overall_corrective_count.append(0)

df['w_corrective_num'] = w_count
df['a_corrective_num'] = a_count
df['z_corrective_num'] = z_count
df['flagged_as_corrective'] = overall_corrective_count

# get negative rows
for i in range(0, len(w_emotion_columnSeries)):
    if (df.iloc[i]['avg_emotion_num'] < 0):
        row_num = i
        df_neg = pd.concat([df_neg, df.iloc[i]], axis=1)
df_neg = df_neg.T

df_neg.to_csv("../results/semiauto_negative_moments.csv")

#get a clean version of dataframe only containing file name, start time, and corrective flag
df_clean = pd.DataFrame();
df_clean['file_name'] = df_neg['file_name']
df_clean['start_time'] = df_neg['start_time']
df_clean['flagged_as_corrective'] = df_neg['flagged_as_corrective']
df_clean['time_in_seconds'] = df_neg['time_in_seconds']
df_clean.to_csv("../results/clean_semiauto_neg_moments.csv")
