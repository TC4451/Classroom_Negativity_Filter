import pandas as pd
import numpy as np

# path to labeled csv
w_path = "../labeling_data/Whitehill_key_phrase_facial_auditory_labeling.csv"
a_path = "../labeling_data/Andrew_key_phrase_facial_auditory_labeling.csv"
z_path = "../labeling_data/Zilin_key_phrase_facial_auditory_labeling.csv"

df_w = pd.read_csv(w_path);
df_a = pd.read_csv(a_path);
df_z = pd.read_csv(z_path);

df = pd.DataFrame();

df['w_corrective'] = df_w['is_two_corrective']
df['a_corrective'] = df_a['is_two_corrective']
df['z_corrective'] = df_z['is_two_corrective']

#get the rate of actually corrective moments
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
df['overall_corrective'] = overall_corrective_count

corr = 0
for i in range(0, len(overall_corrective_count)):
    if (overall_corrective_count[i] == 1):
        corr += 1;
print("The rate of clips actually being corrective out of all clips that is filterd to be corrective is: ", corr/50)


#get the rate of negative moments
df['w_emotion'] = df_w['overall_emotion_two']
df['a_emotion'] = df_a['overall_emotion_two']
df['z_emotion'] = df_z['overall_emotion_two']

w_emotion_columnSeries = df['w_emotion']
a_emotion_columnSeries = df['a_emotion']
z_emotion_columnSeries = df['z_emotion']

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

df['w_emotion_num'] = w_count
df['a_emotion_num'] = a_count
df['z_emotion_num'] = z_count
df['avg_emotion_num'] = (df['w_emotion_num'] + df['a_emotion_num'] + df['z_emotion_num']) / 3
avg_emotion_list = df['avg_emotion_num'].tolist()

# get negative clips out of all clips
neg_count = 0
for i in range(0, len(avg_emotion_list)):
    if (avg_emotion_list[i] <0):
        neg_count += 1;
print("The rate of clips being negative out of all clips that is filterd to be corrective is: ", neg_count/50)

# get negative clips out of corrective clips
count = 0
for i in range(0, len(avg_emotion_list)):
    if (overall_corrective_count[i] == 1 and avg_emotion_list[i] <0):
        count += 1;
print("The rate of clips being negative out of all clips that is labeled to be corrective is: ", count/corr)

print(df)

#results:
#The rate of clips actually being corrective out of all clips that is filterd to be corrective is:  0.66
#The rate of clips being negative out of all clips that is filterd to be corrective is:  0.58
#The rate of clips being negative out of all clips that is labeled to be corrective is:  0.7878787878787878
