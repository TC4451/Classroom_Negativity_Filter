import pandas as pd

df_path = "../data/all_methods_combined_negative.csv"

df = pd.read_csv(df_path)

# find the closest second value
def find_closest(df, time):
    dist = (df['startTime'] - time).abs()
    series = df.loc[dist.idxmin()]
    discrepancy = series['startTime'] - time
    if (discrepancy > 1):
        raise Exception("The discrepancy is too large.")
    return series

time_one_ser = []
time_one_fer = []
time_one_text = []
time_two_ser = []
time_two_fer = []
time_two_text = []

for i in range(0, len(df)):
    fn = df.iloc[i]["file_name"]
    if not fn.endswith((".csv")):
        fn = fn + ".csv"
    path = "../data/per_file_concat_csv/" + fn
    df_f = pd.read_csv(path)
    time_one = df.iloc[i]['time_one_sec']
    time_two = df.iloc[i]['time_two_sec']
    series_one = find_closest(df_f, time_one)
    series_two = find_closest(df_f, time_two)
    # for time one
    time_one_ser.append(series_one['audioAngry'] + series_one['audioDisgust'])
    if (series_one['faceAngry'] != 'NaN'):
        time_one_fer.append(series_one['faceAngry'] + series_one['faceDisgust'])
    else:
        time_one_fer.append('NaN')
    # make text score from 0 to 1
    time_one_text.append(1 - (series_one['score']/2 + 0.5))
    # for time two
    time_two_ser.append(series_two['audioAngry'] + series_two['audioDisgust'])
    if (series_two['faceAngry'] != 'NaN'):
        time_two_fer.append(series_two['faceAngry'] + series_two['faceDisgust'])
    else:
        time_two_fer.append('NaN')
    # make text score from 0 to 1
    time_two_text.append(1 - (series_two['score']/2 + 0.5))

df['machine_ser_one'] = time_one_ser
df['machine_ser_two'] = time_two_ser
df['machine_fer_one'] = time_one_fer
df['machine_fer_two'] = time_two_fer
df['machine_text_one'] = time_one_text
df['machine_text_two'] = time_two_text

df.to_csv("../results/all_methods_with_machine.csv")
