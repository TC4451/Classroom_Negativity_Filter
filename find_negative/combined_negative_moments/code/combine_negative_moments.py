import pandas as pd

# path to csv
path_text = "../data/clean_semiauto_neg_moments.csv"
path_all = "../data/all_methods_neg_moments.csv"

df_text = pd.read_csv(path_text)
df_all = pd.read_csv(path_all)

# reformat dataFrames
df_text = df_text[['file_name', "time_in_seconds", "flagged_as_corrective"]]
df_text = df_text.rename(columns={"time_in_seconds" : "start_time"})
df_all = df_all[['file_name', "start_time"]]
df = pd.DataFrame()
df = pd.concat([df_all, df_text], axis=0)


# find the closest second value
def find_closest(df, time):
    dist = (df['startTime'] - time).abs()
    series = df.loc[dist.idxmin()]
    discrepancy = series['startTime'] - time
    if (discrepancy > 1):
        raise Exception("The discrepancy is too large.")
    return series

# get the transcript of each clip
str_list = []
for i in range (0, len(df)):
    fn = df.iloc[i]["file_name"]
    if not fn.endswith((".csv")):
        fn = fn + ".csv"
    path = "../data/per_file_concat_csv/" + fn
    df_f = pd.read_csv(path)
    time = df.iloc[i]["start_time"]
    series = find_closest(df_f, time)
    text = series["text"]
    if (text == 'NaN'):
        str_list.append("None")
    else:
        str_list.append(text)

df['transcript'] = str_list
df.to_csv("../results/negative_with_transcripts.csv")
        
