import pandas as pd

j_path = "../data/key_phrase_data/whitehill_key_phrase.csv"
a_path = "../data/key_phrase_data/andrew_key_phrase.csv"
z_path = "../data/key_phrase_data/zilin_key_phrase.csv"

df_yes = pd.DataFrame()
df_no = pd.DataFrame()

df_j = pd.read_csv(j_path)
df_j_cp = pd.read_csv(j_path)
df_a = pd.read_csv(a_path)
df_z = pd.read_csv(z_path)

df_j = df_j.iloc[:, :2]
df_j_cp = df_j_cp.iloc[:, :2]
df_a = df_a.iloc[:, :2]
df_z = df_z.iloc[:, :2]
for x in range (0, 30):
    j_val = df_j.iloc[x][1].lower()
    a_val = df_a.iloc[x][1].lower()
    z_val = df_z.iloc[x][1].lower()
    if j_val == a_val and j_val == 'yes':
        df_yes = pd.concat([df_yes, df_j.iloc[[x]]], axis=0)
    elif a_val == z_val and a_val == 'yes':
        df_yes = pd.concat([df_yes, df_a.iloc[[x]]], axis=0)
    elif j_val == a_val and z_val == 'yes':
        df_yes = pd.concat([df_yes, df_z.iloc[[x]]], axis=0)
    else:
        df_j_cp.iloc[x, 1] = 'no'
        df_no = pd.concat([df_no, df_j_cp.iloc[[x]]], axis=0)

def append_rest(df):
    global df_yes, df_no
    for x in range (30, len(df)):
        if df.iloc[x][1].lower() == 'yes':
            df_yes = pd.concat([df_yes, df.iloc[[x]]], axis=0)
        else:
            df_no = pd.concat([df_no, df.iloc[[x]]], axis=0)

append_rest(df_j)
append_rest(df_a)
append_rest(df_z)

df_yes['Is the text either negative or corrective?'] = df_yes['Is the text either negative or corrective?'].str.lower()
df_no['Is the text either negative or corrective?'] = df_no['Is the text either negative or corrective?'].str.lower()

df_yes.to_csv("../key_phrase_encoding_results/concatenated_labeled_csv/yes.csv")
df_no.to_csv("../key_phrase_encoding_results/concatenated_labeled_csv/no.csv")
