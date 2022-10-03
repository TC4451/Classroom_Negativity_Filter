import pandas as pd
import re
import os

df = pd.read_csv('../results/key_phrase_encoding_results/concatenated_labeled_csv/yes.csv')
df_concat = pd.read_csv('../concatenated_csv.csv')
df_concat['un_text'] = df_concat['text'].map(lambda x: re.sub('[^a-zA-Z0-9 \n\.]', '', x))
list_fn = []
list_row_with_text = []
for x in range(0, len(df)):
    text = df.iloc[x][1]
    # removed all special characters from string
    text = re.sub('[^a-zA-Z0-9 \n\.]', '', text)
    row_number = df_concat[df_concat['un_text'] == text].index[0]
    fn = df_concat.loc[row_number]['file_name']
    list_row_with_text.append(row_number)
    list_fn.append(fn)

df_without = pd.read_csv('../results/results_with_without_words/without_words.csv')
list_row_without_text = []
for fn in list_fn:
    row_number = df_without[df_without['file_name'] == fn].index[0]
    row = df_concat.loc[row_number] 
    list_row_without_text.append(row_number)
print(list_fn)
#print(list_row_with_text)
#print(list_row_without_text)

df_result = pd.DataFrame()
for n in range (0, len(list_row_with_text)):
    num_one = list_row_with_text[n]
    num_two = list_row_without_text[n]
    row_with = df_concat.iloc[[num_one]]
    row_wo = df_without.loc[[num_two]]
    df_result = pd.concat([df_result, row_with], axis=0)
    df_result = pd.concat([df_result, row_wo], axis=0)

df_result.to_csv("../csv_for_labeling/with_vs_without_keywords.csv")

