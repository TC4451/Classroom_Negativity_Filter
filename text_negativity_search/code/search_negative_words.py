import pandas as pd
import os

negative_word_list = ['excuse me', 'keep your', 'why are you', 'i need you', 'stop', 'be quiet', 'sit down', 'eyes on me', 'can you please', 'can you stop', 'listen', 'attention', 'don’t talk', 'don’t yell', 'on your bottom', 'noise', 'keep the volume']

df_contains_neg_word = pd.DataFrame()
df_no_neg_word = pd.DataFrame()
file_names = []

df = pd.read_csv('../concatenated_csv.csv')
for ind in df.index:
    text = df.iloc[ind]['text']
    if type(text) == str:
        text = text.lower()
        res = any(word in text for word in negative_word_list)
        if res == True:
            df_contains_neg_word = pd.concat([df_contains_neg_word, df.iloc[[ind]]], axis=0)
        else:
            df_no_neg_word = pd.concat([df_no_neg_word, df.iloc[[ind]]], axis=0)

df_contains_neg_word = df_contains_neg_word.iloc[: , 1:]
df_no_neg_word = df_no_neg_word.iloc[: , 1:]
df_contains_neg_word.to_csv('../results/results_with_without_words/with_words.csv')
df_no_neg_word.to_csv('../results/results_with_without_words/without_words.csv')
