import pandas as pd

df_with_words = pd.read_csv('../results/with_words.csv')
df_without_words = pd.read_csv('../results/without_words.csv')

avg_with_words = df_with_words['score'].mean()
avg_with_words = avg_with_words/2+0.5
avg_without_words = df_without_words['score'].mean()
avg_without_words = avg_without_words/2+0.5

# change text score to 0~1
def process_text_pos_data(x):
    return x/2+0.5

# change text score to opposite
def process_text_neg_data(x):
    return 1-x

def get_composite_col(df):
    df['audioNeg'] = df['audioAngry'] + df['audioDisgust']
    df['posText'] = df['score'].apply(process_text_pos_data)
    df['negText'] = df['posText'].apply(process_text_neg_data)
    df['faceNeg'] = df['faceAngry'] + df['faceDisgust']
    df['serferPos'] = df['audioHappy'] * df['faceHappy']
    df['serferNeg'] = df['audioNeg'] * df['faceNeg']
    df['overallPos'] = df['audioHappy'] * df['faceHappy'] * df['posText']
    df['overallNeg'] = df['audioNeg'] * df['faceNeg'] * df['negText']

get_composite_col(df_with_words)
get_composite_col(df_without_words)

def get_mean(method):
    avg_with_words = df_with_words[method].mean()
    avg_without_words = df_without_words[method].mean()
    print(method)
    print("Average score with words: " + str(avg_with_words))
    print("Average score without words: " + str(avg_without_words))

list_of_methods = ['audioHappy', 'audioNeg', 'posText', 'negText', 'faceHappy', 'faceNeg', 'serferPos', 'serferNeg', 'overallPos', 'overallNeg']
for m in list_of_methods:
    get_mean(m)






