import os
import csv
import sys
import numpy as np
import pandas as pd
from google.cloud import language_v1

# google cloud authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/daizi/Desktop/WPI/2021-2022/2022Summer/SpeechRecognitionResearch/sentiment-analysis/sentiment-analysis-354611-42e2be944e29.json"

# used for testing
# txt_file_path = "C:/Users/daizi/Downloads/transcripts/submission-284e809b_video.txt"
# csv_file_path = "C:/Users/daizi/Downloads/all/284e809b.csv"
# output_csv_path = "test.csv"

# given transcripts files and csv containing start and end times, produce emotion predictions
# one video per csv file
txt_file_path = sys.argv[1]
csv_file_path = sys.argv[2]
output_csv_path = sys.argv[3]

start_times = []
end_times = []
sentiment_score = []
sentiment_text = []

def sample_analyze_sentiment(text_content):
    client = language_v1.LanguageServiceClient()

    type_ = language_v1.Document.Type.PLAIN_TEXT

    language = "en"
    document = {"content": text_content, "type_": type_, "language": language}

    encoding_type = language_v1.EncodingType.UTF8

    response = client.analyze_sentiment(request = {'document': document, 'encoding_type': encoding_type})
    # Get overall sentiment of the input document
    sentiment_text.append(text_content)
    sentiment_score.append(response.document_sentiment.score)

# reformat each file, remove the first empty line and append "NaN" for empty lines
def analyze_line_in_file(txt_file_path):
    with open(txt_file_path, encoding="utf-8") as file:
        next(file)
        for line in file:
            if not line == '\n' and not line.strip('\n') == "None" and not line.strip('\n') == ".":
                sample_analyze_sentiment(line)
            else:
                sentiment_text.append("NaN")
                sentiment_score.append("NaN")    

# append the start time and end time to each row
def process_time_list(csv_file_path):
    with open (csv_file_path, 'r') as f:
        csv_file = list(csv.reader(f))
        starting_time = float(csv_file[1][0])
        ending_time = float(csv_file[-1][1])
    for n in np.arange(starting_time, ending_time, 10):
        start_times.append(round(n, 3))
        end_times.append(round(n+10.0, 3))
   
process_time_list(csv_file_path)
analyze_line_in_file(txt_file_path)

start_time_df = pd.DataFrame(start_times, columns=["startTime"])
end_time_df = pd.DataFrame(end_times, columns=["endTime"])
sentiment_score_df = pd.DataFrame(sentiment_score, columns=["score"])
sentiment_text_df = pd.DataFrame(sentiment_text, columns=["text"])
all_info_df = pd.concat([start_time_df, end_time_df, sentiment_score_df, sentiment_text_df], axis=1)
all_info_df.to_csv(output_csv_path)
