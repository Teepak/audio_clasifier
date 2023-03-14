import os
import librosa
import csv
import numpy as np
import torch

# Define the audio folder path
audio_folder_path = 'C:/Users/pak_a/Documents/introDS/Second course second semester/Machine Learning/recordings'
# Define the CSV file path
csv_file_path = 'C:/Users/pak_a/Documents/introDS/Second course second semester/Machine Learning/audio_classifier-main/features.csv'

# Define the audio feature function
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    feature1 = librosa.feature.zero_crossing_rate(y=y)
    feature2 = librosa.feature.spectral_centroid(y=y)
    feature3 = librosa.feature.spectral_bandwidth(y=y)
    feature4 = librosa.feature.spectral_rolloff(y=y)
    feature5 = librosa.feature.mfcc(y=y, sr=sr)
    return [feature1.mean(), feature2.mean(), feature3.mean(), feature4.mean()] + feature5.mean(axis=1).tolist()


# Extract features for each audio file in the folder and save to CSV
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['zero_crossing_rate', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff'] + ['mfcc{}'.format(i) for i in range(1, 21)])
    for file_name in os.listdir(audio_folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(audio_folder_path, file_name)
            features = extract_features(file_path)
            writer.writerow(features)


