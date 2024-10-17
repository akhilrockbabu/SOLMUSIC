from flask import Flask, request, render_template, flash, redirect, send_from_directory, url_for
import joblib
import librosa
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'  # Required for flashing messages

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Helper function to convert audio to WAV if needed
import os
from pydub import AudioSegment
def convert_to_wav(file_path):
    import subprocess
    import os.path
    extension = os.path.splitext(file_path)[0]
    subprocess.call(['ffmpeg', '-i', file_path, extension+'.wav'])
    from pydub import AudioSegment
    t1 = 60 * 1000  # Works in milliseconds
    t2 = 90 * 1000
    newAudio = AudioSegment.from_wav(extension+'.wav')
    newAudio = newAudio[t1:t2]
    newAudio.export(extension+'.wav', format="wav")  # Exports to a wav file in the current path.
    audio=extension+'.wav'
    return audio





# Helper function to extract audio features
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path)
    y, _ = librosa.effects.trim(y)

    segment_length = 3
    samples_per_segment = int(segment_length * sr)
    features_list = []

    for start in range(0, len(y), samples_per_segment):
        end = start + samples_per_segment
        if end > len(y):
            break

        y_segment = y[start:end]

        features_dict = {
            'chroma_stft_mean': np.mean(librosa.feature.chroma_stft(y=y_segment, sr=sr)),
            'chroma_stft_var': np.var(librosa.feature.chroma_stft(y=y_segment, sr=sr)),
            'rms_mean': np.mean(librosa.feature.rms(y=y_segment)),
            'rms_var': np.var(librosa.feature.rms(y=y_segment)),
            'spectral_centroid_mean': np.mean(librosa.feature.spectral_centroid(y=y_segment, sr=sr)),
            'spectral_centroid_var': np.var(librosa.feature.spectral_centroid(y=y_segment, sr=sr)),
            'spectral_bandwidth_mean': np.mean(librosa.feature.spectral_bandwidth(y=y_segment, sr=sr)),
            'spectral_bandwidth_var': np.var(librosa.feature.spectral_bandwidth(y=y_segment, sr=sr)),
            'rolloff_mean': np.mean(librosa.feature.spectral_rolloff(y=y_segment, sr=sr)),
            'rolloff_var': np.var(librosa.feature.spectral_rolloff(y=y_segment, sr=sr)),
            'zero_crossing_rate_mean': np.mean(librosa.feature.zero_crossing_rate(y=y_segment)),
            'zero_crossing_rate_var': np.var(librosa.feature.zero_crossing_rate(y=y_segment)),
            'harmony_mean': np.mean(librosa.effects.harmonic(y_segment)),
            'harmony_var': np.var(librosa.effects.harmonic(y_segment)),
            'perceptr_mean': np.mean(librosa.feature.spectral_flatness(y=y_segment)),
            'perceptr_var': np.var(librosa.feature.spectral_flatness(y=y_segment)),
            'tempo': librosa.beat.tempo(y=y_segment, sr=sr)[0]
        }

        # Extract MFCC features (10 coefficients)
        mfccs = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=10)
        for i in range(10):
            features_dict[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
            features_dict[f'mfcc{i+1}_var'] = np.var(mfccs[i])

        features_list.append(features_dict)

    return pd.DataFrame(features_list)

# Function to normalize new data based on min/max values
def normalize_new_data(new_data, X_min, X_max):
    new_data_array = np.array(new_data)

    if new_data_array.shape[1] != len(X_min):
        raise ValueError("New data feature size does not match training data feature size.")

    normalized_data = (new_data_array - X_min) / (X_max - X_min)
    return normalized_data

@app.route('/')
def index():
    return render_template('input_form.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_audio():
    if request.method == 'POST':
        file = request.files['audio']
        if file:
            file_extension = os.path.splitext(file.filename)[1].lower()
            if file_extension != '.wav':
                flash('Only WAV files are allowed! <a href="https://cloudconvert.com/mp3-to-wav" target="_blank">Click here</a> to convert your audio file.', 'error')
                return redirect(url_for('index'))

            # Save the file since it's a .wav file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Extract audio features
            features_df = extract_audio_features(file_path)

            # Load the X_min and X_max datasets (assuming they are in CSV format)
            X_min_df = pd.read_csv('X_min_df')  # Update the path accordingly
            X_max_df = pd.read_csv('X_max_df')  # Update the path accordingly

            # Extract the actual min and max values from the dataframe
            X_min = X_min_df.values[0]  # Assuming the values are stored in the first row
            X_max = X_max_df.values[0]  # Assuming the values are stored in the first row

            mean_values = features_df.mean()
            max_values = features_df.max()
            min_values = features_df.min()

            mean_df = pd.DataFrame([mean_values])
            max_df = pd.DataFrame([max_values])
            min_df = pd.DataFrame([min_values])

            normalized_mean_df = pd.DataFrame(normalize_new_data(mean_df, X_min, X_max), columns=mean_df.columns)
            normalized_max_df = pd.DataFrame(normalize_new_data(max_df, X_min, X_max), columns=max_df.columns)
            normalized_min_df = pd.DataFrame(normalize_new_data(min_df, X_min, X_max), columns=min_df.columns)

            # Load the models
            with open('model1.pkl', 'rb') as f:
                model1 = joblib.load(f)
            with open('model2.pkl', 'rb') as f:
                model2 = joblib.load(f)

            # Predict genre using both models
            genre_min_1 = model1.predict(normalized_min_df)[0]
            genre_mean_1 = model1.predict(normalized_mean_df)[0]
            genre_max_1 = model1.predict(normalized_max_df)[0]

            genre_min_2 = model2.predict(normalized_min_df)[0]
            genre_mean_2 = model2.predict(normalized_mean_df)[0]
            genre_max_2 = model2.predict(normalized_max_df)[0]

            result = [genre_min_1, genre_mean_1, genre_max_1, genre_min_2, genre_mean_2, genre_max_2]
            temp = {}
            for i in result:
                temp[i] = temp.get(i, 0) + 1

            final_result = max(temp, key=lambda k: temp[k])
            audio_file_name = file.filename

            return render_template('result.html', res=final_result, audio_file_name=audio_file_name)
    
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
