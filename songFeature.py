import numpy as np
import librosa



def load_song(song_path):
    x, sr = librosa.load(song_path)
    print("LOAD COMPLETED")
    return [x, sr]


def beat(song):
    tempo, beats = librosa.beat.beat_track(song[0], song[1])
    return beats[-1] / tempo


def tempo(song):
    tempo, beats = librosa.beat.beat_track(song[0], song[1])
    return tempo


def chroma_stft(song):
    stft = librosa.feature.chroma_stft(song[0], song[1])
    out = 0
    i = 0
    for list in stft:
        for value in list:
            out = out + value
            i = i + 1
    return out / i


def rmse(song):
    rmse = librosa.feature.rms(song[0])
    return np.mean(rmse)


def spectral_centroid(song):
    centroid = librosa.feature.spectral_centroid(song[0], song[1])
    return np.mean(centroid)


def spectral_bandwidth(song):
    bandwidth = librosa.feature.spectral_bandwidth(song[0], song[1])
    return np.mean(bandwidth)


def spectral_rolloff(song):
    rolloff = librosa.feature.spectral_rolloff(song[0], song[1])
    return np.mean(rolloff)


def zero_crossing_rate(song):
    rate = librosa.feature.zero_crossing_rate(song[0], song[1])
    return np.mean(rate)


def mfcc(song):
    return librosa.feature.mfcc(song[0], song[1])


def mfcc(song, index):
    coefficients = librosa.feature.mfcc(song[0], song[1])
    return coefficients[index]


def get_song_feature(song):
    features = np.ndarray(28)
    print("Processing features, please wait a few seconds...")
    features[0] = tempo(song)
    features[1] = beat(song)
    features[2] = chroma_stft(song)
    features[3] = rmse(song)
    features[4] = spectral_centroid(song)
    features[5] = spectral_bandwidth(song)
    features[6] = spectral_rolloff(song)
    features[7] = zero_crossing_rate(song)
    data = librosa.feature.mfcc(song[0], song[1])
    i = 8
    for value in data:
        features[i] = np.mean(value)
        i = i + 1
    return features
