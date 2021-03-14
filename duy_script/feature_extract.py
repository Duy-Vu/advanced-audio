#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union, MutableMapping, Optional
from pathlib import Path
import pickle
from csv import DictReader
from pandas import read_csv

import numpy as np 
import librosa as lb
from librosa.core import load as lb_load
from librosa.feature import melspectrogram

__author__ = 'Duy Vu'
__docformat__ = 'reStructuredText'
__all__ = [
    'get_audio_file_data',
    'extract_mel_band_energies',
    'serialize_features_and_classes',
    'serialize_features_and_classes',
    'dataset_iteration'
]

SCENE_DICT = {
    "airport": 0,
    "shopping_mall": 1,
    "metro_station": 2,
    "street_pedestrian": 3,
    "public_square": 4,
    "street_traffic": 5,
    "tram": 6,
    "bus": 7,
    "metro": 8,
    "park": 9
}

def get_audio_file_data(audio_file: str) \
        -> np.ndarray:
    """Loads and returns the audio data from the `audio_file`.

    :param audio_file: Path of the `audio_file` audio file.
    :type audio_file: str
    :return: Data of the `audio_file` audio file.
    :rtype: numpy.ndarray
    """
    return lb_load(path=audio_file, sr=None, mono=True, duration=10)[0]

def extract_mel_band_energies(audio_file: str,
                              sr: Optional[int] = 44100,
                              n_fft: Optional[int] = 2048,
                              hop_length: Optional[int] = None,
                              n_mels: Optional[int] = 128) \
        -> np.ndarray:
    """Extracts and returns the log mel-band energies from the `audio_file` audio file.

    :param audio_file: Path of the audio file.
    :type audio_file: str
    :param sr: Sampling frequency of audio file, defaults to 44100.
    :type sr: Optional[int]
    :param n_fft: STFT window length (in samples), defaults to 1024.
    :type n_fft: Optional[int]
    :param hop_length: Hop length (in samples), defaults to 512.
    :type hop_length: Optional[int]
    :param n_mels: Number of MEL frequencies/filters to be used, defaults to 40.
    :type n_mels: Optional[int]
    :return: Mel-band energies of the `audio_file` audio file.
    :rtype: numpy.ndarray
    """
    y = get_audio_file_data(audio_file=audio_file)
    if hop_length == None:
        hop_length = int(n_fft / 2)
    mel_spec = melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, 
                              fmin=0.0, fmax=sr/2, htk=True, norm=None)
    spec_db = lb.power_to_db(mel_spec)
    
    return spec_db

def deltas(fea_in):
    """
    Calculate delta 
    """
    fea_out = (fea_in[:, :, 2:] - fea_in[:, :, :-2])/10.0
    fea_out = fea_out[:, :, 1:-1] + (fea_in[:, :, 4:] - fea_in[:, :, :-4])/5.0
    return fea_out

def serialize_features_and_classes(
        f_name: Path,
        features_and_classes: MutableMapping[str, Union[np.ndarray, int]],
        output_directory: Optional[Union[Path, None]] = None) \
        -> None:
    """Serializes the features and classes.

    :param f_name: File name of the output file.
    :type f_name: Path
    :param features_and_classes: Features and classes.
    :type features_and_classes: dict[str, numpy.ndarray|int]
    :param output_directory: Output directory for the features and classes, defaults to None.
    :type output_directory: Optional[Path|None]
    """
    f_path = f_name if output_directory is None else output_directory.joinpath(f_name)
    with f_path.open('wb') as f:
        pickle.dump(features_and_classes, f, protocol=pickle.HIGHEST_PROTOCOL)

def feature_extration(output_dev_path: str,
                      output_test_path: str,
                      all_data_path: str,
                      test_path: str,
                      parent_dir: str = None) \
        -> None:
    
    df_test = read_csv(parent_dir + test_path, delimiter='\t')
    test_set = set(df_test['filename'])

    # Extract features of each file     ## Using df.apply instead of for loop 
    with open(parent_dir + all_data_path, 'r', encoding='utf8') as data_file:
        csv_reader = DictReader(data_file)
        i = 0
        for row in csv_reader:
            print(i)
            i += 1
            # Get file name
            _, values = list(row.items())[0]
            file_name, scene_label = values.split('\t')[:2]
            audio_file = parent_dir + file_name

            # Extract log-mel features normalize it
            mel_features = extract_mel_band_energies(audio_file=audio_file)
            norm_features = (mel_features - np.min(mel_features)) / (np.max(mel_features) - np.min(mel_features))
            
            # Calculate deltas and delta-deltas on log-mel band 
            features = np.expand_dims(norm_features, axis=0)
            deltas_features = deltas(features)
            deltas_deltas_features = deltas(deltas_features)
            final_features = np.vstack((features[:, :, 4:-4], deltas_features[:, :, 2:-2], deltas_deltas_features))

            # Serialize
            out_file = file_name[6:-4]   # Strip off 2 strings "audio/" and ".wav" at both end of the original file name 
            features_and_classes = {'features': final_features, 'class': SCENE_DICT[scene_label]}
            serialize_features_and_classes(f_name=out_file, 
                                           features_and_classes=features_and_classes, 
                                           output_directory = output_test_path if file_name in test_set else output_dev_path)


if __name__ == "__main__":
    data_path = "/home/fvduvu/dcase2020/baseline/datasets/TAU-urban-acoustic-scenes-2020-mobile-development/"
    feature_extration(
        output_dev_path=Path('log_mel'), 
        output_test_path=Path('testing'),
        all_data_path='meta.csv',
        test_path='evaluation_setup/fold1_test.csv',
        parent_dir=data_path) 

# EOF