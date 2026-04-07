# Author: Dorian Knight
# Created: April 4th 2026
# Updated: April 4th 2026
# Description: Organize training data and save as numpy array to expidate model training

import pandas as pd
import numpy as np
from pathlib import Path
import os
import math


# Model input dimensions
discrete_frequencies = 128
frames_of_audio = 6000

def parse_training_data():
    # Model data information
    print("\nOrganizing data directories and calculating train test split")
    black_capped_donacobius_spectrogram_dir = os.getcwd()+"\Black-capped Donacobius spectrograms"
    brown_crested_flycatcher_spectrogram_dir = os.getcwd()+"\Brown-crested Flycatcher spectrograms"

    black_capped_csvs = list(Path(black_capped_donacobius_spectrogram_dir).glob("*.csv"))
    brown_crested_csvs = list(Path(brown_crested_flycatcher_spectrogram_dir).glob("*.csv"))

    total_black_capped_csvs = len(black_capped_csvs)
    total_brown_crested_csvs = len(brown_crested_csvs)

    train_split = 0.8
    test_split = 1-train_split

    black_capped_training_indices = range(0,math.floor(train_split*total_black_capped_csvs))
    black_capped_testing_indices = range(math.floor(train_split*total_black_capped_csvs), total_black_capped_csvs)

    brown_crested_training_indices = range(0, math.floor(train_split*total_brown_crested_csvs))
    brown_crested_testing_indices = range(math.floor(train_split*total_brown_crested_csvs), total_brown_crested_csvs)

    total_training_indices = len(black_capped_training_indices) + len(brown_crested_training_indices)
    total_testing_indices =  len(black_capped_testing_indices) + len(brown_crested_testing_indices)
    # Create training and testing datasets and labels

    training_data =   []
    training_labels = []
    testing_data =    []
    testing_labels =  []

    # Add black-capped donacobius data to training data array
    print("Adding black-capped donacobius data to training array")
    for i in black_capped_training_indices:
        # print(i)
        datafile_name = black_capped_csvs[i]
        datafile = open(datafile_name)

        data_frame = pd.read_csv(datafile, header=None)
        cropped_spectrogram = data_frame.iloc[:discrete_frequencies, :frames_of_audio].values.tolist()
        data_label = 0 # Black_capped Donacobius

        # Add data to training data array
        training_data.append(cropped_spectrogram)
        training_labels.append(data_label)

    # Add brown-crested flycatcher data to training data array
    print("Adding brown-crested flycatcher data to training array")
    for i in brown_crested_training_indices:
        # print(i)
        datafile_name = brown_crested_csvs[i]
        datafile = open(datafile_name)

        data_frame = pd.read_csv(datafile, header=None)
        cropped_spectrogram = data_frame.iloc[:discrete_frequencies, :frames_of_audio].values.tolist()
        data_label = 1 # Brown-crested flycatcher

        # Add data to training data array
        training_data.append(cropped_spectrogram)
        training_labels.append(data_label)

    # Add black-capped donacobius data to testing data array
    print("Adding black-capped donacobius data to testing array")
    for i in black_capped_testing_indices:
        # print(i)
        datafile_name = black_capped_csvs[i]
        datafile = open(datafile_name)

        data_frame = pd.read_csv(datafile, header=None)
        cropped_spectrogram = data_frame.iloc[:discrete_frequencies, :frames_of_audio].values.tolist()
        data_label = 0 # Black_capped Donacobius

        # Add data to training data array
        testing_data.append(cropped_spectrogram)
        testing_labels.append(data_label)

    # Add brown-crested flycatcher data to testing data array
    print("Adding brown-crested flycatcher data to testing array")
    for i in brown_crested_testing_indices:
        datafile_name = brown_crested_csvs[i]
        datafile = open(datafile_name)

        data_frame = pd.read_csv(datafile, header=None)
        cropped_spectrogram = data_frame.iloc[:discrete_frequencies, :frames_of_audio].values.tolist()
        data_label = 1 # Brown-crested flycatcher

        # Add data to training data array
        testing_data.append(cropped_spectrogram)
        testing_labels.append(data_label)

    # Reformat data arrrays for tensorflow model
    print("Reformatting data\n")
    training_data = np.array(training_data, dtype='float32').reshape(total_training_indices,discrete_frequencies, frames_of_audio)
    training_labels = np.array(training_labels, dtype='uint8').reshape(total_training_indices,)

    testing_data = np.array(testing_data, dtype='float32').reshape(total_testing_indices, discrete_frequencies, frames_of_audio)
    testing_labels = np.array(testing_labels, dtype='uint8').reshape(total_testing_indices,)
    
    return training_data, training_labels, testing_data, testing_labels


def main():
    train_data, train_label, test_data, test_label = parse_training_data()

    # Save formatted data arrays - Commented out for your safety
    # np.save("bird_call_train_data.npy", train_data)
    # np.save("bird_call_train_label.npy", train_label)
    # np.save("bird_call_test_data.npy", test_data)
    # np.save("bird_call_test_label.npy", test_label)
    print("All data saved successfully")


main()
