"""
Steps: 1. Extract audio feature(MFCC) for each of the audio sample of the  dataset .
       2. store it in a json file.
       3.

# MFCC(Mel-frequency cepstral coefficients): It is used for audio classification.They are derived from a type of cepstral representation
# of the audio clip (a nonlinear "spectrum-of-a-spectrum").Extracting MFCC from audio gives snapshots of diferent segment of an audio file.The snapshots
# are MFCC coffefficient.

# Librosa: Audio and music processing library in Python.Used for extracting MFCC.
"""

import os
import json
import librosa

data_path  = "dataset"
json_path  = "data.json"
SAMPLES_TO_CONSIDER = 22050 # 1 second worth of sound i.e it loads 22050 samples in a second when librosa is used for loading.


def prepare_dataset(data_path, json_path, n_mfcc = 13, hop_length = 512, n_fft  = 2048):
    # n_mfcc : number of coffefficients to extract
    # hop_length : Extraction of MFCC divides the audio signal into segments of equal length.Then snapshot is taken with MFCC at each of the segments.
    #              Hop_length tells the size of segments in respect to number of frames.

    # n_fft(fast-fourier-transform): Size of the window for the fast-fourier-transformation.


    #data dictionary
    data  = {
           "mappings": [],
           "labels" : [],
           "MFCCs": [],
           "files":[]

    }

    #loop through the audio-folders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(data_path)):

        #check if in root level
        if dirpath is not data_path:

            #update mappings
            category = dirpath.split("/")[-1]   #dataset/bed -> [dataset, bed] ("-1" indexing returns the folder name , e.g: here -> "bed")
            data["mappings"].append(category)
            print("\nProcessing: '{}'".format(category))

            #Extract MFCCs by looping through all the filenames
            for f in filenames:

                #get filepath
                file_path = os.path.join(dirpath, f)

                #load the audio-file
                signal, sample_rate = librosa.load(file_path)

                #ensure the audio file is atleast 1 sec & ignore the file otherwise
                if len(signal) >= SAMPLES_TO_CONSIDER:

                    #make sure sinnal length is consistent
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    #Extract MFCCs & store data
                    MFCCs = librosa.feature.mfcc(signal,sample_rate, n_mfcc = n_mfcc, hop_length = hop_length, n_fft  = n_fft)

                    data["labels"].append(i-1)
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["files"].append(file_path)
                    print("{}: {}".format(file_path, i-1))

        #save data in json file
        with open(json_path, "w") as fp:
            json.dump(data, fp, indent = 4)

if __name__ == "__main__":
    prepare_dataset(data_path, json_path)
