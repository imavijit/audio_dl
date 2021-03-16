import numpy as np
import librosa
import tensorflow.keras as keras


NUM_SAMPLES_TO_CONSIDER = 22050  # 1 second worth of sound i.e it loads 22050 samples in a second when librosa is used for loading.
MODEL_PATH = "model.h5"

class _keyword_spotting_service:

    model = None
    _mappings = [
            "bed",
            "bird",
            "cat",
            "dog",
            "down",
            "eight",
            "five",
            "four",
            "go",
            "happy",
            "house",
            "left",
            "marvin",
            "nine",
            "no",
            "off",
            "on",
            "one",
            "right",
            "seven",
            "sheila",
            "six",
            "stop",
            "three",
            "tree",
            "two",
            "up",
            "wow",
            "yes",
            "zero"
        ]

    _instance = None
    def predict(self, file_path):

        #extract MFCCs
        MFCCs =  self.preprocess(file_path)  # (#segments, #coefficients)

        #convert 2d MFCCs array into 4d array -> (#samples,#segments, #coefficients, #channels )
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis, ]

        #make predictions
        predictions = self.model.predict(MFCCs)  # [ [0.1, 0.5, 0.7, ...] ]
        predicted_index  = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword


    def preprocess(self, file_path, n_mfcc = 13, n_fft = 2048, hop_length = 512):
        #load audio file
        signal, sr = librosa.load(file_path)

        #ensure consistency in the audio file length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal  = signal[:NUM_SAMPLES_TO_CONSIDER]

        #extract MFCCs
        MFCCs = librosa.feature.mfcc(signal, n_mfcc = n_mfcc, n_fft = n_fft, hop_length = hop_length )

        return MFCCs.T

def keyword_spotting_service():
    #ensure that we have only 1 instance of keyword_spotting_service
    if _keyword_spotting_service._instance is None:
       _keyword_spotting_service._instance = _keyword_spotting_service()
       _keyword_spotting_service.model = keras.models.load_model(MODEL_PATH)
    return _keyword_spotting_service._instance



if __name__ == "__main__":
    kss = keyword_spotting_service()

    keyword1 = kss.predict("test/bird.wav")
    keyword2 = kss.predict("test/wow.wav")
    print(f"Predicted keywords: {keyword1}, {keyword2}")
