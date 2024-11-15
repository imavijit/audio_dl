import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATA_PATH = "data.json"
SAVED_MODEL_PATH = "model.h5"
LEARNING_RATE= 0.0001
EPOCHS = 50
NUM_KEYWORDS = 10
BATCH_SIZE = 32

def load_dataset(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

        #extract inputs and targets
        X = np.array(data["MFCCs"])
        y = np.array(data["labels"])
        print(X.shape, y.shape)
        return X, y




def get_data_splits(data_path, test_size = 0.2, test_validation = 0.2):

    #load dataset
    X, y = load_dataset(data_path)

    #create train/test/validation splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = test_validation)

    #convert inputs from 2D to 3D arrays
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]

    print(len(X_train), len(y_train))
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_cnn(input_shape,learning_rate=0.0001):

    #build network
    model = keras.Sequential()

    #conv layer 1
    model.add(keras.layers.Conv2D(64, (3, 3), activation = "relu",
                                            input_shape = input_shape,
                                            kernel_regularizer = keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3,3), strides = (2, 2), padding = "same"))

    #conv layer 2
    model.add(keras.layers.Conv2D(32, (3, 3), activation = "relu",
                                            kernel_regularizer = keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3,3), strides = (2, 2), padding = "same"))

    #conv layer 3
    model.add(keras.layers.Conv2D(32, (2, 2), activation = "relu",
                                            kernel_regularizer = keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2,2), strides = (2, 2), padding = "same"))

    #flatten the output and feed it into a dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation = "relu"))
    model.add(keras.layers.Dropout(0.3))

    #softmax classifier
    model.add(keras.layers.Dense(30, activation = "softmax")) #

    #compile the model
    optimiser = keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer = optimiser,loss="sparse_categorical_crossentropy", metrics = ["accuracy"])

    #print model overview
    model.summary()

    return model

def main():

    #load train/test/validation data splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)
    print(len(X_train), len(y_train))  
    #build the CNN model
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_cnn(input_shape,learning_rate=LEARNING_RATE)

   
    #train the model
    model.fit(X_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE, validation_data = (X_validation, y_validation))

    #evaluate the model
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Error: {test_error}, Test Accuracy:{test_accuracy}")

    #save the model
    model.save(SAVED_MODEL_PATH)


if __name__ == "__main__":
	main()
