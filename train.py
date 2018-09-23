# for mnist data
import tensorflow as tf 
import argparse
# keras setup
from keras.datasets import mnist
from keras.models import Model 
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout, Input, concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model, to_categorical
from keras.losses import categorical_crossentropy

# from keras.callbacks import ModelCheckpoint

def cnn(dropout_prob_cl, dropout_prob_fc):
    """
    Creates and returns the CNN model
    :param dropout_prob_cl: dropout probability for convolutional layers
    :param dropout_prob_fc: dropout probability for fully connected layers
    :return: keras CNN model
    """
    inputs = Input(shape = (28, 28, 1))
    preprocess = Lambda(lambda x: x / 255.0 - 0.5)(inputs)

    conv1 = Conv2D(32, (3, 3), padding = 'same')(preprocess)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Dropout(dropout_prob_cl)(conv1)
    conv1 = MaxPooling2D((2, 2))(conv1)

    conv2_1 = Conv2D(64, (3, 3), padding = 'same')(conv1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_1 = Activation('relu')(conv2_1)
    conv2_1 = Dropout(dropout_prob_cl)(conv2_1)
    conv2_1 = MaxPooling2D((2, 2))(conv2_1)

    conv2_2 = Conv2D(64, (3, 3), padding = 'same')(conv1)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_2 = Activation('relu')(conv2_2)
    conv2_2 = Dropout(dropout_prob_cl)(conv2_2)
    conv2_2 = MaxPooling2D((2, 2))(conv2_2)

    conv3_1 = Conv2D(256, (3, 3), padding = 'same')(conv2_1)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_1 = Activation('relu')(conv3_1)
    conv3_1 = Dropout(dropout_prob_cl)(conv3_1)
    # conv3_1 = MaxPooling2D((2, 2))(conv3_1)

    conv3_2 = Conv2D(256, (3, 3), padding = 'same')(conv2_2)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_2 = Activation('relu')(conv3_2)
    conv3_2 = Dropout(dropout_prob_cl)(conv3_2)
    # conv3_2 = MaxPooling2D((2, 2))(conv3_2)

    concat = concatenate([conv3_1, conv3_2])
    concat = Dropout(dropout_prob_cl)(concat)
    # concat = MaxPooling2D((2, 2))(concat)

    flat_layer = Flatten()(concat)

    fc1 = Dense(1000)(flat_layer)
    fc1 = BatchNormalization()(fc1)
    fc1 = Activation('relu')(fc1)
    fc1 = Dropout(dropout_prob_fc)(fc1)

    fc2 = Dense(500)(fc1)
    fc2 = BatchNormalization()(fc2)
    fc2 = Activation('relu')(fc2)
    fc2 = Dropout(dropout_prob_fc)(fc2)

    prediction = Dense(10, activation='softmax')(fc2)
    model = Model(inputs = inputs, outputs = prediction)
    return model

def get_args():
    """
    function that parses and returns the command line inputs
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout_prob_cl", type = float, help = "dropout probability for convolutional layer", default = 0.25)
    parser.add_argument("--dropout_prob_fc", type = float, help = "dropout probability for fully connected layer", default = 0.5)
    parser.add_argument("--epochs", type = int, help = "number of epochs", default = 10)
    parser.add_argument("--batch_size", type = int, help = "batch size", default = 32)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # read from comand line
    args = get_args()
    dropout_prob_cl = args.dropout_prob_cl
    dropout_prob_fc = args.dropout_prob_fc
    epochs = args.epochs
    batch_size = args.batch_size
    # hard coding number of labels
    num_classes = 10
    # loading mnist data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    x_train = x_train.reshape(x_train.shape[0], 28, 28,1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28,1)

    # create and train keras model
    model = cnn(dropout_prob_cl, dropout_prob_fc)


    model.compile(loss=categorical_crossentropy, optimizer = 'adam', metrics=['accuracy'])

    # checkpoint = ModelCheckpoint('checkpoints/model-{epoch:03d}.h5', monitor='val_loss', verbose=0, save_best_only=False, mode='auto')

    model.fit(x_train, y_train, validation_split = 0.1, shuffle = True, epochs=epochs)
    model.save('model.h5')
    model.summary()
    _, test_accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('test accuracy = ', test_accuracy)

