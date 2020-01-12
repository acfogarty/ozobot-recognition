import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint


label_names = {0: 'Ozobot', 1: 'Not Ozobot'}
nclasses = 2  # number of categories to predict (Ozobot and not-Ozobot)
nwidth = 28  # width of image in pixels
nheight = 28  # height of image in pixels


def load_images():
    '''
    Returns:
        x: image data, shape: nimages * npixels * npixels
        y: labels, shape: nimages 
    '''

    nimages = 50
    x = np.random.normal(size=(nimages, nheight, nwidth))
    y = np.random.randint(low=1, high=nclasses, size=nimages, dtype=np.int32)

    return x, y


def preprocess_images(x):

    x = x / 255

    # reshape to the shape expected by tensorflow
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

    return x


def preprocess_labels(y):
    """ 
    one-hot encode the class labels
    Args:
        y (shape: nimages)
    Returns:
        y (shape: nimages * nclasses)
    """

    y = tf.keras.utils.to_categorical(y, nclasses)

    return y


def split_test_train(x, y):
    """
    TODO
    """

    x_train = x
    y_train = y

    x_test = x
    y_test = y

    return x_train, y_train, x_test, y_test


def build_cnn_model():

    dropouts = [0.3, 0.3, 0.5]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, 
                                     padding='same', activation='relu', 
                                     input_shape=(nheight, nwidth, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(dropouts[0]))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2,
                                     padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(dropouts[1]))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropouts[2]))
    model.add(tf.keras.layers.Dense(nclasses, activation='softmax'))
    model.summary()

    return model


def plot_labelled_images(x, y, y_predict=None):
    """
    Plot a random sample of images from x, with labels y

    If y_predict is supplied, show both predicted and true labels
    Else only show predicted labels

    Args
        x: image data, shape: nimages * npixels * npixels
        y: labels with one-hot encoding, shape: nimages * nclasses
    """

    nplot = 10
    nrows = 2
    ncols = 5

    # randomly choose which images from the dataset to plot 
    random_indices = np.random.choice(x.shape[0], size=nplot, replace=False)

    figure = plt.gcf()

    for i, index in enumerate(random_indices):
        ax = figure.add_subplot(nrows, ncols, i + 1, xticks=[], yticks=[])

        # plot image
        ax.imshow(np.squeeze(x[index]))

        # add label as title of image
        label_index = np.argmax(y[index])
        label = label_names[label_index]

        # if predicted labels have been supplied in addition to true labels, show both
        if y_predict is not None:
            predicted_label_index = np.argmax(y_predict[index])
            predicted_label = label_names[predicted_label_index]
            title = "true={} (predicted={})".format(label, predicted_label)

        # else only show true labels
        else:
            title = "true={}".format(label)

        ax.set_title(title)


def main():

    x, y = load_images()

    x = preprocess_images(x)
    y = preprocess_labels(y)

    plot_labelled_images(x, y)
    x_train, y_train, x_test, y_test = split_test_train(x, y)
    y_valid = y_test
    x_valid = x_test

    model = build_cnn_model()

    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', 
                                   verbose = 1, 
                                   save_best_only=True)
    model.fit(x_train,
              y_train,
              batch_size=64,
              epochs=10,
              validation_data=(x_valid, y_valid),
              callbacks=[checkpointer])

    model.load_weights('model.weights.best.hdf5')
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', score[1])

    y_predict = model.predict(x_test)

    plot_labelled_images(x, y, y_predict=y_predict)


if __name__ == '__main__':
    main()
