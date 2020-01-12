import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint


label_names = {1: 'Ozobot', 0: 'Not Ozobot'}
nclasses = 2  # number of categories to predict (Ozobot and not-Ozobot)
nwidth = 28  # width of image in pixels
nheight = 28  # height of image in pixels


def load_images():
    '''
    Returns:
        x: image data, shape: nimages * npixels * npixels
        y: labels, shape: nimages 
    '''

    nimages = 1000

    # create "images" of random noise
    x = np.random.normal(size=(nimages, nheight, nwidth))

    # randomly label the images as 0 (not-Ozobot) or 1 (Ozobot)
    y = np.random.randint(low=0, high=nclasses, size=nimages, dtype=np.int32)

    # draw a circle
    xx, yy = np.mgrid[:nheight, :nwidth]
    circle = (xx - nheight/2) ** 2 + (yy - nwidth/2) ** 2

    # rescale circle so that it stands out above the noise
    circle = circle / circle.max() * 5

    # add the circle to the "Ozobot" images
    x[y == 1] = x[y == 1] + circle

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


def split_train_validation_test(x, y, split):
    """
    split data into training, validation and test sets
    Args:
        x: dataset where first dimension has length 'nsamples'
        y: dataset where first dimension has length 'nsamples'
        split (list of floats): [fraction_train,fraction_validation,fraction_test]
    """

    nsamples = x.shape[0]

    if y.shape[0] != nsamples:
        raise Exception('in split_train_validation_test, x has shape {}'.format(x.shape) +
                        'but y has shape {}'.format(y.shape) +
                        'First dimensions do not match')

    # make sure split array sums to 1
    split = np.asarray(split)
    split = split / split.sum()

    nsamples_train = int(split[0] * nsamples)
    nsamples_valid = int(split[1] * nsamples)

    # create a set of randomly shuffled indices
    indices = np.random.permutation(nsamples)

    idx_train = indices[:nsamples_train]
    idx_valid = indices[nsamples_train:nsamples_train+nsamples_valid]
    idx_test = indices[nsamples_train+nsamples_valid:]

    x_train = x[idx_train]
    y_train = y[idx_train]

    x_valid = x[idx_valid]
    y_valid = y[idx_valid]

    x_test = x[idx_test]
    y_test = y[idx_test]

    return x_train, y_train, x_valid, y_valid, x_test, y_test


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
    Else only show true labels

    Args
        x: image data, shape: nimages * npixels * npixels
        y: labels with one-hot encoding, shape: nimages * nclasses
    """

    plt.clf()

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
            title = "true={}\n(predicted={})".format(label, predicted_label)

        # else only show true labels
        else:
            title = "true={}".format(label)

        ax.set_title(title)

    size = figure.get_size_inches()
    figure.set_size_inches(size[0]*2, size[1]*2)

    plt.show()


def main():

    x, y = load_images()

    x = preprocess_images(x)
    y = preprocess_labels(y)

    plot_labelled_images(x, y)

    x_train, y_train, x_valid, y_valid, x_test, y_test = split_train_validation_test(x, y, 
                                                                                     split=[0.7, 0.1, 0.2])

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

    plot_labelled_images(x_test, y_test, y_predict=y_predict)


if __name__ == '__main__':
    main()
