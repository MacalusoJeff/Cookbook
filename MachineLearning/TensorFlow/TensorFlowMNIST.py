import sys
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras

print(time.strftime('%Y/%m/%d %H:%M'))
print('OS:', sys.platform)
print('Python:', sys.version)
print('NumPy:', np.__version__)
print('TensorFlow:', tf.__version__)

# Checking tensorflow processing devices
from tensorflow.python.client import device_lib
local_device_protos = device_lib.list_local_devices()
print([x for x in local_device_protos if x.device_type == 'GPU'])

# Avoiding memory issues with the GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Using Keras to get the data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2)


image_size = 28
num_labels = 10
num_channels = 1


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


X_train, y_train = reformat(X_train, y_train)
X_validation, y_validation = reformat(X_validation, y_validation)
X_test, y_test = reformat(X_test, y_test)
print('Reformatted data shapes:')
print('Training set', X_train.shape, y_train.shape)
print('Validation set', X_validation.shape, y_validation.shape)
print('Test set', X_test.shape, y_test.shape)

# Training Parameters
learning_rate = 0.001
num_steps = y_train.shape[0] + 1  # 200,000 per epoch
batch_size = 128
epochs = 12
display_step = 250  # To print progress

# Network Parameters
num_input = 784  # Data input (image shape: 28x28)
num_classes = 10  # Total classes (10 characters)

num_hidden = 1024

# Defining the model and graph
sgd_hidden_graph = tf.Graph()
with sgd_hidden_graph.as_default():
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(
        tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(X_validation.astype('float32'))
    tf_test_dataset = tf.constant(X_test, name='test_data')

    w0 = tf.Variable(tf.truncated_normal(
        [image_size * image_size, num_hidden]), name='W0')
    w1 = tf.Variable(tf.truncated_normal([num_hidden, num_labels]), name='W1')

    b0 = tf.Variable(tf.zeros([num_hidden]), name='b0')
    b1 = tf.Variable(tf.zeros([num_labels]), name='b1')

    def reluLayer(dataset):
        return tf.nn.relu(tf.matmul(dataset, w0) + b0)

    def logitLayer(dataset):
        return tf.matmul(reluLayer(dataset), w1) + b1

    sgd_hidden_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=tf_train_labels,
            logits=logitLayer(tf_train_dataset)))
    sgd_hidden_optimizer = tf.train.GradientDescentOptimizer(
        0.5).minimize(sgd_hidden_loss)

    sgd_hidden_train_prediction = tf.nn.softmax(
        logitLayer(tf_train_dataset), name='train_predictor')
    sgd_hidden_valid_prediction = tf.nn.softmax(
        logitLayer(tf_valid_dataset), name='validate_predictor')
    sgd_hidden_test_prediction = tf.nn.softmax(
        logitLayer(tf_test_dataset), name='test_predictor')


# Creating the graph and running the model
with tf.Session(graph=sgd_hidden_graph) as sgd_hidden_session:
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()

    # For tracking execution time and progress
    start_time = time.time()
    total_steps = 0

    print('\nInitialized\n')
    try:
        for epoch in range(1, epochs + 1):
            print('Beginning Epoch {0} -'.format(epoch))
            for step in range(num_steps):
                offset = (step * batch_size) % (y_train.shape[0] - batch_size)
                batch_data = X_train[offset:(offset + batch_size), :]
                batch_labels = y_train[offset:(
                    offset + batch_size), :].reshape(batch_size, num_labels)
                feed_dict = {tf_train_dataset: batch_data,
                             tf_train_labels: batch_labels}
                _, l, sgd_hidden_predictions = sgd_hidden_session.run(
                    [sgd_hidden_optimizer, sgd_hidden_loss, sgd_hidden_train_prediction],
                    feed_dict=feed_dict)
                if (step % 1000 == 0) or (step == num_steps):
                    # Calculating percentage of completion
                    total_steps += step
                    pct_epoch = (step / float(num_steps)) * 100
                    pct_total = (total_steps / float(num_steps *
                                                     (epochs + 1))) * 100  # Fix this line

                    # Printing progress
                    print('Epoch %d Step %d (%.2f%% epoch, %.2f%% total)' %
                          (epoch, step, pct_epoch, pct_total))
                    print('------------------------------------')
                    print('Minibatch loss: %f' % l)
                    print("Minibatch accuracy: %.1f%%" %
                          accuracy(sgd_hidden_predictions, batch_labels))
                    print("Validation accuracy: %.1f%%" % accuracy(sgd_hidden_valid_prediction.eval(),
                                                                   y_validation))
                    print(datetime.now())
                    print('Total execution time: %.2f minutes' %
                          ((time.time() - start_time) / 60.))
                    print()
        print("Test accuracy: %.1f%%" % accuracy(
            sgd_hidden_test_prediction.eval(), y_test.astype('int32')))
    except KeyboardInterrupt:
        print('Training manually ended')
#     saver.save(sgd_hidden_session, '{}/model'.format(Path.cwd()))


### Messy code to load a trained model and generate models
# Additionally contains code for creating a pickle file from images within subdirectories
import pickle
# from six.moves import cPickle as pickle
from scipy import ndimage
import math
import numpy as np
import os
from pathlib import Path
import tensorflow as tf
from pathlib import Path

# Change this line to the saved model
modelFile = 'trainedModel.meta'

image_size = 28  # Pixel width and height.
num_labels = 10
pixel_depth = 255.0  # Number of levels per pixel.

total_images = sum([len(files) for r, d, files in os.walk("./data/")])


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = list(folder.iterdir())
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = folder.joinpath(image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' %
                                str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file,
                  ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    data_folders = (i for i in data_folders if i.is_dir())
    for folder in data_folders:
        set_filename = str(folder) + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)
    return dataset_names


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def reformat(dataset, labels):
    dataset = dataset.reshape(
        labels.shape[0], image_size, image_size, 1).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


def merge_datasets(pickle_files, dataset_size):
    """
    Merge multiple glyph pickle files into nd-array dataset and nd-array labels
    for model evaluation.
    Simplification from https://github.com/udacity/deep-learning
    """
    num_classes = len(pickle_files)
    dataset, labels = make_arrays(dataset_size, image_size)
    size_per_class = dataset_size // num_classes

    start_t = 0
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                end_t = start_t + size_per_class
                np.random.shuffle(letter_set)
                dataset[start_t:end_t, :, :] = letter_set[0:size_per_class]
                labels[start_t:end_t] = label
                start_t = end_t
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return dataset, labels


dir_name = 'data'

glyph_dir = Path.cwd().joinpath(dir_name)
test_folders = [glyph_dir.joinpath(i) for i in glyph_dir.iterdir()]
test_datasets = maybe_pickle(test_folders, 2)  # provide only 2 samples for now
test_dataset, test_labels = merge_datasets(test_datasets, total_images)
test_dataset, test_labels = randomize(test_dataset, test_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Testing size', test_dataset.shape, test_labels.shape)


def classify(test_dataset, model_filename=modelFile, checkpoint_path=None):
    """
    A sample classifier to unpickle a TensorFlow model and label the dataset.
    There are magic strings in this function derived from to the model to evaluate.
    Your implementation will likely have different tags that depend upon the model
    implementation.

    We pad the input test_dataset to make it at least as large as the model batch,
    and repeat prediction on chunks of the input if it is larger than the model batch.

    Args:
        test_dataset: Expect an input of N*28*28*1 shape numpy array, where N is number of images and
                      28*28 is pixel width and hieght.
        model_filename: optional file name stored by a previous TensorFlow session.
        checkpoint_path: optional path for previous TensorFlow session.

    Returns:
        The #observations by #labels nd-array labelings.
    """
    # Re-scaling the dataset to a similar scale as the training data in the notMNIST.pickle file
    pixel_depth = 255.0
    test_dataset = (test_dataset - 255.0 / 2) / 255

    num_classes = 10
    n = int(test_dataset.shape[0])
    result = np.ndarray([n, num_classes], dtype=np.float32)

    with tf.Session() as session:
        saver = tf.train.import_meta_graph('./model/' + model_filename)
        saver.restore(session, './model/' + model_filename.split('.')[0])
        graph_predict = tf.get_default_graph()
        test_predict = graph_predict.get_tensor_by_name(
            'Softmax_1:0')  # string from model
        m = int(graph_predict.get_tensor_by_name(
            'Placeholder:0').shape[0])   # string from model
        for i in range(0, int(math.ceil(n / m))):
            start = i * m
            end = min(n, ((i + 1) * m))
            x = np.zeros((128, 28, 28, 1)).astype(np.float32)
            x[0:(end - start)] = test_dataset[start:end]
            result[start:end] = test_predict.eval(
                feed_dict={"Placeholder:0": x})[0:(end - start)]
    return result


def testing_accuracy(data, labels):
    """
    Generates predictions and returns the accuracy
    """
    predictions = classify(data)
    print("Test accuracy: %.1f%%" % accuracy(predictions, labels))


testing_accuracy((test_dataset), test_labels)