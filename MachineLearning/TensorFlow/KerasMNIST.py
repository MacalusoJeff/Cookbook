# Example of training a CNN on the MNIST dataset in Keras
# Some of this code is likely outdated since it was written in ~2016

import sys
import time
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import advanced_activations
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from keras import backend as K

print(time.strftime('%Y/%m/%d %H:%M'))
print('OS:', sys.platform)
print('Python:', sys.version)
print('NumPy:', np.__version__)
print('Keras:', keras.__version__)

# Loading the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2)

# Input image dimensions
img_rows, img_cols = 28, 28
num_channels = 1

# Ensuring the channels are in the correct 
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], num_channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], num_channels, img_rows, img_cols)
    input_shape = (num_channels, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, num_channels)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, num_channels)
    input_shape = (img_rows, img_cols, num_channels)
    
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'Training samples')
print(X_test.shape[0], 'Testing samples')


# Printing backend and GPU information
if keras.backend.backend() == 'tensorflow':
    import tensorflow as tf
    from tensorflow.python.client import device_lib
    print('Backend: TensorFlow', tf.__version__)
    local_device_protos = device_lib.list_local_devices()
    print([x for x in local_device_protos if x.device_type == 'GPU'])

    # Avoiding memory issues with the GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

elif keras.backend.backend() == 'cntk':
    import cntk as C
    print('Backend: CNTK', C.__version__)
    print('GPU: ', C.gpu(0))

# Setting data types and normalizing the images
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'Training samples')
print(X_test.shape[0], 'Testing samples')

# Model settings
batch_size = 128
num_classes = 10
epochs = 12

# Converting class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# Beginning model building
model = Sequential()

# Layer 1 - Conv (5x5)
model.add(Conv2D(32, kernel_size=(5, 5), input_shape=input_shape))
model.add(advanced_activations.LeakyReLU(alpha=0.03))

# Layer 2 - Conv (5x5) & Max Pooling
model.add(Conv2D(32, kernel_size=(5, 5)))
model.add(advanced_activations.LeakyReLU(alpha=0.03))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

# Layer 3 - Conv (3x3)
model.add(Conv2D(64, kernel_size=(3, 3)))
model.add(advanced_activations.LeakyReLU(alpha=0.03))

# Layer 4 - Conv (3x3) & Max Pooling
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(advanced_activations.LeakyReLU(alpha=0.03))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

# Layer 5 - FC 1024
model.add(Flatten())
model.add(Dense(1024, activation='relu'))

# Layer 6 - FC 1024
model.add(Dense(1024, activation='relu'))

# Output Layer
model.add(Dense(num_classes, activation='softmax'))

# Defining loss function, optimizer, and metrics to report
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Summary of the model before fitting
model.summary()


### Setting up callbacks
# Early stopping
earlystop = callbacks.EarlyStopping(monitor='val_loss',
                                    min_delta=0.0001,  # Amount counting as an improvement
                                    patience=5,  # Number of epochs before stopping
                                    verbose=1, 
                                    mode='auto')

# Tracking the training time for each epoch
class TimeHistory(callbacks.Callback):
    '''
    Tracks training time on individual epochs for a Keras model
    '''
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

        
time_callback = TimeHistory()  # Gives training time for all epochs


# Model checkpoints - saves the model with the best validation loss
model_filepath = 'model.val_loss{val_loss:.5f}_epoch{epoch:04d}-.h5'
checkpoint = callbacks.ModelCheckpoint(model_filepath, monitor='val_loss',
                                       save_best_only=True)

# Reducing the learning rate if training loss does not increase
learning_rate_redux = callbacks.ReduceLROnPlateau(monitor='loss', 
                                                  patience=3,  # Reduce after 3 epochs
                                                  verbose=1, 
                                                  factor=0.3,  # Reduce to 1/3
                                                  min_lr=0.00001)


# Fitting the model
model_info = model.fit(X_train, y_train,
                       epochs=epochs,
                       batch_size=batch_size, verbose=1,
                       validation_split=0.1,  # Uses last 10% of data (not shuffled) for validation
                       callbacks=[earlystop, checkpoint, learning_rate_redux, time_callback])

# Getting test information
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print()
print('Test score:', score)
print('Test accuracy:', acc)

# Saving the model
# model.save('model.h5')

# Reporting total training time
total_training_time = round(sum(time_callback.times))
print('Total training time for {0} epochs: {1} seconds'.format(epochs, total_training_time))


def plot_model_loss_acc(model_history):
    '''
    Plots the accuracy and the loss for the training and
    validation sets by epoch
    '''
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    
    # Accuracy plot
    axs[0].plot(range(1, len(model_history.history['acc'])+1),
                      model_history.history['acc'])
    axs[0].plot(range(1, len(model_history.history['val_acc'])+1), 
                      model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['acc'])+1), 
                                len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    
    # Loss plot
    axs[1].plot(range(1, len(model_history.history['loss'])+1),
                      model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss'])+1),
                      model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss'])+1),
                                len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    

plot_model_loss_acc(model_info)

# Data augmentation
datagenerator = ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=None,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

datagenerator.fit(X_train)

# Fit the model on the batches generated by datagen.flow().
model_info = model.fit_generator(generator=datagenerator.flow(X_train, y_train,
                                                              batch_size=batch_size,
                                                              shuffle=True),
                                 steps_per_epoch=10*round(X_train.shape[0] / batch_size),
                                 epochs=epochs,
                                 validation_data=(X_val, y_val),
                                 verbose=1,
                                 callbacks=[time_callback,  # Gives epoch training times
                                            earlystop,
                                            callbacks.ModelCheckpoint('model.h5', save_best_only=True)])