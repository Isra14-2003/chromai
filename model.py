### Importing the libraries

### Data preprocessing

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import keras


### Preprocessing the training dataset

train_datagen = ImageDataGenerator(rescale = 1./255,horizontal_flip = True)
training_set = train_datagen.flow_from_directory('Dataset/train_dataset',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

### Preprocessing the test dataset

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('Dataset/test_dataset',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

### Initializing the CNN

cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units = 256, activation = 'relu'))
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))

cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

### Compiling the CNN
cnn.compile(optimizer = 'nadam', loss = 'binary_crossentropy', metrics = ['accuracy'])

### Training the CNN on the training dataset and testing the CNN of the test dataset

solution = cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

### Saving model to the disk
cnn.save('model.h5')
