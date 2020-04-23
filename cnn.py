# Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential  # to initialize NN
from keras.layers import Convolution2D  # for the convolution step of making CNN (making convolutional layers)
from keras.layers import MaxPooling2D  # for step 2- pooling step
from keras.layers import Flatten  # for step 3- flattening (converting into a large feature vector)
from keras.layers import Dense  # for adding fully connected layers in classic ANN

# Initialising the CNN
classifier = Sequential()  # NN is made in sequential (layers) form or graph form

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer (to improve the test set accuracy)
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))  # no need of input shape
# as it will know the size (which is not the original size) after execution of 1st layer 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
# output_dim not chosen as average as we don't know the no. of nodes of flattened layer.
# So, just a big number is chosen as an experiment
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))  # output layer

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)  # making changes to train images

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image 
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'




