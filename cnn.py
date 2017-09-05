# -*- coding: utf-8 -*-


# Part 1 : Building a CNN 


from keras.models import Sequential 
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten 
from keras.layers import Dense

# Intialization CNN 
classifier = Sequential()

# Step 1 - Convolution layer
classifier.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))

# Step 2 - pooling
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))

# Second Convolutionallayer and pooling layer for high accuracy after first run of model
classifier.add(Conv2D(32, (3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))

# Step 3 - Flattening 
classifier.add(Flatten())

# Step 4 - Full Connection  
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Step 5 - Compiling CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 


# Part 2 : Fitting CNN to images 

# Used for image preprocessing to avoid overfitting because otherwise will give high accuracy on training set
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# Here test images need to be scaled but no other transformations are applicable to test images as others are 
# required for learning purpose on training set.        
test_datagen = ImageDataGenerator(rescale=1./255) 

train_generator = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_generator,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=validation_generator,
        validation_steps=2000)
        
        
# Part 3 - Making new Prediction 

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size=(64,64)) 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis= 0)
result = classifier.predict(test_image)
train_generator.class_indices 
if result[0][0]==1 : 
    prediction = "Dog"
else :
    prediction = "Cat"