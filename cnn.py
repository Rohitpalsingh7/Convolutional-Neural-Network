#!/bin/env python3


#SBATCH -N 1


#SBATCH -n 2


#SBATCH --mem=26G


#SBATCH -p short


#SBATCH -C K40


#SBATCH -o tf_test.out


#SBATCH -t 24:00:00


#SBATCH --gres=gpu:2


# Convolutional Neural Network 

# Data set is the sample data set from Kaggle that is the benchmark for evaluating the performance of CNN


# Part 1 : Building a CNN 

# Importing Keras libraries and packages 
from keras.models import Sequential 
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten 
from keras.layers import Dense
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator


with tf.device('/cpu:0'): 
   
   # Intialization CNN : CNN is sequence of layers 
   classifier = Sequential()

   # First Convolution layer
   classifier.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))

   # First pooling/downsampling layer
   classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))

   # Second Convolutional layer 
   classifier.add(Conv2D(32, (3,3), activation='relu'))

   # Second pooling layer for high accuracy
   classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))

   # Flattening 
   classifier.add(Flatten())

   # Full Connection  
   classifier.add(Dense(units=128, activation='relu'))
   classifier.add(Dropout(rate=0.2))
   classifier.add(Dense(units=1, activation='sigmoid'))

   # Compiling CNN
   classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 



   # Part 2 : Fitting CNN to images 

   # Image Augmentaion used for enriching dataset to avoid overfitting ...
   # For higher accuracy, target size & number of layers can be increased ...


   train_datagen = ImageDataGenerator( rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

   test_datagen = ImageDataGenerator(rescale=1./255) 

   training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                   target_size=(64, 64),
                                                   batch_size=32,
                                                   class_mode='binary')

   test_set = test_datagen.flow_from_directory('dataset/test_set',
                                               target_size=(64, 64),
                                               batch_size=32,
                                               class_mode='binary')
   
with tf.device('/gpu:0'):

   classifier.fit_generator(training_set,
                           steps_per_epoch=8000,
                           epochs=100,
                           validation_data=test_set,
                           validation_steps=2000)

   # K Cross-fold validation is not required in this case as evaluation has been done on validation set



   # Part 3 - Making new Prediction 

   import numpy as np
   from keras.preprocessing import image
   test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size=(64,64)) 
   test_image = image.img_to_array(test_image)  # To have input image with 3 dimensions ... 
   test_image = np.expand_dims(test_image, axis= 0)
   result = classifier.predict(test_image)
train_datagen.class_indices 
if result[0][0]==1 : 
    prediction = "Dog"
else :
    prediction = "Cat"
 
    
