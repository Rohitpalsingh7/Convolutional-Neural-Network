# Convolutional-Neural-Network

Image dataset of dogs and cats from kaggle is used for CNN. It is the benchmark dataset to evaluate the performance of CNN for classifying images into dog and cat. 

# Architechture of CNN 

-- Block I :
a. Convolutional Layer : 32 Feature maps of size (3,3) generating 32 Activation maps. 
b. MaxPooling/Downsamplng Layer : Pool size (3,3) with strides = 2

-- Block II :
a. Convolutional Layer : 32 Feature maps generating 32 Activation maps.
b. MaxPooling/Downsampling Layer : Pool size (3,3) with strides = 2

-- Block III :
a. Flattening 

-- Block IV : 
a. Fully connected layer with dropout to avoid overfitting 


# Image Augmentation 

Used image augmentation. Image augmentation is widely used technique to enrich the dataset if dataset is not large enough for learning process. It performs random transformations per batch. Some of transformations are like flipping vertically, horizontally, rotation, zoom etc. Main idea of image augmentation is to include small perturbations in images and keeps central object intact so that it can deal with real world changes and can genearalize better for unseen data. 


# Tuning 

This work does not include tuning because of lack of compute. However, there are different techniques for performing tuning for complex models such as CNN. One of them includes two steps as follow :

-- Step a : Freeze & Pretraining 

Use CNN's initial architecture and it's learnt weigths from previous work such as ResNet, VGG but exlcudes later layers (fully connected layers). Customize fully connected layers according to our problem.  Freeze the initial layers obtained from previous work and train only the customized fully connected layer (also called pretraining learning). 

-- Step b : Fine Tuning 

Once pretrained weights are obtained from (step a), fine tuning the whole CNN model. 


# Note : GlobalMaxPooling or GlobalAvgPooling is used over flattening to reduce the number of parameters that are fed into fully connected layer so that overfitting can be avoided.  







