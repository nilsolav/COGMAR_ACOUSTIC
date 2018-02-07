# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 08:06:30 2018

@author: Administrator
"""
from keras import backend as K
from keras.utils import plot_model 
#from keras.utils import to_categorical 
from keras.layers.core import Activation, Reshape, Permute
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Embedding,  Conv2D, GlobalAveragePooling1D, MaxPooling2D, concatenate, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model

K.set_image_dim_ordering('th') # Theano dimension ordering in this code
#%% Dice Loss function
smooth = 1e-5

def dice_coef_anti(y_true, y_pred):
    y_true_anti = y_true[:,:,1]
    y_pred_anti = y_pred[:,:,1]
    intersection_anti = K.sum(y_true_anti * y_pred_anti)
    return (2 * intersection_anti + smooth) / (K.sum(y_true_anti)+ K.sum(y_pred_anti) + smooth)

def dice_coef_cyc(y_true, y_pred):
    y_true_cyc = y_true[:,:,2]
    y_pred_cyc = y_pred[:,:,2]
    intersection_cyc = K.sum(y_true_cyc * y_pred_cyc)
    return (2 * intersection_cyc + smooth) / (K.sum(y_true_cyc) + K.sum(y_pred_cyc) + smooth)

def dice_coef_nn(y_true, y_pred):
    y_true_nn = y_true[:,:,0]
    y_pred_nn = y_pred[:,:,0]
    intersection_nn = K.sum(y_true_nn * y_pred_nn)
    return (2 * intersection_nn + smooth) / (K.sum(y_true_nn) + K.sum(y_pred_nn) + smooth)
    
def mean_dice_coef(y_true, y_pred):
    return (dice_coef_anti(y_true, y_pred) + dice_coef_cyc(y_true, y_pred) + dice_coef_nn(y_true, y_pred))/3.
           
def dice_coef_loss(y_true, y_pred):
    return 1 - mean_dice_coef(y_true, y_pred) 


#%%
def model1():
    model1 = Sequential()
    # Step 1 - Convolution (32 filters / feature detectors each having 3x3 dimension)
    model1.add(Conv2D(16, (3, 3), padding = 'same', input_shape = (6, 400, 400), activation = 'relu'))
    # Adding a second convolutional layer (32 filters / feature detectors each having 3x3 dimension)
    model1.add(Conv2D(8, (3,3), padding = 'same', activation = 'relu'))
    #model.add(MaxPooling2D(pool_size = (3,3), strides=2))
    model1.add(Conv2D(1, (3,3), padding = 'same', activation = 'softmax'))
    #model.add(MaxPooling2D(pool_size = (3,3), strides=2))

    # Compile model 1
    model1.compile(optimizer = 'adam', loss=dice_coef_loss,
                     metrics=['categorical_accuracy', mean_dice_coef])
    return model1

#%% Model 2
def model2():
    width = 400
    height = 400
    nbClass = 3
    
    kernel = 3
    
    ###################################### INPUT LAYER
    
    img_input = Input(shape=(6, height, width))
    
    ######################################ENCODER
    
    conv1 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(img_input)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = Dropout(0.25)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(pool1)
    conv2 = BatchNormalization(axis=1)(conv2)
    conv2 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(conv2)
    conv2 = BatchNormalization(axis=1)(conv2)
    conv2 = Dropout(0.25)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(pool2)
    conv3 = BatchNormalization(axis=1)(conv3)
    conv3 = Dropout(0.25)(conv3)
    conv3 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(conv3)
    conv3 = BatchNormalization(axis=1)(conv3)
    conv3 = Dropout(0.25)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    #######################################center
    
    convC = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(pool3)
    convC = BatchNormalization(axis=1)(convC)
    convC = Conv2D(32, (kernel, kernel), padding="same", activation='relu' , kernel_initializer='he_normal')(convC)
    convC = BatchNormalization(axis=1)(convC)
    convC = Dropout(0.25)(convC)
    
    #######################################DECODER
    
    up3 = concatenate([Conv2DTranspose(32, (kernel, kernel), strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_normal' )(convC), conv3], axis=1)
    decod3 = BatchNormalization(axis=1)(up3)
    decod3 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(decod3)
    decod3 = BatchNormalization(axis=1)(decod3)
    decod3 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(decod3)
    decod3 = BatchNormalization(axis=1)(decod3)
    decod3 = Dropout(0.25)(decod3)
    
    up2 = concatenate([Conv2DTranspose(32, (kernel, kernel), strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_normal' )(decod3), conv2], axis=1)
    decod2 = BatchNormalization(axis=1)(up2)
    decod2 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(decod2)
    decod2 = BatchNormalization(axis=1)(decod2)
    decod2 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(decod2)
    decod2 = BatchNormalization(axis=1)(decod2)
    decod2 = Dropout(0.25)(decod2)
    
    up1 = concatenate([Conv2DTranspose(32, (kernel, kernel), strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_normal' )(decod2), conv1], axis=1)
    decod1 = BatchNormalization(axis=1)(up1)
    decod1 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(decod1)
    decod1 = BatchNormalization(axis=1)(decod1)
    decod1 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(decod1)
    decod1 = BatchNormalization(axis=1)(decod1)
    decod1 = Dropout(0.25)(decod1)
    
    ####################################### Segmentation Layer
    
    x = Conv2D(nbClass, (1, 1), padding="valid" )(decod1) 
    #x = Reshape((nbClass, height * width))(x) 
    #x = Permute((2, 1))(x)
    x = Activation("softmax")(x)
    model2 = Model(img_input, x)
    
    # Compiling the CNN
    model2.compile(optimizer='adam', loss=dice_coef_loss,
                    metrics=['categorical_accuracy', mean_dice_coef])
    
    return model2

#%% Model 2
def model3():
    width = 400
    height = 400
    nbClass = 1
    kernel = 3
    
    ###################################### INPUT LAYER
    
    img_input = Input(shape=(6, height, width))
    
    ######################################ENCODER
    
    conv1 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(img_input)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = Dropout(0.25)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
    #######################################center
    
    convC = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(pool1)
    convC = BatchNormalization(axis=1)(convC)
    convC = Conv2D(32, (kernel, kernel), padding="same", activation='relu' , kernel_initializer='he_normal')(convC)
    convC = BatchNormalization(axis=1)(convC)
    convC = Dropout(0.25)(convC)
    
    #######################################DECODER
    
    up1 = concatenate([Conv2DTranspose(32, (kernel, kernel), strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_normal' )(convC), conv1], axis=1)
    decod1 = BatchNormalization(axis=1)(up1)
    decod1 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(decod1)
    decod1 = BatchNormalization(axis=1)(decod1)
    decod1 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(decod1)
    decod1 = BatchNormalization(axis=1)(decod1)
    decod1 = Dropout(0.25)(decod1)
    
    ####################################### Segmentation Layer
    
    x = Conv2D(nbClass, (1, 1), padding="valid" )(decod1) 
    #x = Reshape((nbClass, height * width))(x) 
    #x = Permute((2, 1))(x)
    x = Activation("softmax")(x)
    model3 = Model(img_input, x)
    
    # Compiling the CNN
    model3.compile(optimizer='adam', loss=dice_coef_loss,
                    metrics=['categorical_accuracy', mean_dice_coef])
    
    return model3
