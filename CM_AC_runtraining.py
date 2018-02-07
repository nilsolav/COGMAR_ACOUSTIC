
#%% Train the network per year
#for year in range(2007,2016):
from pathlib import Path
import os
import numpy as np
import scipy.io as spio

#from sklearn.model_selection import train_test_split
#from sklearn import preprocessing
#from sklearn.utils import shuffle 

#from keras.utils import plot_model 
#from keras.utils import to_categorical 
from keras.layers.core import Activation, Reshape, Permute
#from keras.models import Sequential
#from keras.layers import Input, Dense, Dropout, Embedding,  Conv2D, GlobalAveragePooling1D, MaxPooling2D, concatenate, Conv2DTranspose
from keras.layers import Input, Dropout, Embedding,  Conv2D, MaxPooling2D, concatenate, Conv2DTranspose
#import matplotlib.pyplot as plt
#import time
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model

###################################### INPUT LAYER
height = 400
width = 400
K.set_image_dim_ordering('th') # Theano dimension ordering in this code
nbClass = 3
kernel = 3
img_input = Input(shape=(6, height, width))

###################################### ENCODER

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

####################################### Center

convC = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(pool3)
convC = BatchNormalization(axis=1)(convC)
convC = Conv2D(32, (kernel, kernel), padding="same", activation='relu' , kernel_initializer='he_normal')(convC)
convC = BatchNormalization(axis=1)(convC)
convC = Dropout(0.25)(convC)

####################################### DECODER

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
x = Reshape((nbClass, height * width))(x) 
x = Permute((2, 1))(x)
x = Activation("softmax")(x)
eddynet = Model(img_input, x)

###################################

eddynet.compile(optimizer = 'adam',
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])


####################################### Define extraction of the training set

def gettrainingset(filename):

    # File definitions
    
    matfile = filename+'.mat'
    print(os.path.isfile(matfile)) 
    rawfile = filename+'.raw'
    mat = spio.loadmat(matfile)

    #%% Reshape data into training sets
    k=0
    # Initialize training set data array
    S= mat["ind"].shape
    imgs = np.zeros([S[0],6,400,400])
    speciesid = np.zeros([S[0],400,400])

    for i in range(0,S[0]):
        if mat["ind"][i,4]>10000:
            x1 = mat["ind"][i,0]
            x2 = mat["ind"][i,0]+mat["ind"][i,2]
            y1 = mat["ind"][i,1]
            y2 = mat["ind"][i,1]+mat["ind"][i,3]
            nils=np.transpose(mat["sv"][x1:x2,y1:y2,:],(2,0,1))
            imgs[k,:,:,:] =nils[np.newaxis,:,:,:] 
            speciesid[k,:,:] = mat["I"][x1:x2,y1:y2]!=0
            k+=1
    # Release memory
    imgs = imgs[1:(k-1),:,:,:]        
    speciesid = speciesid[1:(k-1),:,:]        
    return imgs,speciesid

#%%
for year in range(2008,2016):
    fld = '/data/deep/data/echosounder/akustikk_all/data/DataOverview_North Sea NOR Sandeel cruise in Apr_May/'  
    fld+=str(year)
    for file in os.listdir(fld):
        if file.endswith(".mat"):
            filename, file_extension = os.path.splitext(file) 
            print(filename)
            imgs,speciesid = gettrainingset(fld+'/'+filename)
            # TRaining step
            eddynet.fit(imgs,speciesid)







