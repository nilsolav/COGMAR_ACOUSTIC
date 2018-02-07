
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
#from keras.layers.core import Activation, Reshape, Permute
#from keras.models import Sequential
#from keras.layers import Input, Dense, Dropout, Embedding,  Conv2D, GlobalAveragePooling1D, MaxPooling2D, concatenate, Conv2DTranspose
#from keras.layers import Input, Dropout, Embedding,  Conv2D, MaxPooling2D, concatenate, Conv2DTranspose
#import matplotlib.pyplot as plt
#import time
#from keras import backend as K
#from keras.layers.normalization import BatchNormalization
#from keras.models import Model, load_model
import CM_AC_models as md


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

#%% Get the model
model = md.model1()

#%% Do da shit
for year in range(2008,2016):
    fld = '/data/deep/data/echosounder/akustikk_all/data/DataOverview_North Sea NOR Sandeel cruise in Apr_May/'  
    fld+=str(year)
    for file in os.listdir(fld):
        if file.endswith(".mat"):
            filename, file_extension = os.path.splitext(file) 
            print(filename)
            #try:
            imgs,speciesid = gettrainingset(fld+'/'+filename)
                # TRaining step
            model.fit(imgs,speciesid)
            #except:
            #    print(filename+' failed')







