# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 10:47:29 2018

@author: Administrator
"""

#%% 
import os
import numpy as np
import scipy.io as spio
import platform
from sklearn.utils import shuffle

# Running on the server on locally?
pl=platform.system()
if pl=='Linux':
    os.chdir('/nethome/nilsolav/repos/github/COGMAR_ACOUSTIC')
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
else:
    os.chdir('D:\\repos\\Github\\COGMAR_ACOUSTIC')

#NB Need to change the code to fit the frequencies present
freqs=4
batchsize=32+1

#%% Get training set and run training
def gettrainingset(filename,freqs):

    # File definitions
    matfile = filename+'.mat'
    #print(os.path.isfile(matfile)) 
    mat = spio.loadmat(matfile)
    #print(mat["F"])
    # Reshape data into training sets
    k=0
    # Initialize training set data array
    S= mat["ind"].shape
    imgs = np.zeros([S[0],freqs,400,400])
    speciesid = np.zeros([S[0],400,400])
    for i in range(0,S[0]):
        if mat["ind"][i,4]>10:
            x1 = mat["ind"][i,0]
            x2 = mat["ind"][i,0]+mat["ind"][i,2]
            y1 = mat["ind"][i,1]
            y2 = mat["ind"][i,1]+mat["ind"][i,3]
            nils=np.transpose(mat["sv"][x1:x2,y1:y2,:],(2,0,1))
            imgs[k,:,:,:] =nils[np.newaxis,:,:,:] 
            speciesid[k,:,:] = mat["I"][x1:x2,y1:y2]!=0
            k+=1
    # Release memory
    imgs = imgs[0:(k-1),:,:,:]        
    speciesid = speciesid[0:(k-1),:,:]
    speciesid = speciesid[:,np.newaxis,:,:]
    if k==0:
        imgs=np.empty([0,4,400,400])
        speciesid=np.empty([0,1,400,400])
        
    return imgs,speciesid


#%% Get the data


# Data folders
if pl=='Linux':
    fld0 = '/data/deep/data/echosounder/akustikk_all/data/DataOverview_North Sea NOR Sandeel cruise in Apr_May/'  
else:
    fld0 = 'D:\\data\\deep\\echosounder\\akustikk_all\\data\DataOverview_North Sea NOR Sandeel cruise in Apr_May\\'

imgs0 = np.empty([0,freqs,400,400])
speciesid0 = np.empty([0,1,400,400])

# Do da shit
for year in range(2008,2009):
    fld=fld0+str(year)
    flds = os.listdir(fld)
    for file in flds:
        if file.endswith(".mat"):
            print(file)
            filename, file_extension = os.path.splitext(file) 
            #try:
            if pl=='Linux':
                filefld = fld+'/'+filename  
            else:
                filefld = fld+'\\'+filename  
            imgs,speciesid = gettrainingset(filefld,freqs)
            # Write to HDF
            S=imgs.shape
            #print(S)
            if S[0]!=0:
                imgs0=np.concatenate((imgs0,imgs),axis=0)
                speciesid0=np.concatenate((speciesid0,speciesid),axis=0)
                
    # Randomize and write files
    S2 = imgs0.shape
    #S2[0]-np.floor(S2[0]/batchsize)*batchsize
    imgs0, speciesid0 = shuffle(imgs0, speciesid0, random_state=0)
    NF = int(np.floor(S2[0]/batchsize))
    for i in range(0,NF):
        imgs0_slice = imgs0[i*batchsize:((i+1)*batchsize-1),:,:,:]
        speciesid0_slice = speciesid0[i*batchsize:((i+1)*batchsize-1),:,:,:]
        np.savez(fld0+'batch'+str(year)+'_'+str(i),imgs=imgs0_slice,speciesid=speciesid0_slice)

    imgs0_slice = imgs0[((i+1)*batchsize):,:,:,:]
    speciesid0_slice = speciesid0[((i+1)*batchsize):,:,:,:]
    np.savez(fld0+'batch'+str(year)+'_'+str(i+1),imgs=imgs0_slice,speciesid=speciesid0_slice)

