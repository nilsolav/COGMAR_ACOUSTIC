# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 10:47:29 2018

@author: Administrator
"""

#%% Preparing the data for training
#
# This cript reads the matlab files generated per mat file and extract the 
# 400 by 400 windows defined in the indices in the mat file. Data is stored in
# batches per year. The frames are randomized per year before storing.
#
# Each stored file contain both the data (imgs) and the classes (speciesid)
#
# imgs.shape = (-1, len(freqs), 400, 400)
# speciesid.shape=(-1, 0, 400, 400)
#
# The frequencies to be stored in the files:
freqs=[18, 38, 70, 120, 200, 333]
# And the number of images per file
batchsize=32+1
# If the frequency is missing it will be set to zeros
# The minimum number of positive class pixels per image that goes into the 
# training set:
minpixels=10

# Import libraries
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


#%% Get training set and run training
def gettrainingset(filename,freqs,minpixels):

    # File
    matfile = filename+'.mat'
    #print(os.path.isfile(matfile)) 
    mat = spio.loadmat(matfile)
    print('Freqs in file:',mat['F'].astype(int),' Freqs stored:',freqs)
    #print(freqs)
    k=0 # Index couning non zero images
    # Initialize training set data array
    S= mat["ind"].shape
    imgs = np.zeros([S[0],len(freqs),400,400])
    speciesid = np.zeros([S[0],400,400])
    print(S[0])
    for i in range(0,S[0]):
        if mat["ind"][i,4]>minpixels:
            x1 = mat["ind"][i,0]
            x2 = mat["ind"][i,0]+mat["ind"][i,2]
            y1 = mat["ind"][i,1]
            y2 = mat["ind"][i,1]+mat["ind"][i,3]
            nils=np.transpose(mat["sv"][x1:x2,y1:y2,:],(2,0,1))
            # Check the number of frequencies, if more than 4, choose
            # [18 38 120 200]
            k1=0
            for fr in freqs:
                for fr1 in mat['F'].astype(int).transpose():
                    imgs[k,k1,:,:] =nils[np.newaxis,k1,:,:] 
                    k1=+1

            speciesid[k,:,:] = mat["I"][x1:x2,y1:y2]!=0
            k+=1
    # Release memory
    imgs = imgs[0:(k-1),:,:,:]        
    speciesid = speciesid[0:(k-1),:,:]
    speciesid = speciesid[:,np.newaxis,:,:]
    if k==0:
        imgs=np.empty([0,len(freqs),400,400])
        speciesid=np.empty([0,1,400,400])
        
    return imgs,speciesid


#%% Get the data


# Data folders
if pl=='Linux':
    fld0 = '/data/deep/data/echosounder/akustikk_all/data/DataOverview_North Sea NOR Sandeel cruise in Apr_May/'  
else:
    fld0 = 'D:\\data\\deep\\echosounder\\akustikk_all\\data\DataOverview_North Sea NOR Sandeel cruise in Apr_May\\'

imgs0 = np.empty([0,len(freqs),400,400])
speciesid0 = np.empty([0,1,400,400])

# Do da shit
for year in range(2005,2016):
    fld=fld0+str(year)
    flds = os.listdir(fld)
    nofiles = False
    for file in flds:
        if file.endswith(".mat"):
            print('Reading '+file)
            nofiles = True
            filename, file_extension = os.path.splitext(file) 
            #try:
            if pl=='Linux':
                filefld = fld+'/'+filename  
            else:
                filefld = fld+'\\'+filename  
            imgs,speciesid = gettrainingset(filefld,freqs,minpixels)
            
            # Write to npz
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
        print('Storing '+fld0+'batch'+str(year)+'_'+str(i)+'.npz')
        imgs0_slice = imgs0[i*batchsize:((i+1)*batchsize-1),:,:,:]
        speciesid0_slice = speciesid0[i*batchsize:((i+1)*batchsize-1),:,:,:]
        np.savez(fld0+'batch'+str(year)+'_'+str(i),imgs=imgs0_slice,speciesid=speciesid0_slice)
    
    if nofiles:
        imgs0_slice = imgs0[((i+1)*batchsize):,:,:,:]
        speciesid0_slice = speciesid0[((i+1)*batchsize):,:,:,:]
        np.savez(fld0+'batch'+str(year)+'_'+str(i+1),imgs=imgs0_slice,speciesid=speciesid0_slice)
    else:
        print('No files for '+str(year))
