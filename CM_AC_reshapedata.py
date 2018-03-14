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
#freqs=[18, 38, 70, 120, 200, 333]
freqs=[18, 38, 120, 200]
# And the number of images per file
batchsize=32
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
    #print('Reading '+file+ '. Freqs in file:'+ str(mat['F'].astype(int))+'. Freqs stored:'+str(freqs))
    #print(freqs)
    k=0 # Index couning non zero images
    # Initialize training set data array
    S = mat["ind"].shape
    S2=mat['F'].shape
    Finfile=mat['F'].astype(int).transpose()
    S3=len(freqs)
    imgs = np.zeros([S[0],len(freqs),200,200])
    speciesid = np.zeros([S[0],200,200])
    for i in range(0,S[0]):
        if mat["ind"][i,4]>minpixels:
            #print(i)
            x1 = mat["ind"][i,0]
            x2 = mat["ind"][i,0]+mat["ind"][i,2]
            y1 = mat["ind"][i,1]
            y2 = mat["ind"][i,1]+mat["ind"][i,3]
            nils=np.transpose(mat["sv"][x1:x2,y1:y2,:],(2,0,1))
            # How many NaN's are there in the intersect?
            nnan=np.count_nonzero(np.isnan(nils.flatten()))
            if nnan>0:
                print('NaNs in intersect, removing')
            else:
                k1=0
                speciesid[k,:,:] = mat["I"][x1:x2,y1:y2]!=0
                
                #print(k)
                for k1 in range(0,S3):
                    for k2 in range(0,S2[1]):
                        if Finfile[k2]==freqs[k1]:
                            imgs[k,k1,:,:] =nils[np.newaxis,k2,:,:] 
                            #print('-----')
                            #print(Finfile[k2])
                            #print(freqs[k1])
                k+=1
                
    # Release memory
    imgs = imgs[0:k:1,:,:,:]        
    speciesid = speciesid[0:k:1,:,:]
    speciesid = speciesid[:,np.newaxis,:,:]
    if k==0:
        imgs=np.empty([0,len(freqs),200,200])
        speciesid=np.empty([0,1,200,200])
        
    return imgs,speciesid

#%% Get the data


# Data folders
if pl=='Linux':
    fld0 = '/data/deep/data/echosounder/akustikk_all/data/DataOverview_North Sea NOR Sandeel cruise in Apr_May/'  
else:
    fld0 = 'C:\\DATA\\deep\\echosounder\\akustikk_all\\data\DataOverview_North Sea NOR Sandeel cruise in Apr_May\\'
            
# Do da shit
for year in range(2007,2017):
    imgs0 = np.empty([0,len(freqs),200,200])
    speciesid0 = np.empty([0,1,200,200])
    fld=fld0+str(year)
    batchno=0
    flds = os.listdir(fld)
    #flds = flds[0:263]#debug hack
    filno=0
    for file in flds:
        filno+=1
        if file.endswith(".mat"):
            filename, file_extension = os.path.splitext(file) 
            try:
                #if file=='2008205-D20080515-T091741.mat': 
                if pl=='Linux':
                    filefld = fld+'/'+filename  
                else:
                    filefld = fld+'\\'+filename  
                    
                imgs,speciesid = gettrainingset(filefld,freqs,minpixels)
                # Write to HDF
                S=imgs.shape
                if S[0]!=0:
                    print('file: '+filename+'; filenumber : '+str(filno)+'; number of images : '+str(S[0])+'\n')
                    imgs0=np.concatenate((imgs0,imgs),axis=0)
                    speciesid0=np.concatenate((speciesid0,speciesid),axis=0)
            except:
                print(file+' failed')
            S2=imgs0.shape
            while S2[0]>batchsize:
                # Write slice to file
                imgs0_slice = imgs0[0:batchsize,:,:,:]
                speciesid0_slice = speciesid0[0:batchsize,:,:,:]
                print('Storing '+fld0+'batch'+str(year)+'_'+str(batchno)+'.npz\n')
                imgs0_slice_sh, speciesid0_slice_sh = shuffle(imgs0_slice, speciesid0_slice, random_state=0)
                np.savez(fld0+'batch'+str(year)+'_'+str(batchno),imgs=imgs0_slice_sh,speciesid=speciesid0_slice_sh,freqs=freqs)
                batchno+=1                
                # Keep remaining slice
                imgs0=imgs0[batchsize:,:,:,:]
                speciesid0=speciesid0[batchsize:,:,:]
                S2=imgs0.shape
                
    # Write remaining stuff        
    print('Storing '+fld0+'batch'+str(year)+'_'+str(batchno)+'.npz\n')
    np.savez(fld0+'batch'+str(year)+'_'+str(batchno),imgs=imgs0,speciesid=speciesid0,freqs=freqs)
                

