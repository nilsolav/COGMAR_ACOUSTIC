
#%% Train the network per year
#for year in range(2007,2016):
import os
import numpy as np
import scipy.io as spio
from keras import backend as K
import platform

pl=platform.system()

if pl=='Linux':
    os.chdir('/nethome/nilsolav/repos/github/COGMAR_ACOUSTIC')
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
else:
    os.chdir('D:\\repos\\Github\\COGMAR_ACOUSTIC')

import CM_AC_models as md

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

#%% Get the model
freqs=4
model = md.model1(freqs)
K.clear_session()
# Do da shit
for year in range(2008,2009):
    if pl=='Linux':
        fld = '/data/deep/data/echosounder/akustikk_all/data/DataOverview_North Sea NOR Sandeel cruise in Apr_May/'  
    else:
        fld = 'D:\\data\\deep\\echosounder\\akustikk_all\\data\DataOverview_North Sea NOR Sandeel cruise in Apr_May\\'
 
    fld+=str(year)
    flds = os.listdir(fld)
    for file in flds[12:30]:
        if file.endswith(".mat"):
            filename, file_extension = os.path.splitext(file) 
            #try:
            if pl=='Linux':
                filefld = fld+'/'+filename  
            else:
                filefld = fld+'\\'+filename  
            imgs,speciesid = gettrainingset(filefld,freqs)
            print(imgs.shape)
            print(speciesid.shape)
            # TRaining step
            if imgs.size>0:
                #break
                #model = md.model1(freqs)
                # Dette fungerer ikkje. Men dersom eg lagar ein ny modell (L56) for kvar rundeså går det fint.
                model.fit(imgs,speciesid)
                print(filefld+' OK')
            else:
                print(filefld+' No masks')
            #except:
            #print(filefld+' failed')







