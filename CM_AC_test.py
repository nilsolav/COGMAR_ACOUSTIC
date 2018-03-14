# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:45:49 2018

@author: Administrator
"""
#%% This script imports the models and make a test run on one raw/snap file

#activate tensorflow
import os
os.chdir('D:\\repos\\Github\\COGMAR_ACOUSTIC')
os.environ["PATH"] += os.pathsep + 'C:\\ProgramData\\Anaconda3\\pkgs\\graphviz-2.38.0-4\\Library\\bin\\graphviz'
import numpy as np
import matplotlib.pyplot as plt

#'D:\\data\\deep\\echosounder\\akustikk_all\\data\\DataOverview_North Sea NOR Sandeel cruise in Apr_May\\2008\\2008205-D20080426-T121128.mat'


#%% Test figures for each year from file 2 and image k1
k1=6
for year in range(2006,2017):
    # Read the reshaped files (from CM_AM_reshapedata.py)
    # Get file number 2
    fil= r'D:\data\deep\echosounder\akustikk_all\data\DataOverview_North Sea NOR Sandeel cruise in Apr_May\batch'+str(year)+'_2.npz'
    mat = np.load(fil)
    speciesid=mat["speciesid"]
    imgs=mat["imgs"]
    freqs=mat["freqs"]
    # Four axes, returned as a 2-d array
    f, axarr = plt.subplots(2, 3)
    axarr[0, 0].imshow(10*np.log10(np.squeeze(imgs[k1,0,:,:])),extent =[-100,-20,0,1], cmap='hot', aspect='auto')
    axarr[0, 0].set_title(str(freqs[0])+'kHz')
    axarr[0, 1].imshow(10*np.log10(np.squeeze(imgs[k1,1,:,:])),extent =[-100,-20,0,1], cmap='hot', aspect='auto')
    axarr[0, 1].set_title(str(freqs[1])+'kHz')
    axarr[0, 2].imshow(10*np.log10(np.squeeze(imgs[k1,2,:,:])),extent =[-100,-20,0,1], cmap='hot', aspect='auto')
    axarr[0, 1].set_title(str(freqs[2])+'kHz')
    axarr[1, 0].imshow(10*np.log10(np.squeeze(imgs[k1,3,:,:])),extent =[-100,-20,0,1], cmap='hot', aspect='auto')
    axarr[1, 0].set_title(str(freqs[3])+'kHz')
    axarr[1, 1].imshow(np.squeeze(speciesid[k1,:,:]),  aspect='auto')
    axarr[1, 1].set_title('ID map')
    
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_xticklabels() for a in axarr[1, :]], visible=False)
    #plt.setp([a.get_xticklabels() for a in axarr[2, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 0]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 2]], visible=False)
    #plt.show()
    plt.savefig('Figure'+str(year)+'_subset.png')
    
    
#%% Vizualize cost function (include weighing)
year=2012
fil= r'D:\data\deep\echosounder\akustikk_all\data\DataOverview_North Sea NOR Sandeel cruise in Apr_May\batch'+str(year)+'_2.npz'
mat = np.load(fil)
speciesid=mat["speciesid"]
imgs=mat["imgs"]
freqs=mat["freqs"]
#%%

#%%
smooth=1

def dice_coef(y_true, y_pred,weights):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    return K.categorical_crossentropy(y_pred, y_true) * final_mask
#%%
k1=6
k2=12
smooth = 1
y_pred = np.squeeze(speciesid[k1,:,:])
y_true = np.squeeze(speciesid[k1,:,:])*0
intersection = np.sum(np.multiply(y_pred,y_true))
dice = (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
print(dice)

#def dice_coef_loss_num(y_true, y_pred,weights):
#    return -dice_coef_num(y_true, y_pred)
#
#test=dice_coef_loss_num(, np.squeeze(speciesid[k2,:,:]))
#print(test)

#%% Choose and fit model
model = md.model1(freqs)
plot_model(model, show_shapes ='True', show_layer_names = 'True', to_file='model1.png')
#model = md.model2()
#model = md.model3()

    
#%%
model.fit(imgs,speciesid)

#%% Validate model
labels_test_pred = model.predict_on_batch(imgs)

for i in range(0,22):
    dum2 = np.squeeze(speciesid[i,:,:])
    dum3 = np.squeeze(labels_test_pred[i,:,:])
    #plt.imshow(10*np.log10(dum),extent =[-150,-60,0,1], cmap='hot', aspect='auto')
    #plt.show
    plt.figure(i)    
    plt.imshow(dum3)#,extent =[0,2,0,1], cmap='hot', aspect='auto')
    plt.show
    
    #plt.figure(i+1)    
    #plt.imshow(dum2)#,extent =[0,2,0,1], cmap='hot', aspect='auto')
    #plt.show

