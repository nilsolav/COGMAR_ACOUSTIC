
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




#%% Get the model
K.clear_session()
freqs=4
model = md.model(freqs)

# File locations
if pl=='Linux':
    #fld = '/data/deep/data/echosounder/akustikk_all/data/DataOverview_North Sea NOR Sandeel cruise in Apr_May/'  
    fld = '/scratch/nilsolav/deep/data/echosounder/akustikk_all/data/DataOverview_North Sea NOR Sandeel cruise in Apr_May/images/'  
    fld2 = '/scratch/nilsolav/deep/data/echosounder/akustikk_all/data/DataOverview_North Sea NOR Sandeel cruise in Apr_May/modelparameters/'  

else:
    fld = 'D:\\data\\deep\\echosounder\\akustikk_all\\data\DataOverview_North Sea NOR Sandeel cruise in Apr_May\\'    
    fld2 = 'D:\\data\\deep\\echosounder\\akustikk_all\\data\DataOverview_North Sea NOR Sandeel cruise in Apr_May\\'

# Do da shit
flds = os.listdir(fld)
#%% Run 20 epochs
for epochs in np.arange(50).astype('int'):
    # Loop over batches
    for file in flds:
        if file.endswith(".npz"):
                print('Epoch:'+str(epochs)+'; file: '+file)
                try:
                    dat=np.load(fld+file)
                    batch_im = dat["imgs"]
                    batch_id = dat["speciesid"]
                    print(batch_id.shape)
                    print(batch_im.shape)
                    r = True
                except:
                    print('Failed to load: '+file)
                    r = False
                if r:
                    model.fit(batch_im,batch_id,verbose=0)
    # Get the weights
    dat=model.get_weights()
    mod=model.to_json()
    # Save the weights as numpy after each epoch
    np.savez(fld2+'modelweights_epoch_'+str(epochs)+'.npz',dat=dat,mod=mod)

#%% 
   




