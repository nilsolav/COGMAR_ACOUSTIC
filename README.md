# COGMAR_ACOUSTIC (CM_AC)
Code for the COGMAR acoustic case.

# Testing the preprocessor [matlab]
The CM_AC_test.m script loads a local .raw and .work file and tests the preprocessing
generation.

# Preprocessing of data [matlab]
The CM_AC_main.m is a matlabscript that reads the .raw and .work files from the 
IMR data storage that are run on the unix server. Mat files are stored in the 
data deep folder under /data/deep/data/echosounder/akustikk_all/data/.

# Reshaping the data
The CM_AC_reshape.py reads the mat data, reshape it, shuffles it by year and write one 
file per batch. If frequencies are missing they are replaced with zeros.

# Setting up the models [python]
The CM_AC_models.py contains the different models and can be called both from the
test script and the server script.

# Testing the models [python]
The CM_AC_test.py script loads a local .raw and .work file run the file as one batch
for testing the models. [REWRITE TO PRODUCE PAPER FIGURES]

# Training the model on the server [python]
The CM_AC_runtraining.py runs on the unix server to train the models on all the data.

