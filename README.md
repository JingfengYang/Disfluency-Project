# Disfluency Project
NXT Switchboard xml parsing script

----------------------------------
Requirement:
python 3.6+
pytorch 1.0+


----------------------------------
Description:

file_list.txt is a list of file indices.

parse_all.py
This is the main script to retrieve plain text information
from corresponding folders
One dialogue per running.
This script will take in one argument, which is the unique 
index of the dialogue file (e.g. sw2005)

parse_all.sh
This is the bash script to run parse_all.py through all 
dialogue files, and list result in plain text with the index
of the dialogue(e.g. sw2012)

parse_all.py This is the main script to retrieve plain text information from corresponding folders One dialogue per running. This script will take in one argument, which is the unique index of the dialogue file (e.g. sw2005)

utils.py
Include functions used in train.py

preprocessPTB.py
This is to read the data and finish the preprocessing (delete punctuations, partial words), transforms the labels to BE IE IP BE_IP O OR.

train.py
This is the main file to train vanilla BiLSTM model, test on development and test set.

----------------------------------
How to run:

First, put the dps/swbd/2 dps/swbd/3 dps/swbd/4 folder in PTB III to the folder dps/swbd/ directory.

Second, 
cd dps/swbd
run ./mv.sh

Finally, cd main directory
run python train.py

