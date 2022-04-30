"""
Author: Keeton Martin
Configured with lots of help from mne documentation: https://mne.tools/stable/auto_tutorials/index.html
and with help from Anjali Thapar: https://www.brynmawr.edu/inside/people/anjali-thapar

If you have questions about this file or it's throwing errors or you're generally confused, 
try the README in the parent directory.
"""

import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
import os.path as op
import pandas as pd
from functools import reduce

#Do you want to see plots when you run this file?
plotsEnabled = False
filename = "3109.cnt"
path_to_data = "../data/"
desired_ICA_components = 20
events_we_care_about = ['10', '15', '18']
channels_we_care_about = ['F3', 'F4', 'P3', 'P4']
good_tmin, good_tmax = .3, .8



#Some of the channels have capitalization differences from what we might expect in our cnt data
channelNameMap = {
    'FP1': 'Fp1',
    'FP2': 'Fp2',
    'FZ': 'Fz',
    'FCZ': 'FCz',
    'CZ': 'Cz',
    'CPZ': 'CPz',
    'PZ': 'Pz',
    'OZ': 'Oz'
}

#We also want to label certain channel types
eog_channels = ["HEOL", "HEOR", "VEOU", "VEOL"]
misc_channels = ["FT9", "FT10"]

#Let's now use the right cap structure (montage)
montage = mne.channels.make_standard_montage("standard_1020")

#Establish scalings to be used for filtering
global_scalings = dict(mag=1e-12, grad=4e-11, eeg=100e-6, eog=150e-6, ecg=5e-4,
                       emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)
global_highpass = 1.0
global_lowpass = 40.0

#Time to import data
print("Importing data from ", filename, " file located in ", path_to_data)
raw = mne.io.read_raw_cnt(path_to_data+filename, eog=eog_channels, misc=misc_channels, preload=True)

raw.rename_channels(channelNameMap)
print("Channel Names: ", raw.ch_names)

raw.set_montage(montage=montage, on_missing='warn')

#Bring in the annotations
annot_from_file = mne.read_annotations(path_to_data+filename)
print("\nAnnotations:\n", annot_from_file, "\n")
raw.set_annotations(annot_from_file)

#Filter the data
raw_highpass = raw.copy().filter(l_freq=global_highpass, h_freq=None)
raw_lowpass = raw_highpass.copy().filter(l_freq=None, h_freq=global_lowpass)
filtered_data = raw_lowpass

#Set reference channels
filtered_data.set_eeg_reference(ref_channels=['A1', 'A2'])

print("Highpass and Lowpass completed, filtered data:", filtered_data)
print("Filtered data info field: ", filtered_data.info)

ui_answerOne = input("Next we will epoch the data. Before we continue, does everything look ok? ")


#Get events from annotations
print("Finding annotations...")
events_from_annot, event_dict = mne.events_from_annotations(filtered_data)
print("event dictionary: ", event_dict)
print("events from annotations: ", events_from_annot[:5])  # show the first 5

#Seperate data into epochs
epochs = mne.Epochs(filtered_data, events_from_annot, event_id=event_dict, tmin=-0.2, tmax=1.2, preload=True)		#Option to reject from criteria later if needed
print("epoch event id: ", epochs.event_id)
filtered_data=epochs 		#TODO: Consider re-naming this var

print("performing ICA, please be patient...")
#Perform ICA, and use consistent random state for consistent components between runs
ica = mne.preprocessing.ICA(n_components=desired_ICA_components, random_state=97, max_iter=800)
ica.fit(filtered_data)

ica.exclude = [0, 1, 2, 4, 6]  #These were hand selected

#Now we know which to exclude, we can reapply the data
orig_raw = filtered_data.copy()
filtered_data.load_data()
ica.apply(filtered_data)

#Plot newly cleaned (ica'd) epochs
print("applied the ICA... trying to plot clean epochs")
epochs = filtered_data

if plotsEnabled: epochs.plot_image(picks=['Cz', 'Pz'])

input("Saving epochs as epochs.fif, and overwriting old version if necessary. Ok?")
epochs_fname = filename[:-4] + "_epochs.fif"
epochs.save(epochs_fname, overwrite=True)

epochs.equalize_event_counts(events_we_care_about)  # this operates in-place
selected_epochs=epochs[events_we_care_about]

#Split into two categories if you'd like
correct_familiar_words_epochs = selected_epochs['10', '15']
correct_unfamiliar_word_epochs = selected_epochs['18']
familiar_evoked = correct_familiar_words_epochs.average()
unfamiliar_evoked = correct_unfamiliar_word_epochs.average()

#There are many options for manipulating the data to get a spreadsheet that you want
familiar_evoked_df = familiar_evoked.to_data_frame()
print(familiar_evoked_df)

familiar_mean_roi = familiar_evoked.copy().pick(channels_we_care_about).crop(tmin=good_tmin, tmax=good_tmax)
print(familiar_mean_roi)
familiar_mean_roi_df = familiar_mean_roi.to_data_frame()
print(familiar_mean_roi_df)

mean_amp_roi = familiar_mean_roi.data.mean(axis=1) * 1e6
print(mean_amp_roi)


#In this example we'll choose this one to save
familiar_mean_roi_df.to_csv("output.csv")








