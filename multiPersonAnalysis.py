# Author: Keeton Martin

"""
This file will help to teach how to make one big dataframe, 
where each subject is a row, and each column is a channel's average value.
"""

#Imports
import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import os.path as op
import pandas as pd

#Configure Channels
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

eog_channels = ["HEOL", "HEOR", "VEOU", "VEOL"]
misc_channels = ["FT9", "FT10"]

montage = mne.channels.make_standard_montage("standard_1020")
# fig = montage.plot(kind='3d') #Uncomment this line to see where locations of sensors are

#Establish scalings to be used for filtering
global_scalings = dict(mag=1e-12, grad=4e-11, eeg=100e-6, eog=150e-6, ecg=5e-4,
                       emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)
global_highpass = 1.0
global_lowpass = 40.0


filenames = ["3109.cnt", "3087.cnt", "3109.cnt"]
path_to_data = "../data/"
#Set up DataFrame to hold row per subject
person_channels_list = []

#Establish loop for 3 files
for filename in filenames:
	print("working on ", filename)
	this_person_row = dict()
	this_person_row["filename"] = filename



	#Read the CNT File
	raw = mne.io.read_raw_cnt("../data/3109.cnt", eog=eog_channels,
                          misc=misc_channels, preload=True)  # .filter(l_freq=1.0, h_freq=50.0)
	print("Channel Names: ", raw.ch_names)
	raw.rename_channels(channelNameMap)
	print("Updated Channel Names for Montage: ", raw.ch_names)
	# beware unlocated channels
	raw.set_montage(montage=montage, on_missing='warn')

	annot_from_file = mne.read_annotations('../data/3109.cnt')
	print("\n", annot_from_file, "\n")
	raw.set_annotations(annot_from_file)

	#Use selected highpass and lowpass values to filter
	raw_highpass = raw.copy().filter(l_freq=global_highpass, h_freq=None)
	raw_lowpass = raw_highpass.copy().filter(l_freq=None, h_freq=global_lowpass)
	filtered_data = raw_lowpass

	#Set reference channels
	filtered_data.set_eeg_reference(ref_channels=['A1', 'A2'])

	print("Highpass and Lowpass completed, filtered data:", filtered_data)
	print("Filtered data info field: ", filtered_data.info)

	#Get events from annotations
	print("Finding annotations...")
	events_from_annot, event_dict = mne.events_from_annotations(filtered_data)
	print("event dictionary: ", event_dict)
	print("events from annotations: ", events_from_annot[:5])  # show the first 5

	#See when the events are happening:
	fig = mne.viz.plot_events(events_from_annot, sfreq=raw.info['sfreq'], first_samp=raw.first_samp, event_id=event_dict)
	# fig.subplots_adjust(right=0.7)  # make room for legend


	#Seperate data into epochs
	epochs = mne.Epochs(filtered_data, events_from_annot, event_id=event_dict, tmin=-0.2, tmax=1.2, preload=True)		#Option to reject from criteria later if needed
	print("epoch event id: ", epochs.event_id)
	filtered_data=epochs 		#TODO: Consider re-naming this var

	#Perform ICA
	ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
	ica.fit(filtered_data)

	#Certain components were excluded for the first filename based on characteristics...
	#Are we sure we still want to drop these components for every participant?...
	ica.exclude = [0, 1, 2, 4, 6]  # details on how we picked these are omitted here

	orig_raw = filtered_data.copy()
	filtered_data.load_data()
	ica.apply(filtered_data)

	#Plot newly cleaned epochs
	print("applied the ICA... trying to plot clean epochs")
	epochs = filtered_data
	epochs.plot_image(picks=['Cz', 'Pz'])

	epochs_fname = filename[:-4] + "-first-save-attempt-epo.fif"
	epochs.save(epochs_fname, overwrite=True)

	conds_we_care_about = ['10', '15', '18']
	epochs.equalize_event_counts(conds_we_care_about)  # this operates in-place
	selected_epochs=epochs['10', '15', '18']

	correct_familiar_words_epochs = epochs['10', '15']
	correct_unfamiliar_word_epochs = epochs['18']

	#Plot differences between familiar and unfamiliar words
	familiar_evoked = correct_familiar_words_epochs.average()
	unfamiliar_evoked = correct_unfamiliar_word_epochs.average()
	evoked_diff = mne.combine_evoked([familiar_evoked, unfamiliar_evoked], weights=[1, -1])
	evoked_diff.pick_types(include=['F3', 'F4', 'P3', 'P4']).plot_topo(color='r', legend=True)

	evokeds = dict(familiar=familiar_evoked, unfamiliar=unfamiliar_evoked)
	filtered_data.plot(n_epochs=10, scalings=global_scalings)

	#Pick the channels we want to save average values for
	channels = ['F3', 'F4', 'P3', 'P4']
	selected_epochs=epochs['10', '15', '18']
	good_tmin, good_tmax = .3, .8

	familiar_mean_roi = familiar_evoked.copy().pick(channels).crop(tmin=good_tmin, tmax=good_tmax)
	print("For ", filename, " we got : ", familiar_mean_roi)
	"""
	For  3109.cnt  we got :  <Evoked | '0.50 × 10 + 0.50 × 15' (average, N=66), 0.3 – 0.8 sec, baseline -0.2 – 0 sec (baseline period was cropped after baseline correction), 4 ch, ~40 kB>
	"""



	#Save this participant with average value to DF

	#Loop to next participant

#Make a dataframe out of the list of dictionaries