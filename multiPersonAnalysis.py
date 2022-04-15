# Author: Keeton Martin

"""
This file will help to teach how to make one big dataframe, 
where each subject is a row, and each column is a channel's average value.
"""

#If one of these lines throws an error, try "pip install ..." on cmd line
#Imports
import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib.pyplot import show, plot
import os.path as op
import pandas as pd
from functools import reduce

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


filenames = ["3109.cnt", "3087.cnt", "3076.cnt"]
path_to_data = "./data/"
#Set up DataFrame to hold row per subject
aggregate_df_list = []

#Establish loop for 3 files
for filename in filenames:
	print("working on ", filename)
	this_person_row = dict()
	this_person_row["filename"] = filename



	#Read the CNT File
	raw = mne.io.read_raw_cnt(path_to_data+filename, eog=eog_channels,
                          misc=misc_channels, preload=True)  # .filter(l_freq=1.0, h_freq=50.0)
	print("Channel Names: ", raw.ch_names)
	raw.rename_channels(channelNameMap)
	print("Updated Channel Names for Montage: ", raw.ch_names)
	# beware unlocated channels
	raw.set_montage(montage=montage, on_missing='warn')

	annot_from_file = mne.read_annotations(path_to_data+filename)
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
	# fig = mne.viz.plot_events(events_from_annot, sfreq=raw.info['sfreq'], first_samp=raw.first_samp, event_id=event_dict)
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
	# epochs.plot_image(picks=['Cz', 'Pz'])

	epochs_fname = filename[:-4] + "-first-save-attempt-epo.fif"
	epochs.save(epochs_fname, overwrite=True)

	conds_we_care_about = ['10', '15', '18']
	epochs.equalize_event_counts(conds_we_care_about)  # this operates in-place
	selected_epochs=epochs['10', '15', '18']

	correct_familiar_words_epochs = epochs['10', '15']
	correct_unfamiliar_word_epochs = epochs['18']

	tens = epochs['10']
	fifteens = epochs['15']

	#Plot differences between familiar and unfamiliar words
	# familiar_evoked = correct_familiar_words_epochs.average()
	# unfamiliar_evoked = correct_unfamiliar_word_epochs.average()
	# evoked_diff = mne.combine_evoked([familiar_evoked, unfamiliar_evoked], weights=[1, -1])
	# evoked_diff.pick_types(include=['F3', 'F4', 'P3', 'P4']).plot_topo(color='r', legend=True)

	# tens_evoked = tens.average()
	# fifteens_evoked = fifteens.average()

	# evokeds = dict(familiar=familiar_evoked, unfamiliar=unfamiliar_evoked)
	# filtered_data.plot(n_epochs=10, scalings=global_scalings)

	# print(familiar_evoked)

	#Pick the channels we want to save average values for
	channels = ['F3', 'F4', 'P3', 'P4']
	selected_epochs=epochs['10', '15', '18']
	good_tmin, good_tmax = .3, .8

	evoked_attempt_1 = selected_epochs.crop(tmin=good_tmin, tmax=good_tmax).average(picks=channels, method="mean", by_event_type=True)

	# familiar_mean_roi = familiar_evoked.copy().pick(channels).crop(tmin=good_tmin, tmax=good_tmax)
	print("For ", filename, " we got : ", evoked_attempt_1)
	"""
	For  3109.cnt  we got :  <Evoked | '0.50 × 10 + 0.50 × 15' (average, N=66), 0.3 – 0.8 sec, baseline -0.2 – 0 sec (baseline period was cropped after baseline correction), 4 ch, ~40 kB>
	"""
	# tens_mean_roi = 

	print("Let's see that as a DataFrame though...\n")

	evoked_event_dfs = []

	subject_id = int(filename[:-4])

	#Basically, they really don't want you to do things this way so it takes some serious wrangling
	#They want you to just keep each dataframe seperate if you're seperating things on different event codes 
	for evoked_event_subset in evoked_attempt_1:
		print("Working on subject: ", subject_id)
		# subject_df = pd.DataFrame(evoked_attempt_1)
		subject_evoked_event_df = evoked_event_subset.to_data_frame()

		collapsed_subject_per_event = subject_evoked_event_df.mean(axis=0)
		collapsed_subject_per_event = collapsed_subject_per_event.to_dict()
		del collapsed_subject_per_event["time"]
		collapsed_subject_per_event_renamed_columns = {k+"for"+evoked_event_subset.comment: v for k, v in collapsed_subject_per_event.items()}
		# collapsed_subject_per_event_renamed_columns["subject"] = subject_id
		print("dictionary with subject attribute in there: ", collapsed_subject_per_event_renamed_columns)
		collapsed_subject_per_event_df = pd.DataFrame(collapsed_subject_per_event_renamed_columns, index=[subject_id])
		#Right now, indexing doesn't seem to work the way I want.

		print(collapsed_subject_per_event_df)
		evoked_event_dfs.append(collapsed_subject_per_event_df)

	merged_events_df = reduce(lambda left,right: pd.merge(left,right,left_index=True, right_index=True), evoked_event_dfs)
	# merged_events_df = merged_events_df.reindex

	print("Merged Events for one subject: \n", merged_events_df)

	# # subject_df = selected_epochs.to_data_frame()
	# print("Type of fam mean roi object: \n", familiar_mean_roi)
	# subject_df = familiar_mean_roi.to_data_frame()
	# print("\nmean roi df: ", subject_df, "\n")
	# # print("\n",subject_df.iloc[:5, :10],"\n")

	print("collapsed subject:\n", merged_events_df)
	print("\n collapsed subject type: ", type(merged_events_df))

	#Save this participant with average value to DF
	# collapsed_subject_clean = collapsed_subject[collapsed_subject.columns.difference(['time'])]
	aggregate_df_list.append(merged_events_df)


	"""
	We can have one row per person, where each column is the average value for a channel for a certain epoch
	3x13 array in this case
	"""

	"""
	For next week: Apr 14
	Split the 4 column dataframe into the 12 columns, one column per channel/response (see above)
	Explore automatic exclusion of certain bad components: https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html#using-an-eog-channel-to-select-ica-components
		The plots of components still comes up so we can identify which components we think are bad
		But the computer doesn't wait for user input, it still identifies the bad ones and continues on its own
	It might be easier to test the new functionality with just one user
	If there's time, try to add some configuration options for plots, perhaps command line arguments, loading from files

	
	"""


	#Loop to next participant

#Make a dataframe out of the list of dictionaries
# output_frame=pd.DataFrame.from_records(aggregate_df)
print(aggregate_df_list)
output_frame = pd.concat(aggregate_df_list)
print(output_frame)

output_frame.to_csv("output/firstFile.csv")

#small change to this file...