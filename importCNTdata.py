# Intro to EEG Analysis

import os
import numpy as np
import mne
import matplotlib.pyplot as plt

# sample_data_folder = mne.datasets.sample.data_path()
# sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
#                                     'sample_audvis_filt-0-40_raw.fif')

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

# raw = mne.io.read_raw_fif(sample_data_raw_file)
eog_channels = ["HEOL", "HEOR", "VEOU", "VEOL"]
misc_channels = ["FT9", "FT10"]
# Bad channel O1?


montage = mne.channels.make_standard_montage("standard_1020")
# fig = montage.plot(kind='3d')
# fig.gca().view_init(azim=70, elev=15)  # set view angle
# montage.plot(kind='topomap', show_names=False)

raw = mne.io.read_raw_cnt("../data/3109.cnt", eog=eog_channels,
                          misc=misc_channels, preload=True)  # .filter(l_freq=1.0, h_freq=50.0)
print("Channel Names: ", raw.ch_names)
raw.rename_channels(channelNameMap)
print("Updated Channel Names for Montage: ", raw.ch_names)
# beware unlocated channels
raw.set_montage(montage=montage, on_missing='warn')
# raw_highpass = raw.copy().filter(l_freq=0.5, h_freq=None)

# plotting params
global_scalings = dict(mag=1e-12, grad=4e-11, eeg=100e-6, eog=150e-6, ecg=5e-4,
                       emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)

# for cutoff in (0.5, 1.0):
#     raw_highpass = raw.copy().filter(l_freq=cutoff, h_freq=None)
#     fig = raw_highpass.plot(duration=60, proj=False,
#                             n_channels=len(raw.ch_names), remove_dc=True, scalings=global_scalings)          #True ?
#     fig.subplots_adjust(top=0.9)
#     fig.suptitle('High-pass filtered at {} Hz'.format(cutoff), size='xx-large',
#                  weight='bold')


# Pick a highpass value, and create a raw object with it
global_highpass = 1.0
raw_highpass = raw.copy().filter(l_freq=global_highpass, h_freq=None)
# Pick a lowpass value and create a raw object with it from the raw highpass object
global_lowpass = 40.0
raw_lowpass = raw_highpass.copy().filter(l_freq=None, h_freq=global_lowpass)
filtered_data = raw_lowpass

filtered_data.set_eeg_reference(ref_channels=['A1', 'A2'])

print("Highpass and Lowpass completed, filtered data:", filtered_data)
print("Filtered data info field: ", filtered_data.info)

# Uses Matplotlib as 2D backend
# filtered_data.plot_psd(fmax=50)
# filtered_data.plot(duration=30, n_channels=len(filtered_data.ch_names), remove_dc=True, scalings=global_scalings)

event_dict = {
    'word seen once': 5,
    'word seen twice': 8,
    'test seen once': 10,
    'test seen twice': 15,
    'new words1': 18,
    'new words2': 20
}

#We would rather reject eyeblinks through ICA
# reject_criteria = dict(
# 	#mag=4000e-15,     # 4000 fT
#                     #    grad=4000e-13,    # 4000 fT/cm
#                        eeg=250e-6,       # 150 µV
#                        eog=500e-6)       # 250 µV

custom_mapping = {'seen_once_correct':"10", 'seen_twice_correct':"15", 
					'not_seen_correct':"18", 'not_seen_correct':"20",
                      'word_shown_once':"5", 'word_shown_twice':"8"}
reversed_custom_mapping = {value : key for (key, value) in custom_mapping.items()}


events_from_annot, event_dict = mne.events_from_annotations(filtered_data)
print("event dictionary: ", event_dict)
print("events from annotations: ", events_from_annot[:5])  # show the first 5

epochs = mne.Epochs(filtered_data, events_from_annot, event_id=event_dict,
                    tmin=-0.2, tmax=1.2, preload=True)		#Option to reject from criteria later if needed
print("epoch event id: ", epochs.event_id)
# epochs.plot(n_epochs=10, scalings=global_scalings)

# For below, you may need to pip install sklearn
# set up and fit the ICA

#Let's try doing the ICA on the epochs instead of filtered_data
filtered_data=epochs

ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(filtered_data)

# ica.plot_components(inst=filtered_data)

ica.exclude = [0, 1, 2, 4, 6]  # details on how we picked these are omitted here
# ica.plot_properties(filtered_data, picks=ica.exclude)

# Now let's apply some independent components analysis
orig_raw = filtered_data.copy()
filtered_data.load_data()
ica.apply(filtered_data)

print("applied the ICA... trying to plot clean epochs")

filtered_data.plot_image(picks=['Cz', 'Pz'])

epochs = filtered_data
conds_we_care_about = ['10', '15', '18']
epochs.equalize_event_counts(conds_we_care_about)  # this operates in-place
selected_epochs=epochs['10', '15', '18']
selected_epochs.plot_image(picks=['Cz', 'Pz'])



filtered_data.plot(n_epochs=10, scalings=global_scalings)

# show some frontal channels to clearly illustrate the artifact removal
frontal_channels = ["F7", "F3", "Fz", "F4", "F8", "FT7", "FC3"]
# chs = ['MEG 0111', 'MEG 0121', 'MEG 0131', 'MEG 0211', 'MEG 0221', 'MEG 0231',
#        'MEG 0311', 'MEG 0321', 'MEG 0331', 'MEG 1511', 'MEG 1521', 'MEG 1531',
#        'EEG 001', 'EEG 002', 'EEG 003', 'EEG 004', 'EEG 005', 'EEG 006',
#        'EEG 007', 'EEG 008']
chan_idxs = [filtered_data.ch_names.index(ch) for ch in frontal_channels]
# orig_raw.plot(order=chan_idxs, start=12, duration=4)
# filtered_data.plot(order=chan_idxs, start=12, duration=4)

filtered_data.plot(n_epochs=10, scalings=global_scalings)

#For next time: why are we not able to see the plot of all the epochs after the ICA
# We'd love to see that the new data is actually de-noised thanks to the ICA
# If it's not denoised, we may have to go back to ICA and remove more artifacts