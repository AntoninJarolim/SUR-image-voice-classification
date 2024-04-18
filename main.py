from audio_augmentation import create_augmented_data

# load data files from data folder



# create_augmented_data(target_train)



# transform audio to MFCC features

# train GMM model

input_folder = "data/non_target_dev/"
output_folder = "data/non_target_dev_augmented/"

create_augmented_data(input_folder, output_folder)
