import argparse

from audio_augmentation import create_audio_data
from audio_classification import classify_audio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classification of images and voices of a person.')
    parser.add_argument('--augment-audio', help='Creates audio augmented data.', action='store_true')
    parser.add_argument('--classify', help='Performs classification', action='store_true')

    args = vars(parser.parse_args())

    if args["augment_audio"]:
        create_audio_data()

    if args["classify"]:
        classify_audio()
