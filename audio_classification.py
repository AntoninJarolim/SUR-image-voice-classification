import torch
import torchaudio
from sklearn.mixture import GaussianMixture
from audio_augmentation import load_data, create_audio_data
import numpy as np


sample_rate = 16000

def transform_mfcc(audio):
    transformator = torchaudio.transforms.MFCC(sample_rate, n_mfcc=20)
    mfcc = transformator(audio)
    return mfcc[0]


def add_features(audio_list):
    for audio in audio_list:
        audio["mfcc"] = transform_mfcc(audio["waveform"])

        pitch = torchaudio.functional.detect_pitch_frequency(audio["waveform"], sample_rate)
        audio["pitch"] = pitch[:, :audio["mfcc"].shape[1]]

        audio["features"] = torch.vstack([audio["mfcc"], audio["pitch"]], )
        audio["features"] = torch.swapdims(audio["features"], -1, -2)
        # audio["features"] = audio["mfcc"]
        pass


def load_data_features(path):
    data_features = load_data(path)
    add_features(data_features)
    return data_features


def train_gmm(gmm_to_train, audio_list):
    mfcc_coefficients = torch.tensor([])
    for audio in audio_list:
        mfcc = audio["features"]
        mfcc_coefficients = torch.cat((mfcc_coefficients, mfcc), dim=0)
    gmm_to_train.fit(mfcc_coefficients.numpy())


def per_audio_accuracy(audio_list, gmm, gmm_target):
    predict_target = 0
    for audio in audio_list:
        feature = audio["features"]
        log_likelihood = gmm.score_samples(feature)
        log_likelihood_target = gmm_target.score_samples(feature)
        if np.sum(log_likelihood) < np.sum(log_likelihood_target):
            predict_target += 1

    return predict_target / len(audio_list)


def classify_audio():
    train_non_target = load_data_features("data/training/non_target_augmented")
    train_target = load_data_features("data/training/target_augmented")

    train_non_target_clean = load_data_features("data/training/non_target")
    train_target_clean = load_data_features("data/training/target")

    validation_non_target = load_data_features("data/validation/non_target_clean")
    validation_target = load_data_features("data/validation/target_clean")

    validation_non_target_aug = load_data_features("data/validation/non_target_augmented")
    validation_target_aug = load_data_features("data/validation/target_augmented")

    gmm = GaussianMixture(n_components=13, covariance_type="full",
                              verbose=1, verbose_interval=1, n_init=1)
    train_gmm(gmm, train_non_target)

    gmm_target = GaussianMixture(n_components=5, covariance_type="full",
                                 verbose=1, verbose_interval=1, n_init=1)
    train_gmm(gmm_target, train_target)

    print(f"train accuracy target   : {per_audio_accuracy(train_target, gmm, gmm_target):0.2f}")
    print(f"train accuracy nontarget: {1 - per_audio_accuracy(train_non_target, gmm, gmm_target):0.2f}")
    print()

    print(f"val accuracy target   : {per_audio_accuracy(validation_target, gmm, gmm_target):0.2f}")
    print(f"val accuracy nontarget: {1 - per_audio_accuracy(validation_non_target, gmm, gmm_target):0.2f}")
    print()

    print(f"val accuracy aug target   : {per_audio_accuracy(validation_target_aug, gmm, gmm_target):0.2f}")
    print(f"val accuracy aug nontarget: {1 - per_audio_accuracy(validation_non_target_aug, gmm, gmm_target):0.2f}")
    print()

    print(f"val accuracy target-split-train-clean   : {per_audio_accuracy(train_target_clean, gmm, gmm_target):0.2f}")
    print(f"val accuracy nontarget-split-train-clean: {1 - per_audio_accuracy(train_non_target_clean, gmm, gmm_target):0.2f}")
    print()



