import os.path

import torch
import torchaudio
from sklearn.mixture import GaussianMixture, GaussianMixture
from audio_augmentation import load_data, create_audio_data, clean_data
import numpy as np
from pathlib import Path

sample_rate = 8000


def transform_mfcc(audio):
    transformator = torchaudio.transforms.MFCC(sample_rate, n_mfcc=13)
    mfcc = transformator(audio)
    return mfcc[0]


def add_features(audio_list):
    for audio in audio_list:
        audio["mfcc"] = transform_mfcc(audio["waveform"])
        audio["mfcc_d1"] = torchaudio.functional.compute_deltas(audio["mfcc"])
        audio["mfcc_d2"] = torchaudio.functional.compute_deltas(audio["mfcc_d1"])

        # pitch = torchaudio.functional.detect_pitch_frequency(audio["waveform"], sample_rate)
        # audio["pitch"] = pitch[:, :audio["mfcc"].shape[1]]

        audio["features"] = torch.vstack([audio["mfcc"]])  # , audio["mfcc_d1"], audio["mfcc_d2"]], )
        audio["features"] = torch.swapdims(audio["features"], -1, -2)
        pass


def load_data_features(path, clean_first=False):
    if clean_first:
        path_clean = path + "_clean"
        clean_data(path, path_clean)
        path = path_clean
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

        audio["confidence"] = np.mean(np.argmax(np.vstack([log_likelihood, log_likelihood_target]), axis=0))

        if np.sum(log_likelihood) < np.sum(log_likelihood_target):
            predict_target += 1
            audio["predict_target"] = True
        else:
            audio["predict_target"] = False

    return predict_target / len(audio_list)


def train_gmms():
    train_non_target = load_data_features("data/training/non_target_augmented")
    train_target = load_data_features("data/training/target_augmented")

    train_non_target_clean = load_data_features("data/training/non_target")
    train_target_clean = load_data_features("data/training/target")

    validation_non_target = load_data_features("data/validation/non_target_clean")
    validation_target = load_data_features("data/validation/target_clean")

    validation_non_target_aug = load_data_features("data/validation/non_target_augmented")
    validation_target_aug = load_data_features("data/validation/target_augmented")

    gmm = GaussianMixture(n_components=30, covariance_type="full",
                                  verbose=1, verbose_interval=1, n_init=1, max_iter=150)
    train_gmm(gmm, train_non_target)

    gmm_target = GaussianMixture(n_components=15, covariance_type="full",
                                         verbose=1, verbose_interval=1, n_init=1, max_iter=150)
    train_gmm(gmm_target, train_target)

    print("parameter for gmm non target:")
    print(gmm.weights_)

    print("parameter for gmm target:")
    print(gmm_target.weights_)

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
    print(
        f"val accuracy nontarget-split-train-clean: {1 - per_audio_accuracy(train_non_target_clean, gmm, gmm_target):0.2f}")
    print()

    save_gmm("gmm-target", gmm_target)
    save_gmm("gmm-non-target", gmm)


def save_gmm(gmm_name, gmm):
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    gmm_path = os.path.join(models_dir, gmm_name)
    np.save(gmm_path + '_weights', gmm.weights_, allow_pickle=False)
    np.save(gmm_path + '_means', gmm.means_, allow_pickle=False)
    np.save(gmm_path + '_covariances', gmm.covariances_, allow_pickle=False)


def load_gmm(gmm_name):
    models_dir = "models"

    gmm_path = os.path.join(models_dir, gmm_name)
    means = np.load(gmm_path + '_means.npy')
    covar = np.load(gmm_path + '_covariances.npy')
    loaded_gmm = GaussianMixture(n_components=len(means), covariance_type='full')
    loaded_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covar))
    loaded_gmm.weights_ = np.load(gmm_path + '_weights.npy')
    loaded_gmm.means_ = means
    loaded_gmm.covariances_ = covar
    return loaded_gmm


def classify_audio(data_path: Path):
    gmm_target = load_gmm("gmm-target")
    gmm = load_gmm("gmm-non-target")

    test_data = load_data_features(str(data_path), clean_first=True)

    per_audio_accuracy(test_data, gmm, gmm_target)

    for audio in test_data:
        name = audio["filename"].strip(".wav")
        confidence = audio["confidence"]
        predict = int(audio["predict_target"])

        print(f"{name} {confidence} {predict}")
