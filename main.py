import os

import matplotlib
import torch
import torchaudio
from sklearn.mixture import GaussianMixture

from audio_augmentation import clean_data, load_data, save_data

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, TimeMask, HighPassFilter, \
    LowPassFilter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_elipses():
    fig, ax = plt.subplots()
    for n, color in enumerate('rgbbb'):
        v, w = np.linalg.eigh(gmm.covariances_[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = matplotlib.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                         angle=180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.1)
        ax.add_artist(ell)

        ax.set_xlim(-25000, 25000)
        ax.set_ylim(-25000, 25000)


def plot_dims(a, b):
    fig, ax = plt.subplots()
    for n in range(n_speakers):
        data = x_train.data[y_train == n]
        plt.scatter(data[:500, a], data[:500, b], 0.8)
    plt.show()


def apply_effect(waveform, sample_rate):
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.25),
        TimeStretch(min_rate=0.8, max_rate=1.22, p=0.3),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.1),
        Shift(min_shift=-0.5, max_shift=0.5, p=0.05),
        HighPassFilter(min_cutoff_freq=300, max_cutoff_freq=800, p=0.1),
        LowPassFilter(min_cutoff_freq=1000, max_cutoff_freq=5000, p=0.1)
    ])
    w = augment(samples=waveform.numpy(), sample_rate=sample_rate)
    return torch.from_numpy(w)


def augment_data(input_folder, out_folder):
    audio_data, sample_rate = load_data(input_folder)

    for audio in audio_data:
        audio["waveform_orig"] = audio["waveform"]
    save_data(audio_data, out_folder, sample_rate)

    for i in range(10):
        for audio in audio_data:
            audio["waveform"] = apply_effect(audio["waveform_orig"], sample_rate)
            audio["augmented"] = i
        save_data(audio_data, out_folder, sample_rate)


def create_augmented_data(input_folder, clean_first=True):
    out_clean = input_folder.strip("/") + "_clean"
    if clean_first:
        clean_data(input_folder, out_clean)

    out_aug = input_folder.strip("/") + "_augmented"
    if os.path.exists(out_aug):
        # force remove directory
        os.system(f"rm -rf {out_aug}")

    augment_data(out_clean, out_aug)


def create_audio_data():
    create_augmented_data("data/training/non_target")
    create_augmented_data("data/training/target")

    create_augmented_data("data/validation/non_target")
    create_augmented_data("data/validation/target")


def load_audio_label(audio_data):
    x = []
    target_label = []
    for audio in audio_data:
        x.append(audio["waveform"])

        label = audio["filename"].split("_")[0][1:]
        target_label.append(label)

    # +1 because we reserve 0 for target
    sorted_labels = list(set(target_label))
    sorted_labels.sort()
    unique_dict = {v: k for k, v in enumerate(sorted_labels)}
    target_label = [unique_dict[label] for label in target_label]
    return x, np.array(target_label)


def mfcc_transform(audios, labels):
    mfcc_coefficients = torch.tensor([])
    labels_for_mfcc = torch.tensor([])

    transform_mfcc = torchaudio.transforms.MFCC(sample_rate, n_mfcc=20)

    for audio, label in zip(audios, labels):
        mfcc = transform_mfcc(audio)
        mfcc = torch.swapdims(mfcc, -1, -2)[0]
        mfcc_coefficients = torch.cat((mfcc_coefficients, mfcc), dim=0)
        labels_for_mfcc = torch.cat((labels_for_mfcc, torch.tensor([label] * mfcc.shape[0])), dim=0)

    return mfcc_coefficients, labels_for_mfcc


def prepare_loaded_data(non_target, target):
    audio, label = load_audio_label(non_target)
    label = label + 1
    assert np.all(label != 0)

    audio_target, label_target = load_audio_label(target)
    if not np.all(label_target == 0):
        raise ValueError("Target label is not 0:", label_target)

    mfccs, labels = mfcc_transform(audio_target + audio, np.concatenate([label_target, label]))
    return mfccs, labels


# create_audio_data()

train_non_target, sample_rate = load_data("data/training/non_target")
train_target, _ = load_data("data/training/target")
x_train_all, y_train_all = prepare_loaded_data(train_non_target, train_target)
x_train, x_train_target = x_train_all[y_train_all != 0], x_train_all[y_train_all == 0]
y_train, Y_train_target = y_train_all[y_train_all != 0], y_train_all[y_train_all == 0]


validation_non_target, _ = load_data("data/validation/non_target")
validation_target, _ = load_data("data/validation/target")
x_val_all, y_val_all = prepare_loaded_data(validation_non_target, validation_target)
x_val, x_val_target = x_val_all[y_val_all != 0], x_val_all[y_val_all == 0]
y_val, y_val_target = y_val_all[y_val_all != 0], y_val_all[y_val_all == 0]


validation_non_target_aug, _ = load_data("data/validation/non_target_augmented")
validation_target_aug, _ = load_data("data/validation/target_augmented")
x_val_aug_all, y_val_aug_all = prepare_loaded_data(validation_non_target_aug, validation_target_aug)
x_val_aug, x_val_target_aug = x_val_aug_all[y_val_aug_all != 0], x_val_aug_all[y_val_aug_all == 0]
y_val_aug, y_val_target_aug = y_val_aug_all[y_val_aug_all != 0], y_val_aug_all[y_val_aug_all == 0]

# x_train_all = PCA().fit_transform(x_train_all)
# x_val_all = PCA().fit_transform(x_val_all)
# x_val_aug_all = PCA().fit_transform(x_val_aug_all)
n_speakers = len(set(y_train.numpy()))
gmm = GaussianMixture(n_components=13, covariance_type="full",
                      verbose=1, verbose_interval=1, n_init=1)
gmm.fit(x_train)

gmm_target = GaussianMixture(n_components=5, covariance_type="full",
                             verbose=1, verbose_interval=1, n_init=1)
gmm_target.fit(x_train_target)
gmm_target.fit(x_train_target)


def get_accuracy(x, y_gt):

    nr_non_target_samples = len(y_gt[y_gt != 0])
    nr_target_samples = len(y_gt[y_gt == 0])

    apriori_non_target = nr_non_target_samples / (nr_non_target_samples + nr_target_samples)
    apriori_target = nr_target_samples / (nr_non_target_samples + nr_target_samples)

    log_likelihood = gmm.score_samples(x)  # + np.log(apriori_non_target)
    log_likelihood_target = gmm_target.score_samples(x)  # + np.log(apriori_target)

    y_train_pred = np.argmax(np.vstack([log_likelihood_target, log_likelihood]), axis=0)

    total_acc = np.mean(y_train_pred == y_gt) * 100
    target_acc = np.mean(y_train_pred[y_gt == 0] == 0) * 100
    non_target_acc = np.mean(y_train_pred[y_gt != 0] != 0) * 100
    return total_acc, target_acc, non_target_acc


print("Per MFCC accuracy:")
for dataset in ["train", "val", "val_aug"]:
    x, y = locals()[f"x_{dataset}_all"], locals()[f"y_{dataset}_all"]
    y_gt = np.array([yi if yi == 0 else 1 for yi in y])
    total_acc, target_acc, non_target_acc = get_accuracy(x, y_gt)
    print(
        f'\t{dataset} accuracy: {total_acc:0.2f}, target_acc: {target_acc:0.2f}, non-target_acc: {non_target_acc:0.2f}')

print("Per audio file accuracy:")
for val_audio in x_train_target:
    y_gt =
