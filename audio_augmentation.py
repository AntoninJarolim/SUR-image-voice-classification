import os

import torch
import torchaudio
from vad import EnergyVAD

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, TimeMask, HighPassFilter, \
    LowPassFilter


def load_data(data_folder, raw=False, return_sr=False):
    audio_data = []
    sample_rate = None
    for filename in os.listdir(data_folder):
        if not filename.endswith(".wav"):
            continue
        file_path = os.path.join(data_folder, filename)
        waveform, sr = torchaudio.load(file_path)
        if raw is False:
            audio_data.append(
                {
                    "waveform": waveform,
                    "sample_rate": sr,
                    "filename": filename
                }
            )
        else:
            audio_data.append(waveform)
        if sample_rate is None:
            sample_rate = sr
        assert sample_rate is None or sample_rate == sr
    if return_sr:
        return audio_data, sample_rate
    return audio_data


def save_data(audio_data, output_folder, sample_rate):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for data in audio_data:
        filename, extension = os.path.splitext(data["filename"])
        try:
            filename = f"{filename}_aug{data['augmented']}"
        except KeyError:
            pass
        file_path = os.path.join(output_folder, f"{filename}{extension}")
        torchaudio.save(file_path, data["waveform"], sample_rate)


def remove_beginning(audio_data, sample_rate):
    second = sample_rate
    for data in audio_data:
        data["waveform"] = data["waveform"][:, second:]
    return audio_data


def remove_dead_space(audio_data, sample_rate):
    vad = EnergyVAD(
        sample_rate=sample_rate,
        frame_length=150,  # in milliseconds
        frame_shift=20,  # in milliseconds
        energy_threshold=0.04,  # you may need to adjust this value
        pre_emphasis=0.95,
    )
    for data in audio_data:
        data["waveform"] = torch.from_numpy(vad.apply_vad(data["waveform"]))
    return audio_data


def clean_data(input_folder, output_folder):
    # load data files from data folder
    audio_data, sample_rate = load_data(input_folder, return_sr=True)

    # remove beginning of audio
    audio_data = remove_beginning(audio_data, sample_rate)
    # remove dead space in the middle of audio
    audio_data = remove_dead_space(audio_data, sample_rate)
    # save augmented data
    save_data(audio_data, output_folder, sample_rate)


def apply_effect(waveform, sample_rate):
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.25),
        TimeStretch(min_rate=0.8, max_rate=1.22, p=0.3),
        Shift(min_shift=-0.5, max_shift=0.5, p=0.05),
        HighPassFilter(min_cutoff_freq=300, max_cutoff_freq=800, p=0.1),
        LowPassFilter(min_cutoff_freq=1000, max_cutoff_freq=5000, p=0.1)
    ])
    w = augment(samples=waveform.numpy(), sample_rate=sample_rate)
    return torch.from_numpy(w)


def augment_data(input_folder, out_folder):
    audio_data, sample_rate = load_data(input_folder, return_sr=True)

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
