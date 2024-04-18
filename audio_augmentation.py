import os

import torch
import torchaudio
import torchaudio.functional as F
import matplotlib.pyplot as plt
from vad import EnergyVAD


def load_data(data_folder):
    audio_data = []
    sample_rate = None
    for filename in os.listdir(data_folder):
        if not filename.endswith(".wav"):
            continue
        file_path = os.path.join(data_folder, filename)
        waveform, sr = torchaudio.load(file_path)
        audio_data.append(
            {
                "waveform": waveform,
                "sample_rate": sr,
                "filename": filename
            }
        )
        if sample_rate is None:
            sample_rate = sr
        assert sample_rate is None or sample_rate == sr
    return audio_data, sample_rate


def save_data(audio_data, output_folder, sample_rate):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for data in audio_data:
        filename, extension = os.path.splitext(data["filename"])
        file_path = os.path.join(output_folder, f"{filename}_augmented{extension}")
        torchaudio.save(file_path, data["waveform"], sample_rate)


def remove_beginning(audio_data, sample_rate):
    second = sample_rate
    for data in audio_data:
        data["waveform"] = data["waveform"][:, second:]
    return audio_data


def remove_dead_space(audio_data, sample_rate):
    vad = EnergyVAD(
        sample_rate=16000,
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
    audio_data, sample_rate = load_data(input_folder)

    # remove beginning of audio
    audio_data = remove_beginning(audio_data, sample_rate)
    # remove dead space in the middle of audio
    audio_data = remove_dead_space(audio_data, sample_rate)
    # save augmented data
    save_data(audio_data, output_folder, sample_rate)
