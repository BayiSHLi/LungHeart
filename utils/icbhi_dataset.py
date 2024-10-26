import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import random
import math

import torch
import torchaudio
from torchaudio import transforms as T
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy

from transformers import ClapProcessor

Processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused", sampling_rate=48000)

class ICBHIDataset(Dataset):

    def __init__(self, data_root, mode, sample_rate, desired_length, pad_types, fade_samples_ratio, transform=None):
        assert mode in ['train', 'test'], "mode should be 'train' or 'test'"

        self.data_root = data_root
        self.mode = mode
        self.sample_rate = sample_rate
        self.data = []
        self.desired_length = desired_length
        self.fade_samples_ratio = fade_samples_ratio
        self.pad_types = pad_types
        self.transform = transform

        self.split_file = os.path.join(data_root, 'official_split.txt')
        self.diagnosis_file = os.path.join(data_root, 'diagnosis.txt')
        self.meta_file = os.path.join(data_root, 'metadata.txt')

        self.diagnosis2label = {'Healthy': 0, 'COPD': 1, 'Bronchiectasis': 2, 'Asthma': 3, 'URTI': 4,
                          'LRTI': 5, 'Pneumonia': 6, 'Bronchiolitis': 7}

        diagnosis = pd.read_csv(self.diagnosis_file,
                                names=['diagnosis'],
                                delimiter='\t')

        meta  = pd.read_csv(self.meta_file,
                                names=['age', 'sex', 'adult_BMI', 'child_weight', 'child_height', 'chest_location'],
                                delimiter='\t')

        self.patient_info = pd.concat([meta, diagnosis], axis=1)

        self.data = []
        # Read the split file and filter samples based on mode
        with open(self.split_file, 'r') as file:
            for line in file:
                name, sample_mode = line.strip().split()
                if sample_mode == self.mode:
                    patient, index, chest_location, Ac_mode, device = name.split('_')
                    age, sex, BMI, c_weight, c_height, _, diagnosis = self.patient_info.loc[int(patient)]

                    txt_file = os.path.join(data_root, name + '.txt')
                    annotations = pd.read_csv(txt_file, names=['Start', 'End', 'Crackles', 'Wheezes'],
                                                   delimiter='\t')

                    wav_file = os.path.join(data_root, name + '.wav')
                    wav_data, sr = torchaudio.load(wav_file)
                    fade_samples = int(sample_rate / fade_samples_ratio)
                    fade = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape='linear')
                    wav_data = fade(wav_data)

                    if sr != sample_rate:
                        resample = T.Resample(sr, sample_rate)
                        wav_data = resample(wav_data)

                    for idx in annotations.index:
                        row = annotations.loc[idx]
                        start = row['Start']  # start time (second)
                        end = row['End']  # end time (second)
                        audio_chunk = self.slice_audio(start, end, wav_data)

                        crackles = row['Crackles']
                        wheezes = row['Wheezes']
                        label = (crackles, wheezes, self.diagnosis2label[diagnosis])
                        meta_info = f"Patient is a {age}-year-old {sex}. Adult BMI: {BMI}. Child weight: {c_weight} kg, " \
                                    f"height: {c_height} cm. Chest location: {chest_location}. Device: {device}."

                        self.data.append((self.padding_audio(audio_chunk),
                                          label,
                                          meta_info))


    def padding_audio(self, wav_data):
        fade_samples = int(self.sample_rate / self.fade_samples_ratio)
        fade_out = T.Fade(fade_in_len=0, fade_out_len=fade_samples, fade_shape='linear')
        target_duration = self.desired_length * self.sample_rate

        if wav_data.shape[-1] > target_duration:
            wav_data = wav_data[..., :target_duration]
            if wav_data.dim() == 1:
                wav_data = wav_data.unsqueeze(0)
        else:
            if self.pad_types == 'zero':
                tmp = torch.zeros(1, target_duration, dtype=torch.float32)
                diff = target_duration - wav_data.shape[-1]
                tmp[..., diff // 2:wav_data.shape[-1] + diff // 2] = wav_data
                wav_data = tmp
            elif self.pad_types == 'repeat':
                ratio = math.ceil(target_duration / wav_data.shape[-1])
                wav_data = wav_data.repeat(1, ratio)
                wav_data = wav_data[..., :target_duration]
                wav_data = fade_out(wav_data)

        return wav_data


    def slice_audio(self, start, end, wav_data):
        """
        SCL paper..
        sample_rate denotes how many sample points for one second
        """
        max_ind = wav_data.shape[1]
        start_ind = min(int(start * self.sample_rate), max_ind)
        end_ind = min(int(end * self.sample_rate), max_ind)

        return wav_data[:, start_ind: end_ind]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        audio, label, meta_info = self.data[index]

        inputs = Processor.tokenizer(text=meta_info, return_tensors="pt", padding="max_length", max_length=64)
        text_inputs = inputs['input_ids']
        attn_masks = inputs['attention_mask']
        if self.transform is not None:
            audio = self.transform(audio)

        return audio.squeeze(1), label, text_inputs.squeeze(0), attn_masks.squeeze(0)




if __name__ == '__main__':
    train_dataset = ICBHIDataset(data_root='/dataset/ICBHI/', mode='train',
                           sample_rate=16000, desired_length=8, pad_types='zero', fade_samples_ratio=16)

    test_dataset = ICBHIDataset(data_root='/dataset/ICBHI/', mode='test',
                           sample_rate=16000, desired_length=8, pad_types='zero', fade_samples_ratio=16)

    train_loader = DataLoader(train_dataset,
                              batch_size=8,
                              num_workers=2,
                              pin_memory=True,
                              shuffle=True,
                              collate_fn=None)

    test_loader = DataLoader(test_dataset,
                              batch_size=8,
                              num_workers=2,
                              pin_memory=True,
                              shuffle=True,
                              collate_fn=None)

    tbar = tqdm(train_loader)
    for i, batch in enumerate(tbar):
        print(batch)


