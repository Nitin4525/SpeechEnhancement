import os
import yaml

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


def slice_signal(file, window_size, hop, sample_rate):
    wav_pre, sr = librosa.load(file, sr=None)
    if sr == sample_rate:
        wav = wav_pre
    else:
        wav = librosa.resample(wav_pre, sr, sample_rate)
    wav = wav/np.max(wav)
    padding_len = hop - ((len(wav) - window_size) % hop)
    wav = np.pad(wav, (0, padding_len), 'constant')
    slices = []
    for start_idx in range(0, len(wav)-window_size+1, hop):
        slice_sig = wav[start_idx:start_idx+window_size]
        slices.append(slice_sig)
    return slices


def process_and_serialize():
    cfg_path = r'config.yaml'
    cfg = yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader)

    data_type = cfg['data']['data_type']
    root_dir = cfg['data']['root_path']
    corpus = cfg['data']['corpus']
    window_size = cfg['data']['window_size']
    sample_rate = cfg['data']['sample_rate']
    window_stride = cfg['data']['window_stride']
    hop = int(window_size * window_stride)

    clean_folder = os.path.join(root_dir, corpus, data_type, 'clean')
    noisy_folder = os.path.join(root_dir, corpus, data_type, 'noise')
    serialized_folder_clean = os.path.join(root_dir, corpus, data_type, 'serialized_data/clean')
    serialized_folder_noisy = os.path.join(root_dir, corpus, data_type, 'serialized_data/noisy')

    if not os.path.exists(serialized_folder_clean):
        os.makedirs(serialized_folder_clean)
    if not os.path.exists(serialized_folder_noisy):
        os.makedirs(serialized_folder_noisy)

    for root, dirs, files in os.walk(clean_folder):
        if len(files) == 0:
            continue
        for filename in tqdm(files, desc='Serialize and down-sample {} audios'.format(data_type)):
            clean_file = os.path.join(clean_folder, filename)

            noisy_file = os.path.join(noisy_folder, filename)
            clean_sliced = slice_signal(clean_file, window_size, hop, sample_rate)
            noisy_sliced = slice_signal(noisy_file, window_size, hop, sample_rate)
            for idx, slice_tuple in enumerate(zip(clean_sliced, noisy_sliced)):
                pair = np.array(slice_tuple[0])
                sf.write(os.path.join(serialized_folder_clean, '{}_{}.wav'.format(filename, idx)), pair, sample_rate, )
                pair = np.array(slice_tuple[1])
                sf.write(os.path.join(serialized_folder_noisy, '{}_{}.wav'.format(filename, idx)), pair, sample_rate, )


if __name__ == '__main__':
    process_and_serialize()
