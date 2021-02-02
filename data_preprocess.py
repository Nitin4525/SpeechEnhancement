import os
import yaml

import librosa
import numpy as np
from tqdm import tqdm


def slice_signal(file, window_size, stride, sample_rate):

    wav, sr = librosa.load(file, sr=None)

    if sr != sample_rate:
        wav = librosa.resample(wav, sr, sample_rate)

    wav = wav / np.max(np.abs(wav))

    if np.max(wav) > 1 or np.min(wav) < -1:
        print('need to norm')

    hop = int(window_size * stride)
    slices = []
    for end_idx in range(window_size, len(wav), hop):
        start_idx = end_idx - window_size
        slice_sig = wav[start_idx:end_idx]
        slices.append(slice_sig)
    return slices


def process_and_serialize(data_type):

    stride = 0.5
    cfg_path = r'config/config.yaml'
    cfg = yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader)

    root_dir = cfg['data']['root_path']
    corpus = cfg['data']['corpus']
    window_size = cfg['data']['window_size']
    sample_rate = cfg['data']['sample_rate']

    clean_folder = os.path.join(root_dir, corpus, data_type, 'clean')
    noisy_folder = os.path.join(root_dir, corpus, data_type, 'noise')
    serialized_folder = os.path.join(root_dir, corpus, data_type, 'serialized_data')

    if not os.path.exists(serialized_folder):
        os.makedirs(serialized_folder)

    for root, dirs, files in os.walk(clean_folder):
        if len(files) == 0:
            continue
        for filename in tqdm(files, desc='Serialize and down-sample {} audios'.format(data_type)):
            clean_file = os.path.join(clean_folder, filename)
            noisy_file = os.path.join(noisy_folder, filename)
            # slice both clean signal and noisy signal
            clean_sliced = slice_signal(clean_file, window_size, stride, sample_rate)
            noisy_sliced = slice_signal(noisy_file, window_size, stride, sample_rate)
            for idx, slice_tuple in enumerate(zip(clean_sliced, noisy_sliced)):
                pair = np.array([slice_tuple[0], slice_tuple[1]])
                np.save(os.path.join(serialized_folder, '{}_{}'.format(filename, idx)), arr=pair)
    data_verify(serialized_folder=serialized_folder, window_size=window_size)


def data_verify(serialized_folder, window_size):
    for root, dirs, files in os.walk(serialized_folder):
        for filename in tqdm(files, desc='Verify serialized audios'):
            data_pair = np.load(os.path.join(root, filename), allow_pickle=True)
            if data_pair.shape[1] != window_size:
                print('Snippet length not {} : {} instead'.format(window_size, data_pair.shape[1]))
                break


if __name__ == '__main__':
    process_and_serialize('train')
    # process_and_serialize('test')

