import yaml
import os

import numpy as np
import torch
import torch.nn as nn
import librosa
import soundfile as sf
from tqdm import tqdm

from model import Generator
from utils import emphasis


def slice_signal(file, window_size, hop, sample_rate):
    wav_pre, sr = librosa.load(file, sr=None)
    if sr == sample_rate:
        wav = wav_pre
    else:
        wav = librosa.resample(wav_pre, sr, sample_rate)
    wav = wav/np.max(wav)
    src_length = len(wav)
    padding_len = hop - ((len(wav) - window_size) % hop)
    wav = np.pad(wav, (0, padding_len), 'constant')
    slices = []
    for start_idx in range(0, len(wav)-window_size+1, hop):
        slice_sig = wav[start_idx:start_idx+window_size]
        slices.append(slice_sig)
    return slices, src_length


def connect_signal(sliceA, sliceB, overlap_length):
    sliceA_self = sliceA[:-overlap_length]
    sliceA_overlap = sliceA[-overlap_length:]
    sliceB_self = sliceB[overlap_length:]
    sliceB_overlap = sliceB[:overlap_length]

    right = len(sliceA_overlap)
    weight_A = []
    weight_B = []
    for i in range(right):
        weight_A.append(1-i / right)
        weight_B.append(i / right)
    overlap = sliceA_overlap * weight_A + sliceB_overlap * weight_B
    return np.concatenate((sliceA_self, overlap, sliceB_self))


if __name__ == '__main__':
    cfg_path = r'config.yaml'
    cfg = yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader)
    model_path = cfg['test']['model_path']
    file_path = cfg['test']['file_path']
    result_save_path = cfg['test']['result_save_path']
    window_size = cfg['data']['window_size']
    sample_rate = cfg['data']['sample_rate']
    window_stride = cfg['data']['window_stride'] if cfg['test']['DistanceFuse'] else 1
    hop = int(window_size * window_stride)

    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    generator = Generator(**cfg['model'])
    generator.load_state_dict(torch.load(model_path, map_location='cpu'))
    z = nn.init.normal_(torch.Tensor(1, 1024, 8))
    if torch.cuda.is_available():
        generator.cuda()
        z = z.cuda()
    generator.eval()

    print('model loaded...')

    for rootname, subdir, files in os.walk(file_path):
        if len(files) == 0:
            continue

        for file_ in tqdm(files, desc='Generate enhanced audio'):
            wav_name = os.sep.join([rootname, file_])

            noisy_slices, src_length = slice_signal(wav_name, window_size, hop, sample_rate)  # 分帧操作
            enhanced_speech = []

            for noisy_slice in noisy_slices:
                noisy_slice = torch.from_numpy(emphasis(noisy_slice[np.newaxis, np.newaxis, :])).type(torch.FloatTensor)
                if torch.cuda.is_available():
                    noisy_slice = noisy_slice.cuda()

                generated_speech = generator(noisy_slice, z).data.cpu().numpy()
                generated_speech = emphasis(generated_speech, emph_coeff=0.95, pre=False)
                generated_speech = generated_speech.reshape(-1)
                enhanced_speech.append(generated_speech)

            if cfg['test']['DistanceFuse']:
                enhanced_speech_ = enhanced_speech[0]
                for i in range(1, len(enhanced_speech)):
                    enhanced_speech_ = connect_signal(enhanced_speech_, enhanced_speech[i], overlap_length=window_size-hop)
            else:
                enhanced_speech_ = np.array(enhanced_speech).reshape(1, -1)

            file_name = os.path.join(result_save_path, '{}.wav'.format(os.path.basename(wav_name).split('.')[0]))
            sf.write(file_name, enhanced_speech_.T[:src_length], sample_rate, )
