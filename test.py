import yaml
import os

import numpy as np
import torch
import librosa
import soundfile as sf
from tqdm import tqdm

from model import Generator
from utils import emphasis


def slice_signal(file, window_size, stride, sample_rate):

    wav_pre, sr = librosa.load(file, sr=None)
    if sr == sample_rate:
        wav = wav_pre
    else:
        wav = librosa.resample(wav_pre, sr, sample_rate)
    wav = wav/np.max(wav)
    hop = int(window_size * stride)
    padding_len = window_size - len(wav) % window_size
    wav = np.pad(wav, (0, padding_len+1), 'constant')
    slices = []
    for end_idx in range(window_size, len(wav), hop):
        start_idx = end_idx - window_size
        slice_sig = wav[start_idx:end_idx]
        slices.append(slice_sig)
    return slices


if __name__ == '__main__':
    cfg_path = r'config/config.yaml'
    cfg = yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader)
    model_path = cfg['test']['model_path']
    file_path = cfg['test']['file_path']
    result_save_path = cfg['test']['result_save_path']
    window_size = cfg['data']['window_size']
    sample_rate = cfg['data']['sample_rate']

    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    generator = Generator()
    generator.load_state_dict(torch.load(model_path, map_location='cpu'))
    if torch.cuda.is_available():
        generator.cuda()
    generator.eval()

    print('model loaded...')

    for rootname, subdir, files in os.walk(file_path):
        if len(files) == 0:
            continue

        for i in range(len(files)):
            wav_name = os.sep.join([rootname, files[i]])

            noisy_slices = slice_signal(wav_name, window_size, 1, sample_rate)  # 分帧操作
            enhanced_speech = []

            for noisy_slice in tqdm(noisy_slices, desc='Generate enhanced audio'):
                noisy_slice = torch.from_numpy(emphasis(noisy_slice[np.newaxis, np.newaxis, :])).type(torch.FloatTensor)
                if torch.cuda.is_available():
                    noisy_slice = noisy_slice.cuda()
                generated_speech = generator(noisy_slice).data.cpu().numpy()
                generated_speech = emphasis(generated_speech, emph_coeff=0.95, pre=False)
                generated_speech = generated_speech.reshape(-1)
                enhanced_speech.append(generated_speech)

            enhanced_speech = np.array(enhanced_speech).reshape(1, -1)
            file_name = os.path.join(result_save_path, '{}.wav'.format(os.path.basename(wav_name).split('.')[0]))
            sf.write(file_name, enhanced_speech.T, sample_rate, )
