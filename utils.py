import os

import numpy as np
import torch
from scipy import signal
from torch.utils import data


def emphasis(signal_batch, emph_coeff=0.95, pre=True):
    """
    Pre-emphasis or De-emphasis of higher frequencies given a batch of signal.

    Args:
        signal_batch: batch of signals, represented as numpy arrays
        emph_coeff: emphasis coefficient
        pre: pre-emphasis or de-emphasis signals

    Returns:
        result: pre-emphasized or de-emphasized signal batch
    """
    '''
    result = np.zeros(signal_batch.shape)
    for sample_idx, sample in enumerate(signal_batch):
        for ch, channel_data in enumerate(sample):
            if pre:
                result[sample_idx][ch] = np.append(channel_data[0], channel_data[1:] - emph_coeff * channel_data[:-1])
            else:
                result[sample_idx][ch] = np.append(channel_data[0], channel_data[1:] + emph_coeff * channel_data[:-1])
    return result
    '''
    if pre:
        return signal.lfilter([1, -emph_coeff], [1], signal_batch)
    else:
        return signal.lfilter([1], [1, -emph_coeff], signal_batch)


class AudioDataset(data.Dataset):
    """
    Audio sample reader.
    """

    def __init__(self, cfg):

        data_path = os.path.join(cfg['root_path'], cfg['corpus'], cfg['data_type'], 'serialized_data')

        if not os.path.exists(data_path):
            raise FileNotFoundError('The folder {} does not exist!'.format(data_path))

        self.data_type = cfg['data_type']
        self.file_names = [os.path.join(data_path, filename) for filename in os.listdir(data_path)]

    def reference_batch(self, batch_size):
        """
        Randomly selects a reference batch from dataset.
        Reference batch is used for calculating statistics for virtual batch normalization operation.

        从数据集中随机选择一个引用批。用于计算虚拟批规范化操作的统计信息。

        Args:
            batch_size(int): batch size

        Returns:
            ref_batch: reference batch
        """
        ref_file_names = np.random.choice(self.file_names, batch_size)
        ref_batch = np.stack([np.load(f, allow_pickle=True) for f in ref_file_names])

        ref_batch = emphasis(ref_batch, emph_coeff=0.95)
        return torch.from_numpy(ref_batch).type(torch.FloatTensor)

    def __getitem__(self, idx):
        pair = np.load(self.file_names[idx], allow_pickle=True)
        pair = emphasis(pair[np.newaxis, :, :], emph_coeff=0.95).reshape(2, -1)
        noisy = pair[1].reshape(1, -1)
        if self.data_type == 'train':
            clean = pair[0].reshape(1, -1)
            return torch.from_numpy(pair).type(torch.FloatTensor), torch.from_numpy(clean).type(
                torch.FloatTensor), torch.from_numpy(noisy).type(torch.FloatTensor)
        elif self.data_type == 'test':
            clean = pair[0].reshape(1, -1)
            return torch.from_numpy(clean).type(torch.FloatTensor), torch.from_numpy(noisy).type(torch.FloatTensor)
        else:
            return os.path.basename(self.file_names[idx]), torch.from_numpy(noisy).type(torch.FloatTensor)

    def __len__(self):
        return len(self.file_names)
