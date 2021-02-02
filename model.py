import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.parameter import Parameter


class VirtualBatchNorm1d(nn.Module):
    """
    References https://discuss.pytorch.org/t/parameter-grad-of-conv-weight-is-none-after-virtual-batch-normalization/9036
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        # batch statistics
        self.num_features = num_features
        self.eps = eps  # epsilon
        # define gamma and beta parameters
        self.gamma = Parameter(torch.normal(mean=1.0, std=0.02, size=(1, num_features, 1)))
        self.beta = Parameter(torch.zeros(1, num_features, 1))

    def get_stats(self, x):
        """
        Calculates mean and mean square for given batch x.
        Args:
            x: tensor containing batch of activations
        Returns:
            mean: mean tensor over features
            mean_sq: squared mean tensor over features
        """
        mean = x.mean(2, keepdim=True).mean(0, keepdim=True)
        mean_sq = (x ** 2).mean(2, keepdim=True).mean(0, keepdim=True)
        return mean, mean_sq

    def forward(self, x, ref_mean, ref_mean_sq):
        """
        Forward pass of virtual batch normalization.
        Virtual batch normalization require two forward passes
        for reference batch and train batch, respectively.

        Args:
            x: input tensor
            ref_mean: reference mean tensor over features
            ref_mean_sq: reference squared mean tensor over features
        Result:
            x: normalized batch tensor
            ref_mean: reference mean tensor over features
            ref_mean_sq: reference squared mean tensor over features
        """
        mean, mean_sq = self.get_stats(x)
        if ref_mean is None or ref_mean_sq is None:
            # reference mode - works just like batch norm
            mean = mean.clone().detach()
            mean_sq = mean_sq.clone().detach()
            out = self.normalize(x, mean, mean_sq)
        else:
            # calculate new mean and mean_sq
            batch_size = x.size(0)
            new_coeff = 1. / (batch_size + 1.)
            old_coeff = 1. - new_coeff
            mean = new_coeff * mean + old_coeff * ref_mean
            mean_sq = new_coeff * mean_sq + old_coeff * ref_mean_sq
            out = self.normalize(x, mean, mean_sq)
        return out, mean, mean_sq

    def normalize(self, x, mean, mean_sq):
        """
        Normalize tensor x given the statistics.

        Args:
            x: input tensor
            mean: mean over features
            mean_sq: squared means over features

        Result:
            x: normalized batch tensor
        """
        assert mean_sq is not None
        assert mean is not None
        assert len(x.size()) == 3  # specific for 1d VBN
        if mean.size(1) != self.num_features:
            raise Exception('Mean tensor size not equal to number of features : given {}, expected {}'
                            .format(mean.size(1), self.num_features))
        if mean_sq.size(1) != self.num_features:
            raise Exception('Squared mean tensor size not equal to number of features : given {}, expected {}'
                            .format(mean_sq.size(1), self.num_features))

        std = torch.sqrt(self.eps + mean_sq - mean ** 2)    # 这个地方减mean**2在BN的资料中没有，但很多例程都是这么干的，这就是VBN相较于BN不同的地方
        x = x - mean
        x = x / std
        x = x * self.gamma
        x = x + self.beta
        return x

    def __repr__(self):
        return ('{name}(num_features={num_features}, eps={eps}'
                .format(name=self.__class__.__name__, **self.__dict__))


class ConvBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: list, dropout: float = None, res: bool = False):
        super(ConvBlock, self).__init__()
        self.convA = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size[0], stride=1, padding=1)
        self.actA = nn.LeakyReLU()
        self.convB = nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size[1], stride=1, padding=1)
        self.actB = nn.LeakyReLU()
        self.res = res

        self.dropout = nn.Identity() if dropout is None else nn.Dropout(p=dropout)

    def forward(self, inputs):
        x = self.convA(inputs)
        x = self.actA(x)
        x = self.convB(x)
        x = self.actB(x)
        if self.res:
            return x, self.dropout(x)
        else:
            return self.dropout(x)


class MergeBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: list):
        super(MergeBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=kernel_size[0])
        self.conv = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size[1], padding=1)
        self.act = nn.LeakyReLU()

    def forward(self, inputs):
        x = self.upsample(inputs[0])
        x = self.conv(x)
        x = self.act(x)
        merge = torch.cat((x, inputs[1]), dim=1)
        return merge


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = ConvBlock(in_channel=1, out_channel=16, kernel_size=[3, 3], res=True)
        self.pooling1 = nn.MaxPool1d(kernel_size=2)
        self.layer2 = ConvBlock(in_channel=16, out_channel=32, kernel_size=[3, 3], res=True)
        self.pooling2 = nn.MaxPool1d(kernel_size=2)
        self.layer3 = ConvBlock(in_channel=32, out_channel=64, kernel_size=[3, 3], res=True)
        self.pooling3 = nn.MaxPool1d(kernel_size=2)
        self.layer4 = ConvBlock(in_channel=64, out_channel=128, kernel_size=[3, 3], res=True)
        self.pooling4 = nn.MaxPool1d(kernel_size=2)
        self.layer5 = ConvBlock(in_channel=128, out_channel=256, kernel_size=[3, 3], res=True)
        self.pooling5 = nn.MaxPool1d(kernel_size=2)
        self.layer6 = ConvBlock(in_channel=256, out_channel=512, kernel_size=[3, 3], dropout=0.5, res=True)
        self.pooling6 = nn.MaxPool1d(kernel_size=2)

        self.layer7 = ConvBlock(in_channel=512, out_channel=1024, kernel_size=[3, 3], dropout=0.5)

        self.layer8 = nn.Sequential(MergeBlock(in_channel=1024, out_channel=512, kernel_size=[2, 3]),
                                    ConvBlock(in_channel=1024, out_channel=512, kernel_size=[3, 3]))
        self.layer9 = nn.Sequential(MergeBlock(in_channel=512, out_channel=256, kernel_size=[2, 3]),
                                    ConvBlock(in_channel=512, out_channel=256, kernel_size=[3, 3]))
        self.layer10 = nn.Sequential(MergeBlock(in_channel=256, out_channel=128, kernel_size=[2, 3]),
                                     ConvBlock(in_channel=256, out_channel=128, kernel_size=[3, 3]))
        self.layer11 = nn.Sequential(MergeBlock(in_channel=128, out_channel=64, kernel_size=[2, 3]),
                                     ConvBlock(in_channel=128, out_channel=64, kernel_size=[3, 3]))
        self.layer12 = nn.Sequential(MergeBlock(in_channel=64, out_channel=32, kernel_size=[2, 3]),
                                     ConvBlock(in_channel=64, out_channel=32, kernel_size=[3, 3]))
        self.layer13 = nn.Sequential(MergeBlock(in_channel=32, out_channel=16, kernel_size=[2, 3]),
                                     nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
                                     nn.LeakyReLU(),
                                     nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
                                     nn.LeakyReLU(),
                                     nn.Conv1d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
                                     nn.LeakyReLU())
        self.final = nn.Sequential(nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1),
                                   nn.Tanh())
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x1, res1 = self.layer1(x)
        x2, res2 = self.layer2(self.pooling1(x1))
        x3, res3 = self.layer3(self.pooling2(x2))
        x4, res4 = self.layer4(self.pooling3(x3))
        x5, res5 = self.layer5(self.pooling4(x4))
        x6, res6 = self.layer6(self.pooling5(x5))
        x7 = self.layer7(self.pooling6(x6))
        x8 = self.layer8([x7, res6])
        x9 = self.layer9([x8, res5])
        x10 = self.layer10([x9, res4])
        x11 = self.layer11([x10, res3])
        x12 = self.layer12([x11, res2])
        x13 = self.layer13([x12, res1])
        outputs = self.final(x13)
        return outputs


class Discriminator(nn.Module):
    """
    References from SEGAN https://arxiv.org/abs/1703.09452
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        negative_slope = 0.03
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=31, stride=2, padding=15)
        self.vbn1 = VirtualBatchNorm1d(32)
        self.lrelu1 = nn.LeakyReLU(negative_slope)
        self.conv2 = nn.Conv1d(32, 64, 31, 2, 15)
        self.vbn2 = VirtualBatchNorm1d(64)
        self.lrelu2 = nn.LeakyReLU(negative_slope)
        self.conv3 = nn.Conv1d(64, 64, 31, 2, 15)
        self.dropout1 = nn.Dropout()
        self.vbn3 = VirtualBatchNorm1d(64)
        self.lrelu3 = nn.LeakyReLU(negative_slope)
        self.conv4 = nn.Conv1d(64, 128, 31, 2, 15)
        self.vbn4 = VirtualBatchNorm1d(128)
        self.lrelu4 = nn.LeakyReLU(negative_slope)
        self.conv5 = nn.Conv1d(128, 128, 31, 2, 15)
        self.vbn5 = VirtualBatchNorm1d(128)
        self.lrelu5 = nn.LeakyReLU(negative_slope)
        self.conv6 = nn.Conv1d(128, 256, 31, 2, 15)
        self.dropout2 = nn.Dropout()
        self.vbn6 = VirtualBatchNorm1d(256)
        self.lrelu6 = nn.LeakyReLU(negative_slope)
        self.conv7 = nn.Conv1d(256, 256, 31, 2, 15)
        self.vbn7 = VirtualBatchNorm1d(256)
        self.lrelu7 = nn.LeakyReLU(negative_slope)
        self.conv8 = nn.Conv1d(256, 512, 31, 2, 15)
        self.vbn8 = VirtualBatchNorm1d(512)
        self.lrelu8 = nn.LeakyReLU(negative_slope)
        self.conv9 = nn.Conv1d(512, 512, 31, 2, 15)
        self.dropout3 = nn.Dropout()
        self.vbn9 = VirtualBatchNorm1d(512)
        self.lrelu9 = nn.LeakyReLU(negative_slope)
        self.conv10 = nn.Conv1d(512, 1024, 31, 2, 15)
        self.vbn10 = VirtualBatchNorm1d(1024)
        self.lrelu10 = nn.LeakyReLU(negative_slope)
        self.conv11 = nn.Conv1d(1024, 2048, 31, 2, 15)
        self.vbn11 = VirtualBatchNorm1d(2048)
        self.lrelu11 = nn.LeakyReLU(negative_slope)
        self.conv_final = nn.Conv1d(2048, 1, kernel_size=1, stride=1)
        self.lrelu_final = nn.LeakyReLU(negative_slope)
        self.fully_connected = nn.Linear(in_features=8, out_features=1)
        self.dropout4 = nn.Dropout()
        self.sigmoid = nn.Sigmoid()

        # initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x, ref_x):
        """
        Forward pass of discriminator.
        Args:
            x: D gets a noisy signal and clear signal as input [B x 2 x 16384]
            ref_x: reference input batch for virtual batch norm
        """
        ref_x = self.conv1(ref_x)
        ref_x, mean1, meansq1 = self.vbn1(ref_x, None, None)
        ref_x = self.lrelu1(ref_x)
        ref_x = self.conv2(ref_x)
        ref_x, mean2, meansq2 = self.vbn2(ref_x, None, None)
        ref_x = self.lrelu2(ref_x)
        ref_x = self.conv3(ref_x)
        ref_x = self.dropout1(ref_x)
        ref_x, mean3, meansq3 = self.vbn3(ref_x, None, None)
        ref_x = self.lrelu3(ref_x)
        ref_x = self.conv4(ref_x)
        ref_x, mean4, meansq4 = self.vbn4(ref_x, None, None)
        ref_x = self.lrelu4(ref_x)
        ref_x = self.conv5(ref_x)
        ref_x, mean5, meansq5 = self.vbn5(ref_x, None, None)
        ref_x = self.lrelu5(ref_x)
        ref_x = self.conv6(ref_x)
        ref_x = self.dropout2(ref_x)
        ref_x, mean6, meansq6 = self.vbn6(ref_x, None, None)
        ref_x = self.lrelu6(ref_x)
        ref_x = self.conv7(ref_x)
        ref_x, mean7, meansq7 = self.vbn7(ref_x, None, None)
        ref_x = self.lrelu7(ref_x)
        ref_x = self.conv8(ref_x)
        ref_x, mean8, meansq8 = self.vbn8(ref_x, None, None)
        ref_x = self.lrelu8(ref_x)
        ref_x = self.conv9(ref_x)
        ref_x = self.dropout3(ref_x)
        ref_x, mean9, meansq9 = self.vbn9(ref_x, None, None)
        ref_x = self.lrelu9(ref_x)
        ref_x = self.conv10(ref_x)
        ref_x, mean10, meansq10 = self.vbn10(ref_x, None, None)
        ref_x = self.lrelu10(ref_x)
        ref_x = self.conv11(ref_x)
        ref_x, mean11, meansq11 = self.vbn11(ref_x, None, None)

        x = self.conv1(x)
        x, _, _ = self.vbn1(x, mean1, meansq1)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x, _, _ = self.vbn2(x, mean2, meansq2)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        x, _, _ = self.vbn3(x, mean3, meansq3)
        x = self.lrelu3(x)
        x = self.conv4(x)
        x, _, _ = self.vbn4(x, mean4, meansq4)
        x = self.lrelu4(x)
        x = self.conv5(x)
        x, _, _ = self.vbn5(x, mean5, meansq5)
        x = self.lrelu5(x)
        x = self.conv6(x)
        x = self.dropout2(x)
        x, _, _ = self.vbn6(x, mean6, meansq6)
        x = self.lrelu6(x)
        x = self.conv7(x)
        x, _, _ = self.vbn7(x, mean7, meansq7)
        x = self.lrelu7(x)
        x = self.conv8(x)
        x, _, _ = self.vbn8(x, mean8, meansq8)
        x = self.lrelu8(x)
        x = self.conv9(x)
        x = self.dropout3(x)
        x, _, _ = self.vbn9(x, mean9, meansq9)
        x = self.lrelu9(x)
        x = self.conv10(x)
        x, _, _ = self.vbn10(x, mean10, meansq10)
        x = self.lrelu10(x)
        x = self.conv11(x)
        x, _, _ = self.vbn11(x, mean11, meansq11)
        x = self.lrelu11(x)
        x = self.conv_final(x)
        x = self.lrelu_final(x)
        x = torch.squeeze(x)
        x = self.fully_connected(x)
        x = self.dropout4(x)
        return self.sigmoid(x)


class Adam(optim.Adam):
    def __init__(self, params, cfg):
        super(Adam, self).__init__(params, cfg['learning_rate'], cfg['betas'], cfg['eps'], cfg['weight_decay'], cfg['amsgrad'])

    def cuda(self):
        for state in self.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    def cuda_gpu(self, gpu):
        for state in self.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda(gpu)


if __name__ == '__main__':
    net = Generator()
    inputs = torch.rand([1, 1, 16384], dtype=torch.float32)

    out = net(inputs)
    pass


