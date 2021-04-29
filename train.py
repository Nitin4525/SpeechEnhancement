import os
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from model import Generator, Discriminator
from utils import AudioDataset
from loss import LossFunc

import warnings
warnings.filterwarnings('ignore')

cfg_path = r'config.yaml'
cfg = yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader)

BATCH_SIZE = cfg['hparas']['batch_size']
EPOCHS = cfg['hparas']['epochs']
LOG_DIR = os.path.join(cfg['hparas']['log_path'], cfg['hparas']['mode'])
SAVE_PATH = os.path.join(cfg['hparas']['save_path'], cfg['hparas']['mode'])
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

alpa = cfg['hparas']['spem_loss']
epoch_start = 0
torch.cuda.set_device(cfg['hparas']['cuda_device'])
cuda_available = torch.cuda.is_available()

generator_writer = SummaryWriter(os.path.join(LOG_DIR, 'generator'))
discriminator_writer = SummaryWriter(os.path.join(LOG_DIR, 'discriminator'))

generator = Generator(**cfg['model'])
discriminator = Discriminator()

generator_parameters = sum(param.numel() for param in generator.parameters())
discriminator_parameters = sum(param.numel() for param in discriminator.parameters())

print('cuda available: {}  Shink: {}  SpecLoss: {}'.format(cuda_available, cfg['model']['Shrink'], alpa))

print("# generator parameters:", generator_parameters)
print("# discriminator parameters:", discriminator_parameters)

generator_writer.add_graph(generator, [torch.rand([1, 1, 16384], dtype=torch.float32), torch.rand([1, 1024, 8], dtype=torch.float32)])
discriminator_writer.add_graph(discriminator, [torch.rand([1, 2, 16384], dtype=torch.float32), torch.rand([1, 2, 16384], dtype=torch.float32), ])

g_optimizer = optim.RMSprop(generator.parameters(), lr=cfg['hparas']['lr'], weight_decay=cfg['hparas']['weight_decay'])
d_optimizer = optim.RMSprop(discriminator.parameters(), lr=cfg['hparas']['lr'], weight_decay=cfg['hparas']['weight_decay'])

add_loss = LossFunc(cfg['loss'])

if cfg['hparas']['train_continue']:
    print('loading models ...')
    generator.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'generator-{}.pkl'.format(cfg['hparas']['checkpoint']))))
    discriminator.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'discriminator-{}.pkl'.format(cfg['hparas']['checkpoint']))))
    epoch_start = int(cfg['hparas']['checkpoint']) + 1

# load data
print('loading data...')
train_dataset = AudioDataset(cfg['data'])
train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=cfg['hparas']['num_workers'])
# generate reference batch
ref_batch = train_dataset.reference_batch(BATCH_SIZE)
if cuda_available:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    ref_batch = ref_batch.cuda()


for epoch in range(epoch_start, EPOCHS):
    train_bar = tqdm(train_data_loader)
    g_loss_list = []
    d_loss_list = []
    for train_batch, train_clean, train_noisy in train_bar:

        # latent vector - normal distribution
        z = nn.init.normal_(torch.Tensor(train_batch.size(0), 1024, 8))
        if cuda_available:
            train_batch, train_clean, train_noisy = train_batch.cuda(), train_clean.cuda(), train_noisy.cuda()
            z = z.cuda()

        # TRAIN D to recognize clean audio as clean
        # training batch pass
        discriminator.zero_grad()
        outputs = discriminator(train_batch, ref_batch)
        clean_loss = torch.mean((outputs - 1.0) ** 2)  # L2 loss - we want them all to be 1
        clean_loss.backward()

        # TRAIN D to recognize generated audio as noisy
        generated_outputs = generator(train_noisy, z)
        outputs = discriminator(torch.cat((generated_outputs, train_noisy), dim=1), ref_batch)
        noisy_loss = torch.mean(outputs ** 2)  # L2 loss - we want them all to be 0
        noisy_loss.backward()

        # d_loss = clean_loss + noisy_loss
        d_optimizer.step()  # update parameters

        # TRAIN G so that D recognizes G(z) as real
        generator.zero_grad()
        generated_outputs = generator(train_noisy, z)
        gen_noise_pair = torch.cat((generated_outputs, train_noisy), dim=1)
        outputs = discriminator(gen_noise_pair, ref_batch)

        g_loss_ = 0.5 * torch.mean((outputs - 1.0) ** 2)
        # L1 loss between generated output and clean sample
        l1_dist = torch.abs(torch.add(generated_outputs, torch.neg(train_clean)))
        g_cond_loss = 100 * torch.mean(l1_dist)  # conditional loss
        G_loss = g_loss_ + g_cond_loss + alpa * add_loss(generated_outputs, train_clean)

        # backprop + optimize
        G_loss.backward()
        g_optimizer.step()

        log = 'Epoch {}: G_loss {:.3f}, D_clean_loss {:.3f}, D_noisy_loss {:.3f}'.format(epoch + 1, G_loss.item(), clean_loss.item(), noisy_loss.item())

        train_bar.set_description(log)

        g_loss_list.append(G_loss.item())
        d_loss_list.append((clean_loss.item() + noisy_loss.item()) / 2)

    G_LOSS = np.mean(np.array(g_loss_list))
    D_LOSS = np.mean(np.array(d_loss_list))

    generator_writer.add_scalar('Loss', G_LOSS, global_step=epoch)
    discriminator_writer.add_scalar('Loss', D_LOSS, global_step=epoch)
    generator_writer.add_scalar('LR', g_optimizer.param_groups[0]['lr'], global_step=epoch)
    discriminator_writer.add_scalar('LR', d_optimizer.param_groups[0]['lr'], global_step=epoch)

    for name, param in generator.named_parameters():
        generator_writer.add_histogram(name, param, epoch)
    for name, param in discriminator.named_parameters():
        discriminator_writer.add_histogram(name, param, epoch)

    # save the model parameters for each epoch
    g_path = os.path.join(SAVE_PATH, 'generator-{}.pkl'.format(epoch))
    d_path = os.path.join(SAVE_PATH, 'discriminator-{}.pkl'.format(epoch))
    torch.save(generator.state_dict(), g_path)
    torch.save(discriminator.state_dict(), d_path)