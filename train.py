import os
import yaml
import numpy as np
from tqdm import tqdm

import torch
from torch.nn import DataParallel
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from model import Generator, Discriminator, Adam
from utils import AudioDataset


cuda_available = torch.cuda.is_available()
num_gpu = torch.cuda.device_count()

use_cuda = False
use_multi_gpu = False

cfg_path = r'config/config.yaml'
cfg = yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader)

BATCH_SIZE = cfg['hparas']['batch_size']
EPOCHS = cfg['hparas']['epochs']
SAVE_PATH = cfg['hparas']['save_path']
LOG_DIR = cfg['hparas']['log_path']

generator_writer = SummaryWriter(os.path.join(LOG_DIR, 'generator'))
discriminator_writer = SummaryWriter(os.path.join(LOG_DIR, 'discriminator'))

generator = Generator()
discriminator = Discriminator()

generator_writer.add_graph(generator, [torch.rand([1, 1, 16384], dtype=torch.float32)])
discriminator_writer.add_graph(discriminator, [torch.rand([1, 2, 16384], dtype=torch.float32), torch.rand([1, 2, 16384], dtype=torch.float32), ])

g_optimizer = Adam(generator.parameters(), cfg=cfg['hparas']['optim'])
d_optimizer = Adam(discriminator.parameters(), cfg=cfg['hparas']['optim'])

g_lr_change = CosineAnnealingWarmRestarts(optimizer=g_optimizer, T_0=10, T_mult=2, eta_min=0, last_epoch=-1)
d_lr_change = CosineAnnealingWarmRestarts(optimizer=d_optimizer, T_0=10, T_mult=2, eta_min=0, last_epoch=-1)

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

if cfg['hparas']['train_continue']:
    print('loading models ...')
    generator.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'generator-{}'.format(cfg['hparas']['checkpoint']))))
    discriminator.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'discriminator-{}'.format(cfg['hparas']['checkpoint']))))

if cuda_available and cfg['hparas']['select_gpu']:
    torch.cuda.set_device(cfg['hparas']['select_gpu'][0])
    generator = generator.cuda()
    discriminator = discriminator.cuda()

    g_optimizer.cuda()
    d_optimizer.cuda()

    use_cuda = True

    if 1 < len(cfg['hparas']['select_gpu']) <= num_gpu:
        use_multi_gpu = True
        generator = DataParallel(generator, device_ids=cfg['hparas']['select_gpu'])
        discriminator = DataParallel(discriminator, device_ids=cfg['hparas']['select_gpu'])
        batch_size = cfg['hparas']['batch_size'] * len(cfg['hparas']['select_gpu'])

print('cuda available:', cuda_available, '  num gpu:', num_gpu,
      '\nuse_cuda:', use_cuda, '  use_multi_gpu:', use_multi_gpu, '  select ', cfg['hparas']['select_gpu'])

train_dataset = AudioDataset(cfg=cfg['data'])
# test_dataset = AudioDataset(cfg=cfg['data'], data_type='test')

train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# test_data_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
# generate reference batch
ref_batch = train_dataset.reference_batch(BATCH_SIZE)
# test_ref_batch = test_dataset.reference_batch(BATCH_SIZE)

if use_cuda:
    ref_batch = ref_batch.cuda()


for epoch in range(EPOCHS):
    train_bar = tqdm(train_data_loader)
    g_loss_list = []
    d_loss_list = []
    for train_batch, train_clean, train_noisy in train_bar:

        if use_cuda:
            train_batch, train_clean, train_noisy = train_batch.cuda(), train_clean.cuda(), train_noisy.cuda()

        discriminator.zero_grad()
        outputs = discriminator(train_batch, ref_batch)
        clean_loss = torch.mean((outputs - 1.0) ** 2)
        clean_loss.backward()

        generated_outputs = generator(train_noisy)
        outputs2 = discriminator(torch.cat((generated_outputs, train_noisy), dim=1), ref_batch)
        noisy_loss = torch.mean(outputs2 ** 2)
        noisy_loss.backward()

        d_optimizer.step()

        g_optimizer.zero_grad()
        generated_outputs = generator(train_noisy)
        outputs = discriminator(torch.cat((generated_outputs, train_noisy), dim=1), ref_batch)
        l1_dist = torch.mean(torch.abs(torch.add(generated_outputs, torch.neg(train_clean))))
        G_loss = 0.5 * torch.mean((outputs - 1.0) ** 2) + 100.0 * l1_dist
        G_loss.backward()
        g_optimizer.step()

        log = 'Epoch {}: G_loss {:.3f}, D_clean_loss {:.3f}, D_noisy_loss {:.3f}, G_LR {:.2e}, D_LR {:.2e}'.format(epoch + 1,
                                                                                                                   G_loss.item(),
                                                                                                                   clean_loss.item(),
                                                                                                                   noisy_loss.item(),
                                                                                                                   g_optimizer.param_groups[0]['lr'],
                                                                                                                   d_optimizer.param_groups[0]['lr'])

        train_bar.set_description(log)

        g_loss_list.append(G_loss.item())
        d_loss_list.append((clean_loss.item() + noisy_loss.item()) / 2)

    G_LOSS = np.mean(np.array(g_loss_list))
    D_LOSS = np.mean(np.array(d_loss_list))

    generator_writer.add_scalar('Loss', G_LOSS, global_step=epoch)
    discriminator_writer.add_scalar('Loss', D_LOSS, global_step=epoch)
    generator_writer.add_scalar('LR', g_optimizer.param_groups[0]['lr'], global_step=epoch)
    discriminator_writer.add_scalar('LR', d_optimizer.param_groups[0]['lr'], global_step=epoch)

    g_lr_change.step()
    d_lr_change.step()

    # test_bar = tqdm(test_data_loader, desc='Test model')
    # generator = generator.cpu().eval()
    # discriminator = discriminator.cpu().eval()
    # G_test_loss = []
    # for test_clean, test_noisy in test_bar:
    #     z = torch.rand([test_noisy.size(0), 1024, 8])
    #     fake_speech = generator(test_noisy, z)
    #     outputs = discriminator(torch.cat((fake_speech, test_noisy), dim=1), test_ref_batch)
    #     l1_dist = torch.mean(torch.abs(torch.add(fake_speech, torch.neg(test_clean))))
    #     G_loss = 0.5 * torch.mean((outputs - 1.0) ** 2) + 100.0 * l1_dist
    #
    #     log = 'Epoch {}: test_G_loss {:.4f}'.format(epoch + 1, G_loss.data)
    #     test_bar.set_description(log)
    #     test_g_loss = G_loss.item()
    #     G_test_loss.append(test_g_loss)
    # G_test_LOSS = np.mean(np.array(G_test_loss))
    # generator_writer.add_scalar('test_LOSS', G_test_LOSS, global_step=epoch)
    # generator = generator.cuda().train()
    # discriminator = discriminator.cuda().train()

    # save the model parameters for each epoch
    if epoch % 10 == 0 or G_loss.item() <= 0.6:
        g_path = os.path.join(cfg['hparas']['save_path'], 'generator-{}.pkl'.format(epoch + 1))
        d_path = os.path.join(cfg['hparas']['save_path'], 'discriminator-{}.pkl'.format(epoch + 1))
        torch.save(generator.state_dict(), g_path)
        torch.save(discriminator.state_dict(), d_path)
