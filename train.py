import sys
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
import argparse
import numpy as np
import matplotlib.pyplot as plt
from  datasets import LabeledDataset
from scd_vg import *
from sagan import *
from config import *
import os
import random
import utils
from torch.utils.data import TensorDataset, DataLoader
import torch
print(torch.version.cuda)
def main():

    global args
    args = get_config()
    args.commond = 'python ' + ' '.join(sys.argv)

    if 'pendulum' in args.dataset:
        label_idx = range(4)
    else:
        if args.labels == 'smile':
            label_idx = [31, 20, 19, 21, 23, 13]
        elif args.labels == 'age':
            label_idx = [39, 20, 28, 18, 13, 3]
        else:
            raise NotImplementedError("Not supported structure.")
    num_label = len(label_idx)

    save_dir = './results/{}/{}_{}_sup{}/'.format(
        args.dataset, args.labels, args.prior, str(args.sup_type))
    utils.make_folder(save_dir)
    utils.write_config_to_file(args, save_dir)

    global device
    device = torch.device('cuda')
    global celoss
    celoss = torch.nn.BCEWithLogitsLoss()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset = LabeledDataset()
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    # train_set,train_loader, test_set,test_loader = utils.make_dataloader(args)
    log_file_name = os.path.join(save_dir, 'log.txt')
    global log_file
    if args.resume:
        log_file = open(log_file_name, "at")
    else:
        log_file = open(log_file_name, "wt")

    if 'scm' in args.prior:
        A = torch.zeros((num_label, num_label))
        if args.labels == 'smile':
            A[0, 2:6] = 1
            A[1, 4] = 1
        elif args.labels == 'age':
            A[0, 2:6] = 1
            A[1, 2:4] = 1
        elif args.labels == 'pend':
            A[0, 2:4] = 1
            A[1, 2:4] = 1
    else:
        A = None

    print('Build models...')

    model = BGM(args.latent_dim, args.g_conv_dim, args.image_size,
                args.enc_dist, args.enc_arch, args.enc_fc_size, args.enc_noise_dim, args.dec_dist,
                args.prior, num_label, A)

    discriminator = BigJointDiscriminator(args.latent_dim, args.d_conv_dim, args.image_size,
                                          args.dis_fc_size)

    A_optimizer = None
    prior_optimizer = None
    if 'scm' in args.prior:
        enc_param = model.encoder.parameters()
        dec_param = list(model.decoder.parameters())
        prior_param = list(model.prior.parameters())
        A_optimizer = optim.Adam(prior_param[0:1], lr=args.lr_a)
        prior_optimizer = optim.Adam(prior_param[1:], lr=args.lr_p, betas=(args.beta1, args.beta2))
    else:
        enc_param = model.encoder.parameters()
        dec_param = model.decoder.parameters()
    encoder_optimizer = optim.Adam(enc_param, lr=args.lr_e, betas=(args.beta1, args.beta2))
    decoder_optimizer = optim.Adam(dec_param, lr=args.lr_g, betas=(args.beta1, args.beta2))
    D_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    # Load model from checkpoint
    if args.resume:
        ckpt_dir = args.ckpt_dir if args.ckpt_dir != '' else save_dir + args.model_type + str(
            args.start_epoch - 1) + '.sav'
        checkpoint = torch.load(ckpt_dir)
        model.load_state_dict(checkpoint['model'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        del checkpoint

    model = nn.DataParallel(model.to(device))
    discriminator = nn.DataParallel(discriminator.to(device))

    # Fixed noise from prior p_z for generating from G
    global fixed_noise, fixed_unif_noise, fixed_zeros
    if args.prior == 'uniform':
        fixed_noise = torch.rand(args.save_n_samples, args.latent_dim, device=device) * 2 - 1
    else:
        fixed_noise = torch.randn(args.save_n_samples, args.latent_dim, device=device)
    fixed_unif_noise = torch.rand(1, args.latent_dim, device=device) * 2 - 1
    fixed_zeros = torch.zeros(1, args.latent_dim, device=device)

    # Train
    print('Start training...')
    for i in range(args.start_epoch, args.start_epoch + args.n_epochs):
        train(i, model, discriminator, encoder_optimizer, decoder_optimizer, D_optimizer, train_loader, label_idx,
                  args.print_every, save_dir, prior_optimizer, A_optimizer)
        if i % args.save_model_every == 0:
            torch.save({'model': model.module.state_dict(), 'discriminator': discriminator.module.state_dict()},
                       save_dir + 'model' + str(i) + '.sav')


def train(epoch, model, discriminator, encoder_optimizer, decoder_optimizer, D_optimizer,
              train_loader, label_idx, print_every, save_dir,
              prior_optimizer, A_optimizer):
    model.train()
    discriminator.train()
   
    for batch_idx, (x, label) in enumerate(train_loader):
        x = x.to(device)
        # supervision flag
        sup_flag = label[:, 0] != -1
        if sup_flag.sum() > 0:
            label = label[sup_flag, :][:, label_idx].float()
       
        num_labels = len(label_idx)
        label = label.to(device)
      
        # ================== TRAIN DISCRIMINATOR ================== #
        for _ in range(args.d_steps_per_iter):
            discriminator.zero_grad()
            z = torch.randn(x.size(0), args.latent_dim, device=x.device)
            z_fake, x_fake, z, z_mu ,log_det= model(x, z)
            # Compute D loss
            encoder_score = discriminator(x, z_fake.detach())
            decoder_score = discriminator(x_fake.detach(), z.detach())
            del z_fake
            del x_fake

            loss_d = F.softplus(decoder_score).mean() + F.softplus(-encoder_score).mean()
            loss_d.backward()
            D_optimizer.step()

        for _ in range(args.g_steps_per_iter):

            z = torch.randn(x.size(0), args.latent_dim, device=x.device)
            z_fake, x_fake, z_fake_mean, log_det= model(x, z)
         
            

            # ================== TRAIN ENCODER ================== #
            model.zero_grad()
            # WITH THE GENERATIVE LOSS
            encoder_score = discriminator(x, z_fake)
            loss_encoder = encoder_score.mean()

            # WITH THE SUPERVISED LOSS
            if sup_flag.sum() > 0:
                label_z = z_fake_mean[sup_flag, :num_labels]
                sup_loss = celoss(label_z, label)
            else:
                sup_loss = torch.zeros([1], device=device)
            loss_encoder = loss_encoder + sup_loss * args.sup_coef-alpha*log_det
            loss_encoder.backward()
            encoder_optimizer.step()

            prior_optimizer.step()

            # ================== TRAIN GENERATOR ================== #
            model.zero_grad()

            decoder_score = discriminator(x_fake, z)
            # with scaling clipping for stabilization
            r_decoder = torch.exp(decoder_score.detach())
            s_decoder = r_decoder.clamp(0.5, 2)
            loss_decoder = -(s_decoder * decoder_score).mean()

            loss_decoder.backward()
            decoder_optimizer.step()

            model.module.prior.set_zero_grad()
            A_optimizer.step()
            prior_optimizer.step()

        # Print out losses
        if batch_idx == 0 or (batch_idx + 1) % print_every == 0:
            log = ('Train Epoch: {} ({:.0f}%)\tD loss: {:.4f}, Encoder loss: {:.4f}, Sup loss: {:.4f}, '
                   'E_score: {:.4f}, D score: {:.4f}'.format(
                epoch, 100. * batch_idx / len(train_loader),
                loss_d.item(), loss_encoder.item(), loss_decoder.item(), sup_loss.item(),
                encoder_score.mean().item(), decoder_score.mean().item()))
            print(log)
            log_file.write(log + '\n')
            log_file.flush()



if __name__ == '__main__':
    main()
