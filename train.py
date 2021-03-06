import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from visual_concept import EncoderCNN, Decoder
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import torch.nn.functional as F
import time
import random
import json
from data_loader import get_loader
from tensorboardX import SummaryWriter
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def visualization(features):
    pass


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    print("load vocabulary ...")
    # Load vocabulary wrapper
    with open(args.vocab_path, 'r') as f:
        vocab = json.load(f)

    print("build data loader ...")
    # load data
    train_loader = get_loader(root=args.root, origin_file=args.caption_path, split=args.split,
                              img_tags=args.img_tags, vocab=args.vocab_path, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # for i, (imgs, tars, lens) in enumerate(train_loader):
    #     images = imgs
    #     targets = tars
    #     lengths = lens
    #     print(images.shape, targets.shape, len(lengths))
    #     if i == 2:
    #         break

    print("build the models ...")
    # Build the models
    encoder = nn.DataParallel(EncoderCNN()).cuda()
    decoder = nn.DataParallel(Decoder()).cuda()

    params = list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    time_start = time.time()
    total_step = len(train_loader)
    writer = SummaryWriter(log_dir='./log')
    for epoch in range(args.num_epochs):
        for i, (imgs, tars, lens) in enumerate(train_loader):
            images = imgs.cuda()
            targets = tars.float().cuda()

            features = encoder(images)
            outputs = decoder(features)

            # begin with 10.69
            pos = nn.functional.binary_cross_entropy(
                outputs*targets, targets)*1
            # begin with 0.7439
            neg = nn.functional.binary_cross_entropy(
                outputs*(1-targets)+targets, targets)*1

            # begin with 11.4362
            loss = (pos+neg)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                time_end = time.time()
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time:{}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), time_end-time_start))
                time_start = time_end
            writer.add_scalars('three loss', {'loss': loss.item(
            ), 'pos loss': pos.item(), 'neg loss': neg.item()}, i)
            # if i == 10:
            #     writer.close()
        # Save the model checkpoints
        if (epoch+1) % args.save_step == 0:
            torch.save(decoder.state_dict(), os.path.join(
                args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
            torch.save(encoder.state_dict(), os.path.join(
                args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        default='/home/lkk/datasets/coco2014', help='root path')
    parser.add_argument('--model_path', type=str,
                        default='models/', help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str,
                        default='./vocab.json', help='path for vocabulary wrapper')
    parser.add_argument('--split', type=str,
                        default='train', help='train/val/test')
    parser.add_argument('--img_tags', type=str,
                        default='./img_tags.json', help='imgages id and tags')
    parser.add_argument('--caption_path', type=str, default='/home/lkk/datasets/coco2014/dataset_coco.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=100,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1,
                        help='step size for saving trained models')

    # paraneters
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
