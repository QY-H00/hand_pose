import os

os.environ["CUDA_VISIBLE_DEVICES"] = '7'

import os.path as osp

import argparse
import time
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as torch_f
import torch.optim
from progress.bar import Bar
from termcolor import cprint
from datetime import datetime
from tensorboardX import SummaryWriter

import data.eval_utils as eval_utils
from data.eval_utils import AverageMeter
from data.RHD import RHD_DataReader_With_File
from models.hourglass import NetStackedHourglass
from decoder.softargmax_decoder import SoftargmaxDecoder
from loss.heatmap_loss import HeatmapLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def main(args):
    best_acc = 0

    print("\nCREATE NETWORK")
    encoder = NetStackedHourglass()
    decoder = SoftargmaxDecoder()
    model = nn.Sequential(encoder, decoder)
    model = model.to(device)

    criterion = HeatmapLoss()

    optimizer = torch.optim.Adam(
        [
            {
                'params': model.parameters(),
                'initial_lr': args.learning_rate
            }
        ],
        lr=args.learning_rate,
    )

    print("\nCREATE DATASET...")
    hand_crop, hand_flip, use_wrist, BL, root_id, rotate, uv_sigma = True, True, True, 'small', 12, 180, 0.0
    # train_dataset = RHD_DataReader(path=args.data_root, mode='training', hand_crop=hand_crop, use_wrist_coord=use_wrist,
    #                                sigma=5,
    #                                data_aug=False, uv_sigma=uv_sigma, rotate=rotate, BL=BL, root_id=root_id,
    #                                right_hand_flip=hand_flip, crop_size_input=256)
    #
    # val_dataset = RHD_DataReader(path=args.data_root, mode='evaluation', hand_crop=hand_crop, use_wrist_coord=use_wrist,
    #                              sigma=5,
    #                              data_aug=False, uv_sigma=uv_sigma, rotate=rotate, BL=BL, root_id=root_id,
    #                              right_hand_flip=hand_flip, crop_size_input=256)
    train_dataset = RHD_DataReader_With_File(mode="training", path="data_v2.0")
    val_dataset = RHD_DataReader_With_File(mode="evaluation", path="data_v2.0")

    print("Total train dataset size: {}".format(len(train_dataset)))
    print("Total test dataset size: {}".format(len(val_dataset)))

    print("\nLOAD DATASET...")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    model = torch.nn.DataParallel(model)
    print("\nUSING {} GPUs".format(torch.cuda.device_count()))

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_decay_step, gamma=args.gamma,
        last_epoch=args.start_epoch
    )

    best_acc = 0
    loss_log_dir = osp.join('tensorboard', f'baseline_loss_{datetime.now().strftime("%Y%m%d_%H%M")}')
    loss_writer = SummaryWriter(log_dir=loss_log_dir)
    error_log_dir = osp.join('tensorboard', f'baseline_error_{datetime.now().strftime("%Y%m%d_%H%M")}')
    error_writer = SummaryWriter(log_dir=error_log_dir)

    for epoch in range(args.start_epoch, args.epochs + 1):
        print('\nEpoch: %d' % (epoch + 1))
        for i in range(len(optimizer.param_groups)):
            print('group %d lr:' % i, optimizer.param_groups[i]['lr'])
        #############  train for on epoch  ###############
        loss_avg = train(
            train_loader,
            model,
            criterion,
            optimizer,
            args=args,
        )
        ##################################################
        val_indicator = best_acc
        if epoch % 5 == 4:
            train_indicator = validate(train_loader, model, criterion, args=args)
            val_indicator = validate(val_loader, model, criterion, args=args)
            print(f'Save skeleton_model.pkl after {epoch + 1} epochs')
            error_writer.add_scalar('Validation Indicator', val_indicator, epoch)
            error_writer.add_scalar('Training Indicator', train_indicator, epoch)
            torch.save(model, f'trained_model_hm_soft/skeleton_model_after_{epoch + 1}_epochs.pkl')
        if val_indicator > best_acc:
            best_acc = val_indicator
        scheduler.step()

        # Draw the loss curve and validation indicator curve
        loss_writer.add_scalar('Loss', loss_avg.avg, epoch)
    print("\nSave model as hourglass_model.pkl...")
    torch.save(model, 'trained_model_hm_soft/hourglass_model.pkl')
    cprint('All Done', 'yellow', attrs=['bold'])
    return 0  # end of main


def one_forward_pass(sample, model, criterion, args, is_training=True):
    """ prepare target """
    img = sample['img_crop'].to(device, non_blocking=True)
    kp2d = sample['uv_crop'].to(device, non_blocking=True)
    vis = sample['vis21'].to(device, non_blocking=True)

    ''' heatmap generation '''
    # need to modify
    hm = sample['hm'].to(device, non_blocking=True)

    ''' prepare infos '''
    # need to modify
    hm_veil = sample['hm_veil'].to(device, non_blocking=True)
    infos = {
        'hm_veil': hm_veil,
        'batch_size': args.train_batch
    }

    targets = {
        'clr': img,
        'hm': hm,
        'kp2d': kp2d,
        'vis': vis
    }
    ''' ----------------  Forward Pass  ---------------- '''
    results = model(img)
    ''' ----------------  Forward End   ---------------- '''

    loss = torch.Tensor([0]).cuda()
    if not is_training:
        return results, {**targets, **infos}, loss

    ''' compute losses '''
    loss = criterion.compute_loss(results, targets, infos)

    return results, {**targets, **infos}, loss


def train(train_loader, model, criterion, optimizer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    am_loss_hm = AverageMeter()

    last = time.time()
    # switch to train mode
    model.train()
    bar = Bar('\033[31m Train \033[0m', max=len(train_loader))
    for i, sample in enumerate(train_loader):
        data_time.update(time.time() - last)
        results, targets, loss = one_forward_pass(
            sample, model, criterion, args, is_training=True
        )
        am_loss_hm.update(
            loss.item(), targets['batch_size']
        )

        ''' backward and step '''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ''' progress '''
        batch_time.update(time.time() - last)
        last = time.time()
        bar.suffix = (
            '({batch}/{size}) '
            'lH: {lossH:.5f} | '
        ).format(
            batch=i + 1,
            size=len(train_loader),
            lossH=am_loss_hm.avg
        )
        bar.next()
    bar.finish()

    return am_loss_hm


def validate(val_loader, model, criterion, args, stop=-1):
    # switch to evaluate mode
    am_accH = AverageMeter()

    model.eval()
    bar = Bar('\033[33m Eval  \033[0m', max=len(val_loader))
    with torch.no_grad():
        for i, metas in enumerate(val_loader):
            results, targets, loss = one_forward_pass(
                metas, model, criterion, args, is_training=False
            )
            pred_kps = results[0][-1]
            targ_kps = targets['kp2d']
            val = eval_utils.MeanEPE(pred_kps, targ_kps)
            bar.suffix = (
                '({batch}/{size}) '
                'accH: {accH:.4f} | '
            ).format(
                batch=i + 1,
                size=len(val_loader),
                accH=val,
            )
            am_accH.update(val, args.train_batch)
            bar.next()
            if stop != -1 and i >= stop:
                break
        bar.finish()
        print("accH: {}".format(am_accH.avg))
    return am_accH.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch Train Hourglass On 2D Keypoint Detection')
    # Dataset setting
    parser.add_argument(
        '-dr',
        '--data_root',
        type=str,
        default='RHD_published_v2',
        help='dataset root directory'
    )

    # Model Structure
    # hourglass:
    parser.add_argument(
        '-hgs',
        '--hg-stacks',
        default=2,
        type=int,
        metavar='N',
        help='Number of hourglasses to stack'
    )
    parser.add_argument(
        '-hgb',
        '--hg-blocks',
        default=1,
        type=int,
        metavar='N',
        help='Number of residual modules at each location in the hourglass'
    )
    parser.add_argument(
        '-nj',
        '--njoints',
        default=21,
        type=int,
        metavar='N',
        help='Number of heatmaps calsses (hand joints) to predict in the hourglass'
    )

    parser.add_argument(
        '-r', '--resume',
        dest='resume',
        action='store_true',
        help='whether to load checkpoint (default: none)'
    )
    parser.add_argument(
        '-e', '--evaluate',
        dest='evaluate',
        action='store_true',
        help='evaluate model on validation set'
    )
    parser.add_argument(
        '-d', '--debug',
        dest='debug',
        action='store_true',
        default=False,
        help='show intermediate results'
    )

    # Training Parameters
    parser.add_argument(
        '-j', '--workers',
        default=8,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 8)'
    )
    parser.add_argument(
        '--epochs',
        default=100,
        type=int,
        metavar='N',
        help='number of total epochs to run'
    )
    parser.add_argument(
        '-se', '--start_epoch',
        default=0,
        type=int,
        metavar='N',
        help='manual epoch number (useful on restarts)'
    )
    parser.add_argument(
        '-b', '--train_batch',
        default=32,
        type=int
    )
    parser.add_argument(
        '-tb', '--test_batch',
        default=32,
        type=int
    )

    parser.add_argument(
        '-lr', '--learning-rate',
        default=1.0e-4,
        type=float,
        metavar='LR',
        help='initial learning rate'
    )
    parser.add_argument(
        "--lr_decay_step",
        default=50,
        type=int,
        help="Epochs after which to decay learning rate",
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.1,
        help='LR is multiplied by gamma on schedule.'
    )
    parser.add_argument(
        "--net_modules",
        nargs="+",
        default=['seed'],
        type=str,
        help="sub modules contained in model"
    )
    parser.add_argument(
        '--ups_loss',
        dest='ups_loss',
        action='store_true',
        help='Calculate upstream loss',
        default=True
    )

    main(parser.parse_args())
