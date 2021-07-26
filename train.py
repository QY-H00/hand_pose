import argparse
import time
import random
import os.path as osp
import torch
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from tensorboardX import SummaryWriter
from progress.bar import Bar
from termcolor import cprint

import process_data
from data import eval_utils
from models.hourglass import NetStackedHourglass
from data.RHD import RHD_DataReader_With_File

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SkeletonLoss:
    """Computes the loss of skeleton formed of dis_map and flux_map"""

    def __init__(
            self,
            lambda_dis=1.0,
            lambda_kps=1.0,
            lambda_flux=1.0,
            lambda_mask=1.0
    ):
        self.lambda_dis = lambda_dis
        self.lambda_flux = lambda_flux
        self.lambda_kps = lambda_kps
        self.lambda_mask = lambda_mask

    def compute_loss(self, preds, targs, infos):
        # target
        targs_flux_map = targs['flux_map']  # (B, 20, res, res, 3)
        targs_flux = targs_flux_map[:, :, :, :, :2]  # (B, 20, res, res, 2)
        targs_mask = targs_flux_map[:, :, :, :, 2]  # (B, 20, res, res, 1)
        targs_dis = targs['dis_map']  # (B, 20, res, res, 2)
        targs_kps = targs['kp2d']  # (B, 21, 2)

        # prediction
        pred_skeleton = preds[0][-1]  # (B, 20, res, res, 5)
        pred_maps = torch.split(pred_skeleton, split_size_or_sections=[3, 2], dim=-1)
        pred_flux = pred_maps[0][:, :, :, :2]  # (B, 20, res, res, 2)
        pred_mask = pred_maps[0][:, :, :, 2]  # (B, 20, res, res, 1)
        pred_dis = pred_maps[1]  # (B, 20, res, res, 2)
        pred_kps = preds[1][-1]  # (B, 21, 2)

        # compute loss
        flux_loss = torch.sum(torch.sum(torch.abs(pred_flux - targs_flux), dim=-1) * pred_mask)
        dis_loss = torch.sum(torch.sum(torch.abs(pred_dis - targs_dis), dim=-1) * pred_mask)
        mask_loss = torch.sum(torch.abs(pred_mask - targs_mask))
        kps_loss = torch.sum(torch.abs(pred_kps - targs_kps))
        maps_loss = self.lambda_flux * flux_loss + self.lambda_dis * dis_loss + self.lambda_mask * mask_loss
        loss = maps_loss + self.lambda_kps * kps_loss

        return loss


def main(args):
    """Main process"""

    '''Set up the network'''
    print("\nCREATE NETWORK")
    model = NetStackedHourglass()
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    print("\nUSING {} GPUs".format(torch.cuda.device_count()))

    criterion = SkeletonLoss()

    optimizer = torch.optim.Adam(
        [
            {
                'params': model.parameters(),
                'initial_lr': args.learning_rate
            },
        ],
        lr=args.learning_rate,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_decay_step, gamma=args.gamma,
        last_epoch=args.start_epoch
    )

    print("\nCREATE DATASET...")

    # The argument needed to process the data
    hand_crop, hand_flip, use_wrist, BL, root_id, rotate, uv_sigma = True, True, True, 'small', 12, 180, 0.0

    '''Generate evaluation dataset'''
    if args.process_evaluation_data:
        process_data.process_evaluation_data(args)

    eval_dataset = RHD_DataReader_With_File(mode="evaluation", path="data")
    print("Total test dataset size: {}".format(len(eval_dataset)))

    val_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.test_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    print("f")

    '''Generate training dataset'''
    if args.process_training_data:
        print("?")
        process_data.process_training_data(args)

    train_dataset = RHD_DataReader_With_File(mode="training", path="data")
    print("Total train dataset size: {}".format(len(train_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    '''Set up the monitor'''
    best_acc = 0
    loss_log_dir = osp.join('tensorboard', 'loss')
    loss_writer = SummaryWriter(log_dir=loss_log_dir)
    val_log_dir = osp.join('tensorboard', 'val')
    val_writer = SummaryWriter(log_dir=val_log_dir)

    '''Start Training'''
    for epoch in range(args.start_epoch, args.epochs + 1):
        print('\nEpoch: %d' % (epoch + 1))
        for i in range(len(optimizer.param_groups)):
            print('group %d lr:' % i, optimizer.param_groups[i]['lr'])

        loss_avg = train(
            train_loader,
            model,
            criterion,
            optimizer,
            args=args,
            epoch=epoch
        )

        # Validate the correctness every 5 epochs
        val_indicator = best_acc
        if epoch % 5 == 0:
            val_indicator = validate(val_loader, model, criterion, args=args)
            print(f'Save skeleton_model.pkl after {epoch} epochs')
            val_writer.add_scalar('Validation Indicator', val_indicator, epoch)
            torch.save(model, f'skeleton_model_after_{epoch}_epochs.pkl')
        if val_indicator > best_acc:
            best_acc = val_indicator

        # Draw the loss curve and validation indicator curve
        loss_writer.add_scalar('Loss', loss_avg, epoch)

        scheduler.step()

    '''Save Model'''
    print("Save skeleton_model.pkl after total training")
    torch.save(model, 'skeleton_model.pkl')
    cprint('All Done', 'yellow', attrs=['bold'])

    return 0  # end of main


def one_forward_pass(sample, model, criterion, args, is_training=True):
    # forward pass the sample into the model and compute the loss of corresponding sample
    """ prepare target """
    img = sample['img_crop'].to(device, non_blocking=True)
    kp2d = sample['uv_crop'].to(device, non_blocking=True)

    ''' skeleton map generation '''
    flux_map = sample['flux_map'].to(device, non_blocking=True)
    dis_map = sample['dis_map'].to(device, non_blocking=True)

    ''' prepare infos '''
    infos = {
        'batch_size': args.train_batch
    }

    targets = {
        'clr': img,
        'flux_map': flux_map,
        'dis_map': dis_map,
        'kp2d': kp2d
    }
    ''' ----------------  Forward Pass  ---------------- '''
    results = model(img)
    ''' ----------------  Forward End   ---------------- '''

    loss = torch.Tensor([0]).cuda()
    if not is_training:
        return results, {**targets, **infos}, loss
    else:
        ''' compute losses '''
        loss = criterion.compute_loss(results, targets, infos)
        return results, {**targets, **infos}, loss


# 24/07/2021 15:18: Use tensorboard to draw the loss curve
def train(train_loader, model, criterion, optimizer, args, epoch):
    """Train process"""
    '''Set up configuration'''
    batch_time = AverageMeter()
    data_time = AverageMeter()
    am_loss = AverageMeter()
    last = time.time()
    model.train()
    bar = Bar('\033[31m Train \033[0m', max=len(train_loader))

    if epoch == 0:
        sample_idxs = []
        for i in range(5):
            sample_idx = random.randint(0, len(train_loader))
            sample_idxs.append(sample_idx)
        for i, sample in enumerate(train_loader):
            for idx in sample_idxs:
                if idx == i:
                    image = sample["img_crop"][0]
                    flux = sample["flux_map"][0]
                    dis = sample["dis_map"][0]
                    eval_utils.draw_maps(flux, dis, image, idx)

    '''Start Training'''
    for i, sample in enumerate(train_loader):
        data_time.update(time.time() - last)
        results, targets, loss = one_forward_pass(
            sample, model, criterion, args, is_training=True
        )

        '''Update the loss after each sample'''
        am_loss.update(
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
            'epoch: {epoch:} | '
            'd: {data:.2f}s | '
            'b: {bt:.2f}s | '
            't: {total:}s | '
            'eta:{eta:}s | '
            'lH: {lossH:.5f} | '
        ).format(
            epoch=epoch,
            batch=i + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            lossH=am_loss.avg
        )
        bar.next()
    bar.finish()
    return am_loss.avg


# 24/07/2021 15:00: Maybe work now
def validate(val_loader, model, criterion, args, stop=-1):
    # switch to evaluate mode
    am_accH = AverageMeter()

    model.eval()
    bar = Bar('\033[33m Eval  \033[0m', max=len(val_loader))
    with torch.no_grad():
        for i, metas in enumerate(val_loader):
            results, targets, loss = one_forward_pass(
                metas, model, criterion, args=args, is_training=False
            )
            pred_kps = results[1][-1]  # (B, 21, 2)
            targs_kps = targets["kp2d"]  # (B, 21, 2)

            val = eval_utils.MeanEPE(pred_kps, targs_kps)

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

    # Dataset setting
    parser.add_argument(
        '--process_training_data',
        default=False,
        action='store_true',
        help='true if the data has been processed'
    )

    # Dataset setting
    parser.add_argument(
        '--process_evaluation_data',
        default=False,
        action='store_true',
        help='true if the data has been processed'
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

    # Dataset setting
    parser.add_argument(
        '-cc',
        '--checking_cycle',
        type=int,
        default=5,
        help='How many batches to save the model at once'
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
        type=int,
        metavar='N',
        help='train batch size'
    )
    parser.add_argument(
        '-tb', '--test_batch',
        default=32,
        type=int,
        metavar='N',
        help='test batch size'
    )

    parser.add_argument(
        '-lr', '--learning_rate',
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

    main(parser.parse_args())
