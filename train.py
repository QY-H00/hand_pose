import argparse
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '6, 7'

import os.path as osp
import torch
import torch.utils.data
import torch.nn.parallel
import torch.nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from datetime import datetime

from tensorboardX import SummaryWriter
from progress.bar import Bar
from termcolor import cprint

import process_data
from data import eval_utils
from models.hourglass import NetStackedHourglass_2
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


# class SkeletonLoss:
#     """Computes the loss of skeleton formed of dis_map and flux_map"""
#
#     def __init__(
#             self,
#             lambda_dis=1.0,
#             lambda_kps=0.1,
#             lambda_flux=1.0,
#             lambda_mask=1.0
#     ):
#         self.lambda_dis = lambda_dis
#         self.lambda_flux = lambda_flux
#         self.lambda_kps = lambda_kps
#         self.lambda_mask = lambda_mask
#
#     def compute_loss(self, preds, targs, infos):
#         # target
#         targs_flux_map = targs['flux_map']  # (B, 20, res, res, 3)
#         targs_flux = targs_flux_map[:, :, :, :, :2]  # (B, 20, res, res, 2)
#         targs_mask = targs_flux_map[:, :, :, :, 2, None]  # (B, 20, res, res, 1)
#         targs_dis = targs['dis_map']  # (B, 20, res, res, 2)
#         targs_kps = targs['kp2d']  # (B, 21, 2)
#
#         flux_loss = F.l1_loss(torch.FloatTensor(0), torch.FloatTensor(0))
#         mask_loss = F.l1_loss(torch.FloatTensor(0), torch.FloatTensor(0))
#         dis_loss = F.l1_loss(torch.FloatTensor(0), torch.FloatTensor(0))
#         kps_loss = 0
#         loss = 0
#
#         # prediction
#         for layer in range(len(preds[0])):
#             pred_flux = preds[0][-1]
#             pred_flux = pred_flux.reshape(pred_flux.shape[0], 20, pred_flux.shape[2], pred_flux.shape[3], -1)
#             pred_mask = preds[1][-1]  # (B, 20, res, res, 1)
#             pred_mask = pred_mask.reshape(pred_mask.shape[0], 20, pred_mask.shape[2], pred_mask.shape[3], -1)
#             pred_dis = preds[2][-1]  # (B, 20, res, res, 2)
#             pred_dis = pred_dis.reshape(pred_dis.shape[0], 20, pred_dis.shape[2], pred_dis.shape[3], -1)
#             pred_kps = preds[3][-1]  # (B, 21, 2)
#
#             # _flux_loss = F.l1_loss(pred_flux * pred_mask, targs_flux * pred_mask)
#             # _dis_loss = F.l1_loss(pred_dis * pred_mask, targs_dis * pred_mask)
#             # _mask_loss = F.l1_loss(pred_mask, targs_mask)
#             # _kps_loss = F.l1_loss(pred_kps, targs_kps)
#             #
#             # flux_loss += self.lambda_flux * _flux_loss
#             # mask_loss += self.lambda_mask * _mask_loss
#             # dis_loss += self.lambda_dis * _dis_loss
#             # kps_loss += self.lambda_kps * _kps_loss
#
#             # loss += flux_loss + dis_loss + mask_loss + kps_loss
#
#             _kps_loss = F.l1_loss(pred_kps, targs_kps)
#             kps_loss += self.lambda_kps * _kps_loss
#             loss += kps_loss
#
#         return loss, flux_loss, dis_loss, mask_loss, kps_loss


class SkeletonLoss_2:
    """Computes the loss of skeleton formed of dis_map and flux_map"""

    def __init__(
            self,
            lambda_vecs=1.0,
            lambda_diss=1.0,
            lambda_ske=1.0,
            lambda_kps=0.004
    ):
        self.lambda_vecs = lambda_vecs
        self.lambda_diss = lambda_diss
        self.lambda_ske = lambda_ske
        self.lambda_kps = lambda_kps

    def compute_loss(self, preds, targs, infos):
        # target
        targs_front_vec = targs['front_vec']  # (B, 20, res, res, 2)
        targs_back_vec = targs['back_vec']  # (B, 20, res, res, 2)
        targs_front_dis = targs['front_dis']  # (B, 20, res, res, 1)
        targs_back_dis = targs['back_dis']  # (B, 20, res, res, 1)
        targs_ske_mask = targs['ske_mask']  # (B, 20, res, res, 1)
        targs_weit_map = targs['weit_map']  # (B, 20, res, res, 1)
        targs_kps = targs['kp2d']  # (B, 21, 2)
        vis = targs['vis'][:, :, None]  # (B, 21, )
        batch_size = targs_kps.shape[0]
        targs_front_vec = targs_front_vec.reshape(batch_size, 20, -1).split(1, 1)
        targs_back_vec = targs_back_vec.reshape(batch_size, 20, -1).split(1, 1)
        targs_front_dis = targs_front_dis.reshape(batch_size, 20, -1).split(1, 1)
        targs_back_dis = targs_back_dis.reshape(batch_size, 20, -1).split(1, 1)
        targs_ske_mask = targs_ske_mask.reshape(batch_size, 20, -1).split(1, 1)

        # Loss Set Up
        vec_loss = torch.Tensor([0]).to(device, non_blocking=True)
        ske_loss = torch.Tensor([0]).to(device, non_blocking=True)
        dis_loss = torch.Tensor([0]).to(device, non_blocking=True)
        kps_loss = torch.Tensor([0]).to(device, non_blocking=True)

        # prediction
        for layer in range(len(preds[0])):
            front_vec = preds[0][layer]
            front_dis = preds[1][layer]
            back_vec = preds[2][layer]
            back_dis = preds[3][layer]
            ske_mask = preds[4][layer]  # (B, 20, res, res, 1)
            pred_kps = preds[5][layer]  # (B, 21, 2)

            front_vec = front_vec.reshape(front_vec.shape[0], 20, front_vec.shape[2], front_vec.shape[3], -1)
            front_dis = front_dis.reshape(front_dis.shape[0], 20, front_dis.shape[2], front_dis.shape[3], -1)
            back_vec = back_vec.reshape(back_vec.shape[0], 20, back_vec.shape[2], back_vec.shape[3], -1)
            back_dis = back_dis.reshape(back_dis.shape[0], 20, back_dis.shape[2], back_dis.shape[3], -1)
            ske_mask = ske_mask.reshape(ske_mask.shape[0], 20, ske_mask.shape[2], ske_mask.shape[3], -1)

            front_vec = targs_weit_map * front_vec
            front_dis = targs_weit_map * front_dis
            back_vec = targs_weit_map * back_vec
            back_dis = targs_weit_map * back_dis
            ske_mask = ske_mask * back_dis
            front_vec = front_vec.reshape(batch_size, 20, -1).split(1, 1)
            front_dis = front_dis.reshape(batch_size, 20, -1).split(1, 1)
            back_vec = back_vec.reshape(batch_size, 20, -1).split(1, 1)
            back_dis = back_dis.reshape(batch_size, 20, -1).split(1, 1)
            ske_mask = ske_mask.reshape(batch_size, 20, -1).split(1, 1)

            for idx in range(20):
                front_veci = front_vec[idx].squeeze()  # (B, 1, 4096)->(B, 4096)
                front_disi = front_dis[idx].squeeze()  # (B, 1, 4096)->(B, 4096)
                back_veci = back_vec[idx].squeeze()  # (B, 1, 4096)->(B, 4096)
                back_disi = back_dis[idx].squeeze()  # (B, 1, 4096)->(B, 4096)
                ske_maski = ske_mask[idx].squeeze()  # (B, 1, 4096)->(B, 4096)
                targs_front_veci = targs_front_vec[idx].squeeze()  # (B, 1, 4096)->(B, 4096)
                targs_front_disi = targs_front_dis[idx].squeeze()  # (B, 1, 4096)->(B, 4096)
                targs_back_veci = targs_back_vec[idx].squeeze()  # (B, 1, 4096)->(B, 4096)
                targs_back_disi = targs_back_dis[idx].squeeze()  # (B, 1, 4096)->(B, 4096)
                targs_ske_maski = targs_ske_mask[idx].squeeze()  # (B, 1, 4096)->(B, 4096)

                _vec_loss = (F.l1_loss(front_veci, targs_front_veci) + F.l1_loss(back_veci, targs_back_veci)) / 2
                _dis_loss = F.l1_loss(front_disi, targs_front_disi) \
                            + F.l1_loss(back_disi, targs_back_disi)
                _ske_loss = F.l1_loss(ske_maski, targs_ske_maski)

                vec_loss += _vec_loss * self.lambda_vecs
                dis_loss += _dis_loss * self.lambda_diss
                ske_loss += _ske_loss * self.lambda_ske

            _kps_loss = eval_utils.MeanEPE(pred_kps * vis, targs_kps * vis)
            kps_loss += _kps_loss * self.lambda_kps

        vec_loss = vec_loss / len(preds[0])
        dis_loss = dis_loss / len(preds[0])
        ske_loss = ske_loss / len(preds[0])
        kps_loss = kps_loss / len(preds[0])
        loss = vec_loss + dis_loss + ske_loss + kps_loss

        return loss, vec_loss, dis_loss, ske_loss, kps_loss


def main(args):
    """Main process"""

    '''Set up the network'''

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    print("\nCREATE NETWORK")
    # # Version 1.0
    # model = NetStackedHourglass()
    # Version 2.0
    model = NetStackedHourglass_2()
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    print("\nUSING {} GPUs".format(torch.cuda.device_count()))
    # # Version 1.0
    # criterion = SkeletonLoss()
    # Version 2.0
    criterion = SkeletonLoss_2()
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

    '''Generate dataset'''

    print("\nCREATE DATASET...")
    # Create evaluation dataset
    if args.process_evaluation_data:
        process_data.process_evaluation_data(args)

    eval_dataset = RHD_DataReader_With_File(mode="evaluation", path="data_v2.0")

    val_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    print("Total test dataset size: {}".format(len(eval_dataset)))

    # Create training dataset
    if args.process_training_data:
        process_data.process_training_data(args)

    train_dataset = RHD_DataReader_With_File(mode="evaluation", path="data_v2.0")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    print("Total train dataset size: {}".format(len(train_dataset)))

    '''Set up the monitor'''

    best_acc = 0
    loss_log_dir = osp.join('tensorboard', f'loss_{datetime.now().strftime("%Y%m%d_%H%M")}')
    loss_writer = SummaryWriter(log_dir=loss_log_dir)
    error_log_dir = osp.join('tensorboard', f'error_{datetime.now().strftime("%Y%m%d_%H%M")}')
    error_writer = SummaryWriter(log_dir=error_log_dir)

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
        if epoch % 5 == 4:
            # train_indicator = validate(train_loader, model, criterion, args=args)
            val_indicator = validate(val_loader, model, criterion, args=args)
            print(f'Save skeleton_model.pkl after {epoch + 1} epochs')
            error_writer.add_scalar('Validation Indicator', val_indicator, epoch)
            # error_writer.add_scalar('Training Indicator', train_indicator, epoch)
            torch.save(model, f'trained_model_v1.2/skeleton_model_after_{epoch + 1}_epochs.pkl')
        if val_indicator > best_acc:
            best_acc = val_indicator

        # Draw the loss curve and validation indicator curve
        loss_writer.add_scalar('Loss', loss_avg, epoch)

        scheduler.step()

    '''Save Model'''

    print("Save skeleton_model.pkl after total training")
    torch.save(model, 'trained_model/skeleton_model.pkl')
    cprint('All Done', 'yellow', attrs=['bold'])

    return 0  # end of main


def one_forward_pass(sample, model, criterion, args, is_training=True):
    # forward pass the sample into the model and compute the loss of corresponding sample
    """ prepare target """
    img = sample['img_crop'].to(device, non_blocking=True)
    kp2d = sample['uv_crop'].to(device, non_blocking=True)
    vis = sample['vis21'].to(device, non_blocking=True)

    ''' skeleton map generation '''
    front_vec = sample['front_vec'].to(device, non_blocking=True)
    front_dis = sample['front_dis'].to(device, non_blocking=True)
    back_vec = sample['back_vec'].to(device, non_blocking=True)
    back_dis = sample['back_dis'].to(device, non_blocking=True)
    ske_mask = sample['skeleton'].to(device, non_blocking=True)
    weit_map = sample['weit_map'].to(device, non_blocking=True)

    ''' prepare infos '''
    infos = {
        'batch_size': args.train_batch
    }

    targets = {
        'clr': img,
        'front_vec': front_vec,
        'front_dis': front_dis,
        'back_vec': back_vec,
        'back_dis': back_dis,
        'ske_mask': ske_mask,
        'kp2d': kp2d,
        'vis': vis,
        'weit_map': weit_map
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


def train(train_loader, model, criterion, optimizer, args, epoch):
    """Train process"""
    '''Set up configuration'''
    batch_time = AverageMeter()
    data_time = AverageMeter()
    am_loss = AverageMeter()
    vec_loss = AverageMeter()
    dis_loss = AverageMeter()
    ske_loss = AverageMeter()
    kps_loss = AverageMeter()
    last = time.time()
    model.train()
    bar = Bar('\033[31m Train \033[0m', max=len(train_loader))

    '''Start Training'''
    for i, sample in enumerate(train_loader):
        data_time.update(time.time() - last)
        results, targets, loss = one_forward_pass(
            sample, model, criterion, args, is_training=True
        )

        '''Update the loss after each sample'''

        am_loss.update(
            loss[0].item(), targets['batch_size']
        )
        vec_loss.update(
            loss[1].item(), targets['batch_size']
        )
        dis_loss.update(
            loss[2].item(), targets['batch_size']
        )
        ske_loss.update(
            loss[3].item(), targets['batch_size']
        )
        kps_loss.update(
            loss[4].item(), targets['batch_size']
        )

        ''' backward and step '''
        optimizer.zero_grad()
        loss[1].backward()
        # if epoch < 60:
        #     # loss[1].backward(retain_graph=True)
        #     # loss[2].backward(retain_graph=True)
        #     # loss[3].backward()
        # else:
        #     loss[0].backward()
        optimizer.step()

        ''' progress '''
        batch_time.update(time.time() - last)
        last = time.time()
        bar.suffix = (
            '({batch}/{size}) '
            'lH: {lossH:.5f} | '
            'lV: {lossV:.5f} | '
            'lD: {lossD:.5f} | '
            'lM: {lossM:.5f} | '
            'lK: {lossK:.5f} | '
        ).format(
            batch=i + 1,
            size=len(train_loader),
            lossH=am_loss.avg,
            lossV=vec_loss.avg,
            lossD=dis_loss.avg,
            lossM=ske_loss.avg,
            lossK=kps_loss.avg
        )
        bar.next()
    bar.finish()
    return am_loss.avg


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
            pred_kps = results[5][-1]  # (B, 21, 2)
            targ_kps = targets["kp2d"]  # (B, 21, 2)
            vis = targets['vis'][:, :, None]

            val = eval_utils.MeanEPE(pred_kps * vis, targ_kps * vis)
            #             print(targ_kps[idx])
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
        default=119,
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
