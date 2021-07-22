import argparse
import time
import torch
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as torch_f
import torch.optim
from progress.bar import Bar
from termcolor import cprint
import pickle

from data import eval_utils
from models.hourglass import NetStackedHourglass
import process_data
from data.RHD import RHD_DataReader
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
            lambda_hm=1.0,
            lambda_mask=1.0,
            lambda_joint=1.0,
            lambda_dep=1.0,
    ):
        self.lambda_dep = lambda_dep
        self.lambda_hm = lambda_hm
        self.lambda_joint = lambda_joint
        self.lambda_mask = lambda_mask

    def compute_loss(self, preds, targs, infos):
        batch_size = infos['batch_size']

        # compute hm_loss anyway
        loss = torch.Tensor([0]).cuda()
        for pred_skeleton in preds[0]:
            targs_flux = targs['flux_map']  # (B, 20, res, res, 3)
            targs_dis = targs['dis_map']  # (B, 20, res, res, 2)

            flux_dimension = targs_flux.shape[-1]
            dis_dimension = targs_dis.shape[-1]
            count_phalanges = 20
            total_dimension = flux_dimension + dis_dimension

            pred_maps = pred_skeleton.reshape((batch_size, count_phalanges, -1, total_dimension))  # (B, 20, res*res, 5)
            # split it into (B, 20, res*res, 3) and (B, 20, res*res, 2)
            pred_maps = torch.split(pred_maps, split_size_or_sections=[3, 2], dim=-1)
            pred_flux = pred_maps[0].split(split_size=1, dim=1)  # tuple of (B, 1, res*res, 3), len=20
            pred_dis = pred_maps[1].split(split_size=1, dim=1)  # tuple of (B, 1, res*res, 2), len=20

            targs_flux = targs_flux.reshape((batch_size, 20, -1, flux_dimension)).split(split_size=1, dim=1)
            targs_dis = targs_dis.reshape((batch_size, 20, -1, dis_dimension)).split(split_size=1, dim=1)
            for idx in range(count_phalanges):
                pred_flux_i = pred_flux[idx].squeeze()  # (B, 1, 4096, 3)->(B, 4096, 3)
                pred_dis_i = pred_dis[idx].squeeze()  # (B, 1, 4096, 2)->(B, 4096, 2)
                targs_flux_i = targs_flux[idx].squeeze() # (B, 1, 4096, 3)->(B, 4096, 3)
                targs_dis_i = targs_dis[idx].squeeze()  # (B, 1, 4096, 2)->(B, 4096, 2)
                flux_loss = torch_f.mse_loss(pred_flux_i.float(), targs_flux_i.float())
                targs_loss = torch_f.mse_loss(pred_dis_i.float(), targs_dis_i.float())
                loss += 0.5 * flux_loss + 0.5 * targs_loss
        return loss


def main(args):
    best_acc = 0

    # Create the model
    print("\nCREATE NETWORK")
    model = NetStackedHourglass()
    model = model.to(device)

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

    print("\nCREATE DATASET...")

    # The argument needed to process the data
    hand_crop, hand_flip, use_wrist, BL, root_id, rotate, uv_sigma = True, True, True, 'small', 12, 180, 0.0

    '''Generate evaluation dataset'''
    eval_dataset = []
    if args.process_evaluation_data:
        process_data.process_evaluation_data(args)

    eval_dataset = RHD_DataReader_With_File(mode="evaluation", path="data")
    print("Total test dataset size: {}".format(len(eval_dataset)))

    print("\nLOAD DATASET...")

    val_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.test_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    '''Generate training dataset'''
    train_dataset = []
    if args.process_training_data:
        process_data.process_training_data(args)

    train_dataset = RHD_DataReader_With_File(mode="training", path="data")
    # print("Total train dataset size: {}".format(len(train_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    model = torch.nn.DataParallel(model)
    print("\nUSING {} GPUs".format(torch.cuda.device_count()))

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_decay_step, gamma=args.gamma,
        last_epoch=args.start_epoch
    )

    # Start training
    for epoch in range(args.start_epoch, args.epochs + 1):
        print('\nEpoch: %d' % (epoch + 1))
        for i in range(len(optimizer.param_groups)):
            print('group %d lr:' % i, optimizer.param_groups[i]['lr'])

        train(
            train_loader,
            model,
            criterion,
            optimizer,
            args=args,
            epoch=epoch
        )
    
    # Validate the correctness of every 5 epochs
        val_indicator = best_acc
        if epoch % 5 == 0:
            val_indicator = validate(val_loader, model, criterion, args=args)
            print(f'Save skeleton_model.pkl after {epoch} epochs')
            torch.save(model, f'skeleton_model_after_{epoch}_epochs.pkl')
        if val_indicator > best_acc:
            best_acc = val_indicator
        scheduler.step()
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


def train(train_loader, model, criterion, optimizer, args, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    am_loss_hm = AverageMeter()

    last = time.time()
    
    # switch to train mode
    model.train()

    # Create the bar to record the progress
    bar = Bar('\033[31m Train \033[0m', max=len(train_loader))
    for i, sample in enumerate(train_loader):
        data_time.update(time.time() - last)
        results, targets, loss = one_forward_pass(
            sample, model, criterion, args, is_training=True
        )

        # Update the skeleton loss after each sample
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
            'epoch: {epoch:}s | '
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
            lossH=am_loss_hm.avg
        )
        bar.next()
    bar.finish()


def val_loss(preds, targs, args):
    # compute the loss when validating
    
    batch_size = args.train_batch

    # compute hm_loss anyway
    loss = torch.Tensor([0]).cuda()
    for pred_skeleton in preds[0]:
        targs_flux = targs['flux_map']  # (B, 20, res, res, 3)
        targs_dis = targs['dis_map']  # (B, 20, res, res, 2)

        flux_dimension = targs_flux.shape[-1]
        dis_dimension = targs_dis.shape[-1]
        count_phalanges = 20
        total_dimension = flux_dimension + dis_dimension

        pred_maps = pred_skeleton.reshape((batch_size, count_phalanges, -1, total_dimension))  # (B, 20, res*res, 5)
        # split it into (B, 20, res*res, 3) and (B, 20, res*res, 2)
        pred_maps = torch.split(pred_maps, split_size_or_sections=[3, 2], dim=-1)
        pred_flux = pred_maps[0].split(split_size=1, dim=1)  # tuple of (B, 1, res*res, 3), len=20
        pred_dis = pred_maps[1].split(split_size=1, dim=1)  # tuple of (B, 1, res*res, 2), len=20

        targs_flux = targs_flux.reshape((batch_size, 20, -1, flux_dimension)).split(split_size=1, dim=1)
        targs_dis = targs_dis.reshape((batch_size, 20, -1, dis_dimension)).split(split_size=1, dim=1)
        for idx in range(count_phalanges):
            pred_flux_i = pred_flux[idx].squeeze()  # (B, 1, 4096, 3)->(B, 4096, 3)
            pred_dis_i = pred_dis[idx].squeeze()  # (B, 1, 4096, 2)->(B, 4096, 2)
            targs_flux_i = targs_flux[idx].squeeze() # (B, 1, 4096, 3)->(B, 4096, 3)
            targs_dis_i = targs_dis[idx].squeeze()  # (B, 1, 4096, 2)->(B, 4096, 2)
            flux_loss = torch_f.mse_loss(pred_flux_i, targs_flux_i)
            targs_loss = torch_f.mse_loss(pred_dis_i, targs_dis_i)
            loss += 0.5 * flux_loss + 0.5 * targs_loss
    return loss


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
            pred_skeleton = results[-1]
            targs_flux = targets['flux_map']  # (B, 20, res, res, 3)
            targs_dis = targets['dis_map']  # (B, 20, res, res, 2)

            flux_dimension = targs_flux.shape[-1]
            dis_dimension = targs_dis.shape[-1]
            count_phalanges = 20
            total_dimension = flux_dimension + dis_dimension

            pred_maps = pred_skeleton.reshape((args.train_batch, count_phalanges, -1, total_dimension))  # (B, 20, res*res, 5)
            # split it into (B, 20, res*res, 3) and (B, 20, res*res, 2)
            pred_maps = torch.split(pred_maps, split_size_or_sections=[3, 2], dim=-1)
            pred_flux = pred_maps[0].split(split_size=1, dim=1)  # tuple of (B, 1, res*res, 3), len=20
            pred_dis = pred_maps[1].split(split_size=1, dim=1)  # tuple of (B, 1, res*res, 2), len=20
            pred_kp = eval_utils.maps_to_kp(pred_flux, pred_dis)
            targs_kp = eval_utils.maps_to_kp(targs_flux, targs_dis)
            val = eval_utils.MeanEPE(pred_kp, targs_kp)

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
