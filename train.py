import argparse
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '7'

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
from models.hourglass import NetStackedHourglass_2, NetStackedHourglass
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


class SkeletonBaseModel:
    class SkeletonLoss:
        """Computes the loss of skeleton formed of dis_map and flux_map"""

        def __init__(
                self,
                lambda_vecs=1.0,
                lambda_diss=2.0,
                lambda_ske=2.0,
                lambda_kps=0.1
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
            vis = targs['vis'][:, :, None]  # (B, 21, 1)
            bone_vis = eval_utils.generate_bone_vis(vis, device)  # (B, 20, 1)
            bone_vis = bone_vis.split(1, 1)
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
                ske_mask = targs_weit_map * ske_mask
                front_vec = front_vec.reshape(batch_size, 20, -1).split(1, 1)
                front_dis = front_dis.reshape(batch_size, 20, -1).split(1, 1)
                back_vec = back_vec.reshape(batch_size, 20, -1).split(1, 1)
                back_dis = back_dis.reshape(batch_size, 20, -1).split(1, 1)
                ske_mask = ske_mask.reshape(batch_size, 20, -1).split(1, 1)

                for idx in range(20):
                    bone_visi = bone_vis[idx].squeeze()  # (B, 1, 1)->(B, 1)
                    bone_visi = bone_visi[:, None]
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

                    _vec_loss = (F.l1_loss(front_veci * bone_visi, targs_front_veci * bone_visi)
                                 + F.l1_loss(back_veci * bone_visi, targs_back_veci * bone_visi)) / 2
                    _dis_loss = (F.l1_loss(front_disi * bone_visi, targs_front_disi * bone_visi)
                                 + F.l1_loss(back_disi * bone_visi, targs_back_disi * bone_visi)) / 2
                    _ske_loss = F.l1_loss(ske_maski * bone_visi, targs_ske_maski * bone_visi)

                    # _vec_loss = (F.l1_loss(front_veci, targs_front_veci)
                    #              + F.l1_loss(back_veci, targs_back_veci)) / 2
                    # _dis_loss = (F.l1_loss(front_disi, targs_front_disi)
                    #              + F.l1_loss(back_disi, targs_back_disi)) / 2
                    # _ske_loss = (F.l1_loss(ske_maski, targs_ske_maski)
                    #              + F.l1_loss(ske_maski, targs_ske_maski)) / 2

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
            map_loss = vec_loss + dis_loss + ske_loss

            return loss, vec_loss, dis_loss, ske_loss, kps_loss, map_loss

    def main(self, args):
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
        criterion = self.SkeletonLoss()
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

        last = time.time()
        train_dataset = RHD_DataReader_With_File(mode="training", path="data_v2.0")
        print("loading training dataset time", time.time() - last)

        train_loader = torch.utils.data.DataLoader(
            eval_dataset,
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

            loss_avg = self.train(
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
                train_indicator = self.validate(train_loader, model, criterion, args=args)
                val_indicator = self.validate(val_loader, model, criterion, args=args)
                print(f'Save skeleton_model.pkl after {epoch + 1} epochs')
                error_writer.add_scalar('Validation Indicator', val_indicator, epoch)
                error_writer.add_scalar('Training Indicator', train_indicator, epoch)
                torch.save(model, f'trained_model_v1.5/skeleton_model_after_{epoch + 1}_epochs.pkl')
            if val_indicator > best_acc:
                best_acc = val_indicator

            # Draw the loss curve and validation indicator curve
            loss_writer.add_scalar('Loss', loss_avg, epoch)

            scheduler.step()

        '''Save Model'''

        print("Save skeleton_model.pkl after total training")
        torch.save(model, 'trained_model_v1.5/skeleton_model.pkl')
        cprint('All Done', 'yellow', attrs=['bold'])

        return 0  # end of main

    def one_forward_pass(self, sample, model, criterion, args, is_training=True):
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

    def train(self, train_loader, model, criterion, optimizer, args, epoch):
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
            results, targets, loss = self.one_forward_pass(
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
            # loss[1].backward()
            if epoch < 60:
                loss[5].backward()
            else:
                loss[0].backward()
            optimizer.step()

            ''' progress '''
            batch_time.update(time.time() - last)
            last = time.time()
            bar.suffix = (
                '({batch}/{size}) '
                'l: {loss:.5f} | '
                'lV: {lossV:.5f} | '
                'lD: {lossD:.5f} | '
                'lM: {lossM:.5f} | '
                'lK: {lossK:.5f} | '
            ).format(
                batch=i + 1,
                size=len(train_loader),
                loss=am_loss.avg,
                lossV=vec_loss.avg,
                lossD=dis_loss.avg,
                lossM=ske_loss.avg,
                lossK=kps_loss.avg
            )
            bar.next()
        bar.finish()
        return am_loss.avg

    def validate(self, val_loader, model, criterion, args, stop=-1):
        # switch to evaluate mode
        am_accH = AverageMeter()

        model.eval()
        bar = Bar('\033[33m Eval  \033[0m', max=len(val_loader))
        with torch.no_grad():
            for i, metas in enumerate(val_loader):
                results, targets, loss = self.one_forward_pass(
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


class BaselineModel:
    class HeatmapLoss:
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
            hm_veil = infos['hm_veil']
            batch_size = infos['batch_size']

            # compute hm_loss anyway
            hm_loss = torch.Tensor([0]).cuda()
            hm_veil = hm_veil.unsqueeze(-1)
            for pred_hm in preds[0]:
                njoints = pred_hm.size(1)
                batch_size = pred_hm.shape[0]
                pred_hm = pred_hm.reshape((batch_size, njoints, -1)).split(1, 1)
                targ_hm = targs['hm'].reshape((batch_size, njoints, -1)).split(1, 1)
                for idx in range(njoints):
                    pred_hmi = pred_hm[idx].squeeze()  # (B, 1, 4096)->(B, 4096)
                    targ_hmi = targ_hm[idx].squeeze()
                    hm_loss += 0.5 * F.mse_loss(
                        pred_hmi.mul(hm_veil[:, idx]),  # (B, 4096) mul (B, 1)
                        targ_hmi.mul(hm_veil[:, idx])
                    )
            return hm_loss

    def main(self, args):
        best_acc = 0

        print("\nCREATE NETWORK")
        model = NetStackedHourglass()
        model = model.to(device)

        criterion = self.HeatmapLoss()

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
            loss_avg = self.train(
                train_loader,
                model,
                criterion,
                optimizer,
                args=args,
            )
            ##################################################
            val_indicator = best_acc
            if epoch % 5 == 4:
                train_indicator = self.validate(train_loader, model, criterion, args=args)
                val_indicator = self.validate(val_loader, model, criterion, args=args)
                print(f'Save skeleton_model.pkl after {epoch + 1} epochs')
                error_writer.add_scalar('Validation Indicator', val_indicator, epoch)
                error_writer.add_scalar('Training Indicator', train_indicator, epoch)
                torch.save(model, f'trained_model_baseline/skeleton_model_after_{epoch + 1}_epochs.pkl')
            if val_indicator > best_acc:
                best_acc = val_indicator
            scheduler.step()

            # Draw the loss curve and validation indicator curve
            loss_writer.add_scalar('Loss', loss_avg.avg, epoch)
        print("\nSave model as hourglass_model.pkl...")
        torch.save(model, 'trained_model_baseline/hourglass_model.pkl')
        cprint('All Done', 'yellow', attrs=['bold'])
        return 0  # end of main

    def one_forward_pass(self, sample, model, criterion, args, is_training=True):
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

    def train(self, train_loader, model, criterion, optimizer, args):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        am_loss_hm = AverageMeter()

        last = time.time()
        # switch to train mode
        model.train()
        bar = Bar('\033[31m Train \033[0m', max=len(train_loader))
        for i, sample in enumerate(train_loader):
            data_time.update(time.time() - last)
            results, targets, loss = self.one_forward_pass(
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

    def validate(self, val_loader, model, criterion, args, stop=-1):
        # switch to evaluate mode
        am_accH = AverageMeter()

        model.eval()
        bar = Bar('\033[33m Eval  \033[0m', max=len(val_loader))
        with torch.no_grad():
            for i, metas in enumerate(val_loader):
                results, targets, loss = self.one_forward_pass(
                    metas, model, criterion, args, is_training=False
                )
                val = eval_utils.accuracy_heatmap(
                    results[0][-1],
                    targets['hm'],
                    targets['vis'][:, :, None]
                )
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


class HeatmapSoftmaxModel:
    class HeatmapLoss:
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

        def compute_loss(self, preds, targs, beta, infos):
            hm_veil = infos['hm_veil']

            # compute hm_loss anyway
            hm_loss = torch.Tensor([0]).cuda()
            hm_veil = hm_veil.unsqueeze(-1)
            for pred_hm in preds[0]:
                # # Debug
                # print("beta", beta)
                targ_kps = targs['kp2d']
                pred_kps = eval_utils.regress25d(pred_hm, beta)
                hm_loss += eval_utils.MeanEPE(pred_kps * hm_veil, targ_kps * hm_veil)
            return hm_loss

    def main(self, args):
        best_acc = 0

        print("\nCREATE NETWORK")
        model = NetStackedHourglass()
        model = model.to(device)

        criterion = self.HeatmapLoss()

        beta = torch.ones(21, device=device, requires_grad=True)

        optimizer = torch.optim.Adam(
            [
                {
                    'params': model.parameters(),
                    'initial_lr': args.learning_rate
                },
                {
                    'params': beta,
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
            loss_avg = self.train(
                train_loader,
                model,
                beta,
                criterion,
                optimizer,
                args=args,
            )
            ##################################################
            val_indicator = best_acc
            if epoch % 5 == 4:
                train_indicator = self.validate(train_loader, model, criterion, beta=beta, args=args)
                val_indicator = self.validate(val_loader, model, criterion, beta=beta, args=args)
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

    def one_forward_pass(self, sample, model, criterion, beta, args, is_training=True):
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
        loss = criterion.compute_loss(results, targets, beta, infos)

        return results, {**targets, **infos}, loss

    def train(self, train_loader, model, beta, criterion, optimizer, args):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        am_loss_hm = AverageMeter()

        last = time.time()
        # switch to train mode
        model.train()
        bar = Bar('\033[31m Train \033[0m', max=len(train_loader))
        for i, sample in enumerate(train_loader):
            data_time.update(time.time() - last)
            results, targets, loss = self.one_forward_pass(
                sample, model, criterion, beta, args, is_training=True
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

    def validate(self, val_loader, model, criterion, args, beta, stop=-1):
        # switch to evaluate mode
        am_accH = AverageMeter()

        model.eval()
        bar = Bar('\033[33m Eval  \033[0m', max=len(val_loader))
        with torch.no_grad():
            for i, metas in enumerate(val_loader):
                results, targets, loss = self.one_forward_pass(
                    metas, model, criterion, beta, args, is_training=False
                )
                pred_hm = results[0][-1]
                targ_kps = targets['kp2d']
                pred_kps = eval_utils.regress25d(pred_hm, beta)
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

    # Model Choice
    parser.add_argument(
        '--baseline_model',
        default=False,
        action='store_true',
        help='true if using the baseline model'
    )

    parser.add_argument(
        '--heatmap_softmax_model',
        default=False,
        action='store_true',
        help='true if using the heatmap with soft argmax model'
    )

    parser.add_argument(
        '--skeleton_model',
        default=True,
        action='store_true',
        help='true if using the sekeleton based model'
    )

    # Dataset setting
    parser.add_argument(
        '-dr',
        '--data_root',
        type=str,
        default='RHD_published_v2',
        help='dataset root directory'
    )

    parser.add_argument(
        '--process_training_data',
        default=False,
        action='store_true',
        help='true if the data has been processed'
    )

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

    # Training Parameters
    parser.add_argument(
        '-j', '--workers',
        default=8,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 8)'
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
        '--gamma',
        type=float,
        default=0.1,
        help='LR is multiplied by gamma on schedule.'
    )

    model = []
    args = parser.parse_args()
    if args.baseline_model:
        parser.add_argument(
            '-lr', '--learning-rate',
            default=1.0e-4,
            type=float,
            metavar='LR',
            help='initial learning rate'
        )

        parser.add_argument(
            '--epochs',
            default=100,
            type=int,
            metavar='N',
            help='number of total epochs to run'
        )

        parser.add_argument(
            "--lr_decay_step",
            default=50,
            type=int,
            help="Epochs after which to decay learning rate",
        )

        model = BaselineModel()
    elif args.heatmap_softmax_model:
        parser.add_argument(
            '-lr', '--learning-rate',
            default=1.0e-3,
            type=float,
            metavar='LR',
            help='initial learning rate'
        )

        parser.add_argument(
            '--epochs',
            default=60,
            type=int,
            metavar='N',
            help='number of total epochs to run'
        )

        parser.add_argument(
            "--lr_decay_step",
            default=50,
            type=int,
            help="Epochs after which to decay learning rate",
        )

        model = HeatmapSoftmaxModel()
    elif args.skeleton_model:
        parser.add_argument(
            '-lr', '--learning-rate',
            default=1.0e-4,
            type=float,
            metavar='LR',
            help='initial learning rate'
        )

        parser.add_argument(
            '--epochs',
            default=120,
            type=int,
            metavar='N',
            help='number of total epochs to run'
        )

        parser.add_argument(
            "--lr_decay_step",
            default=40,
            type=int,
            help="Epochs after which to decay learning rate",
        )

        parser.add_argument(
            '-cc',
            '--checking_cycle',
            type=int,
            default=5,
            help='How many batches to save the model at once'
        )

        model = SkeletonBaseModel()
    model.main(parser.parse_args())
