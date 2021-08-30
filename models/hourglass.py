import torch.nn as nn
import torch
import torch.nn.functional as F

from models.bottleneck import BottleneckBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Hourglass(nn.Module):
    def __init__(self, block, nblocks, in_planes, depth=4):
        super(Hourglass, self).__init__()
        self.depth = depth
        # Structure of hg: [[r0, r1, r2, r3], [r0, r1, r2], [r0, r1, r2], ...] where its length is depth
        self.hg = self._make_hourglass(block, nblocks, in_planes, depth)

    def _make_hourglass(self, block, nblocks, in_planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, nblocks, in_planes))
            if i == 0:
                res.append(self._make_residual(block, nblocks, in_planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _make_residual(self, block, nblocks, in_planes):
        layers = []
        for i in range(0, nblocks):
            layers.append(block(in_planes, in_planes))
        return nn.Sequential(*layers)

    def _hourglass_foward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hourglass_foward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hourglass_foward(self.depth, x)


# Need to modify, focus on batching part
# class Mapping(nn.Module):
#     def __init__(self):
#         super(Mapping, self).__init__()
#
#     def get_direction_from_flux(self, flux, mask):
#         direction = torch.clone(flux).detach()
#         direction[:, :, :, 0] = flux[:, :, :, 1] * mask[:, :, :, 0]
#         direction[:, :, :, 1] = -flux[:, :, :, 0] * mask[:, :, :, 0]
#         return direction
#
#     # Updating: Make this works for batching
#     # 07/24/2021, 14:55 Maybe work now
#     def maps_to_kp(self, flux_map, dis_map, res=64):
#         batch_size = flux_map.shape[0]
#         locs = torch.zeros([batch_size, res, res, 2]).to(device, non_blocking=True)
#         for y in range(res):
#             locs[:, :, y, 0] = torch.FloatTensor(range(res))
#             locs[:, :, y, 1] = y
#         kp = torch.zeros([batch_size, 21, 2]).to(device, non_blocking=True)
#
#         # predicts the wrist keypoint 0
#         for i in range(5):
#             flux = flux_map[:, 4 * i, :, :, :2]
#             masks = flux_map[:, 4 * i, :, :, 2, None]
#             direction = self.get_direction_from_flux(flux, masks)
#             j = 0
#             for mask in masks:
#                 if torch.sum(torch.abs(mask)) != 0:
#                     proposal = torch.abs(mask) * (direction[j] * dis_map[j, 4 * i, :, :, 0, None] + locs[j])
#                     kp[j][0] += torch.sum(torch.sum(proposal, dim=0), dim=0) / torch.sum(torch.abs(mask))
#                 j = j + 1
#         kp[:, 0] = kp[:, 0] / 5
#
#         for i in range(5):
#             for j in range(4):
#                 as_second_point_flux = flux_map[:, 4 * i + j, :, :, :2]
#                 as_second_point_masks = flux_map[:, 4 * i + j, :, :, 2, None]  # (64, 64, 1)
#                 as_second_point_direction = self.get_direction_from_flux(as_second_point_flux, as_second_point_masks)
#                 as_first_point_flux = flux_map[:, 4 * i + j + 1, :, :, :2] if j != 3 else 0
#                 as_first_point_masks = flux_map[:, 4 * i + j + 1, :, :, 2, None] if j != 3 else 0
#                 as_first_point_direction = self.get_direction_from_flux(as_first_point_flux, as_first_point_masks) \
#                     if j != 3 else 0
#                 k = 0
#                 for as_second_point_mask in as_second_point_masks:
#                     as_second_point_votes = torch.abs(as_second_point_mask) * (
#                             -as_second_point_direction[k] * dis_map[k, 4 * i + j, :, :, 1, None] + locs[k])
#                     as_second_point_pred = as_second_point_votes / torch.sum(torch.abs(as_second_point_mask))
#                     if j != 3:
#                         # predict keypoint (4, 3, 2), (8, 7, 6), (12, 11, 10), (16, 15, 14), (20, 19, 18)
#                         as_first_point_votes = torch.abs(as_first_point_masks[k]) * (
#                                 as_first_point_direction[k] * dis_map[k, 4 * i + j + 1, :, :, 0, None] + locs[k])
#                         as_first_point_pred = as_first_point_votes / torch.sum(torch.abs(as_first_point_masks[k]))
#                         kp[k, 4 * i + 4 - j] = torch.sum(torch.sum(as_second_point_pred + as_first_point_pred, dim=0),
#                                                          dim=0) / 2
#                     else:
#                         # predicts the location of fingertips 1, 5, 9, 13, 17
#                         kp[k, 4 * i + 4 - j] = torch.sum(torch.sum(as_second_point_pred, dim=0), dim=0)
#                     k = k + 1
#
#         return kp * 4
#
#     def forward(self, x):
#         flux_dimension = 3
#         dis_dimension = 2
#         count_phalanges = 20
#         total_dimension = flux_dimension + dis_dimension  # 5
#         # Need to modify this into any resolution
#         x = x.reshape((x.shape[0], count_phalanges, 64, 64, total_dimension))  # (B, 20, res*res, 5)
#         # split it into (B, 20, res*res, 3) and (B, 20, res*res, 2)
#         pred_flux, pred_dis = torch.split(x, split_size_or_sections=[3, 2], dim=-1)
#         kps = self.maps_to_kp(pred_flux, pred_dis)
#         return kps


# Need to modify, focus on batching part
class Mapping_2(nn.Module):
    def __init__(self):
        super(Mapping_2, self).__init__()

    def maps_to_kp_2(self, front_vecs, front_diss, back_vecs, back_diss, maskss, res=64):
        batch_size = front_vecs.shape[0]
        locs = torch.zeros([batch_size, res, res, 2]).to(device, non_blocking=True)
        for y in range(res):
            locs[:, :, y, 0] = torch.FloatTensor(range(res))
            locs[:, :, y, 1] = y
        kp = torch.zeros([batch_size, 21, 2]).to(device, non_blocking=True)

        front_vecs = front_vecs.reshape(batch_size, 20, res, res, 2)
        front_diss = front_diss.reshape(batch_size, 20, res, res, 1)
        back_vecs = back_vecs.reshape(batch_size, 20, res, res, 2)
        back_diss = back_diss.reshape(batch_size, 20, res, res, 1)
        maskss = maskss.reshape(batch_size, 20, res, res, 1)

        # predicts the wrist keypoint 0
        for i in range(5):
            back_vec = back_vecs[:, 4 * i, :, :, :]
            back_dis = back_diss[:, 4 * i, :, :, :]
            masks = maskss[:, 4 * i, :, :, :]
            j = 0
            for mask in masks:
                if torch.sum(mask) != 0:
                    proposal = mask * (back_vec[j] * back_dis[j, :, :, 0, None] * res + locs[j])
                    # for x in range(64):
                    #     for y in range(64):
                    #         if mask[x, y, 0]:
                    #             print(back_vec[j, x, y] * back_dis[j, x, y, 0, None] + locs[j][x][y])
                    # print("proposal:", proposal.shape)
                    kp[j][0] += torch.sum(torch.sum(proposal, dim=0), dim=0) / (torch.sum(mask) + 1e-6)
                j = j + 1
        kp[:, 0] = kp[:, 0] / 5

        for i in range(5):
            for j in range(4):
                k = 0
                for masks in maskss:
                    front_votes = masks[4 * i + j] * (
                            front_vecs[k][4 * i + j] * front_diss[k, 4 * i + j, :, :, 0, None] * res + locs[k])
                    front_pred = front_votes / (torch.sum(masks[4 * i + j]) + 1e-6)
                    if j != 3:
                        # predict keypoint (4, 3, 2), (8, 7, 6), (12, 11, 10), (16, 15, 14), (20, 19, 18)
                        back_votes = masks[4 * i + j + 1] * (
                                back_vecs[k][4 * i + j + 1] * back_diss[k, 4 * i + j + 1, :, :, 0, None] * res + locs[
                            k])
                        back_pred = back_votes / (torch.sum(masks[4 * i + j + 1]) + 1e-6)
                        kp[k, 4 * i + 4 - j] = torch.sum(torch.sum(front_pred + back_pred, dim=0), dim=0) / 2
                    else:
                        # predicts the location of fingertips 1, 5, 9, 13, 17
                        kp[k, 4 * i + 4 - j] = torch.sum(torch.sum(front_pred, dim=0), dim=0)
                    k = k + 1

        return kp * 4

    def forward(self, front_vec, front_dis, back_vec, back_dis, ske_mask):
        kps = self.maps_to_kp_2(front_vec, front_dis, back_vec, back_dis, ske_mask)
        return kps


# classical
class NetStackedHourglass_2(nn.Module):
    def __init__(
            self,
            nstacks=2,
            nblocks=1,
            nclasses=140,
            block=BottleneckBlock
    ):
        super(NetStackedHourglass_2, self).__init__()
        self.nclasses = nclasses
        self.nstacks = nstacks
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.layer1 = self._make_residual(block, nblocks, self.in_planes, 2 * self.in_planes)
        # current self.in_planes is 64 * 2 = 128
        self.layer2 = self._make_residual(block, nblocks, self.in_planes, 2 * self.in_planes)
        # current self.in_planes is 128 * 2 = 256
        self.layer3 = self._make_residual(block, nblocks, self.in_planes, self.in_planes)

        ch = self.in_planes  # 256
        self.nfeats = ch

        hgs, res, fc, front_vec, front_dis, back_vec, back_dis, ske_mask, kp, _fc, _front_vec, _front_dis, _back_vec, _back_dis, _ske_mask, bn2, bn3, bn4, bn5, bn6  \
            = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        for i in range(nstacks):  # stacking the hourglass
            hgs.append(Hourglass(block, nblocks, ch, depth=4))
            res.append(self._make_residual(block, nblocks, ch, ch))
            fc.append(self._make_residual(block, nblocks, ch, ch))
            front_vec.append(nn.Conv2d(ch, 40, kernel_size=1, bias=True))
            bn2.append(nn.BatchNorm2d(40))
            front_dis.append(nn.Conv2d(ch, 20, kernel_size=1, bias=True))
            bn3.append(nn.BatchNorm2d(20))
            back_vec.append(nn.Conv2d(ch, 40, kernel_size=1, bias=True))
            bn4.append(nn.BatchNorm2d(40))
            back_dis.append(nn.Conv2d(ch, 20, kernel_size=1, bias=True))
            bn5.append(nn.BatchNorm2d(20))
            ske_mask.append(nn.Conv2d(ch, 20, kernel_size=1, bias=True))
            bn6.append(nn.BatchNorm2d(20))
            kp.append(Mapping_2())

            if i < nstacks - 1:
                _fc.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                _front_vec.append(nn.Conv2d(40, ch,  kernel_size=1, bias=True))
                _front_dis.append(nn.Conv2d(20, ch, kernel_size=1, bias=True))
                _back_vec.append(nn.Conv2d(40, ch, kernel_size=1, bias=True))
                _back_dis.append(nn.Conv2d(20, ch,  kernel_size=1, bias=True))
                _ske_mask.append(nn.Conv2d(20, ch,  kernel_size=1, bias=True))

        self.hgs = nn.ModuleList(hgs)  # hgs: hourglass stack
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)  # change back to use the pre-trained
        self.front_vec = nn.ModuleList(front_vec)
        self.front_dis = nn.ModuleList(front_dis)
        self.back_vec = nn.ModuleList(back_vec)
        self.back_dis = nn.ModuleList(back_dis)
        self.ske_mask = nn.ModuleList(ske_mask)
        self.kp = nn.ModuleList(kp)
        self._fc = nn.ModuleList(_fc)
        self._front_vec = nn.ModuleList(_front_vec)
        self._front_dis = nn.ModuleList(_front_dis)
        self._back_vec = nn.ModuleList(_back_vec)
        self._back_dis = nn.ModuleList(_back_dis)
        self._ske_mask = nn.ModuleList(_ske_mask)
        self.bn2 = nn.ModuleList(bn2)
        self.bn3 = nn.ModuleList(bn3)
        self.bn4 = nn.ModuleList(bn4)
        self.bn5 = nn.ModuleList(bn5)
        self.bn6 = nn.ModuleList(bn6)

    def _make_residual(self, block, nblocks, in_planes, out_planes):
        layers = [block(in_planes, out_planes)]
        self.in_planes = out_planes
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, out_planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        front_vecs = []
        front_diss = []
        back_vecs = []
        back_diss = []
        ske_masks = []
        kps = []
        # x: (B,3,256,256)
        x = self.conv1(x)  # x: (B,64,128,128)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # x: (B,128,128,128)
        x = self.maxpool(x)  # x: (B,128,64,64)
        x = self.layer2(x)  # x: (B,256,64,64)
        x = self.layer3(x)  # x: (B,256,64,64)
        for i in range(self.nstacks):
            y = self.hgs[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            front_vec = self.front_vec[i](y)
            # front_vec = torch.sigmoid(front_vec)
            # front_vec = self.bn2[i](front_vec)

            front_dis = self.front_dis[i](y)
            front_dis = torch.sigmoid(front_dis)
            # front_dis = self.bn3[i](front_dis)

            back_vec = self.back_vec[i](y)
            # back_vec = torch.sigmoid(back_vec)
            # back_vec = self.bn4[i](back_vec)

            back_dis = self.back_dis[i](y)
            back_dis = torch.sigmoid(back_dis)
            # back_dis = self.bn5[i](back_dis)

            ske_mask = self.ske_mask[i](y)
            ske_mask = torch.sigmoid(ske_mask)
            # ske_mask = self.bn6[i](ske_mask)

            kp = self.kp[i](front_vec, front_dis, back_vec, back_dis, ske_mask)
            front_vecs.append(front_vec)  # fluxs: (nstacks, B, 40, 64, 64), maps[-1] is the final prediction
            front_diss.append(front_dis)  # masks: (nstacks, B, 20, 64, 64), maps[-1] is the final prediction
            back_vecs.append(back_vec)  # fluxs: (nstacks, B, 40, 64, 64), maps[-1] is the final prediction
            back_diss.append(back_dis)  # masks: (nstacks, B, 20, 64, 64), maps[-1] is the final prediction
            ske_masks.append(ske_mask)  # masks: (nstacks, B, 20, 64, 64), maps[-1] is the final prediction
            kps.append(kp)  # kps: (nstacks, B, 21, 2), kps[-1] is the final prediction

            if i < self.nstacks - 1:
                _fc = self._fc[i](y)
                _front_vec = self._front_vec[i](front_vec)
                _front_dis = self._front_dis[i](front_dis)
                _back_vec = self._back_vec[i](back_vec)
                _back_dis = self._back_dis[i](back_dis)
                _ske_mask = self._ske_mask[i](ske_mask)
                x = x + _fc \
                    # + _front_vec + _front_dis + _back_vec + _back_dis + _ske_mask

        return front_vecs, front_diss, back_vecs, back_diss, ske_masks, kps


# classical
# class NetStackedHourglass(nn.Module):
#     def __init__(
#             self,
#             nstacks=2,
#             nblocks=1,
#             nclasses=100,
#             block=BottleneckBlock
#     ):
#         super(NetStackedHourglass, self).__init__()
#         self.nclasses = nclasses
#         self.nstacks = nstacks
#         self.in_planes = 64
#
#         self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=True)
#         self.bn1 = nn.BatchNorm2d(self.in_planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(2, stride=2)
#         self.layer1 = self._make_residual(block, nblocks, self.in_planes, 2 * self.in_planes)
#         # current self.in_planes is 64 * 2 = 128
#         self.layer2 = self._make_residual(block, nblocks, self.in_planes, 2 * self.in_planes)
#         # current self.in_planes is 128 * 2 = 256
#         self.layer3 = self._make_residual(block, nblocks, self.in_planes, self.in_planes)
#
#         ch = self.in_planes  # 256
#         self.nfeats = ch
#
#         hgs, res, fc, flux, mask, dis, kp, _score = [], [], [], [], [], [], [], []
#         for i in range(nstacks):  # stacking the hourglass
#             hgs.append(Hourglass(block, nblocks, ch, depth=4))
#             res.append(self._make_residual(block, nblocks, ch, ch))
#             fc.append(self._make_residual(block, nblocks, ch, ch))
#             flux.append(nn.Conv2d(ch, 40, kernel_size=1, bias=True))
#             mask.append(nn.Conv2d(ch, 20, kernel_size=1, bias=True))
#             dis.append(nn.Conv2d(ch, 40, kernel_size=1, bias=True))
#             kp.append(Mapping())
#
#             if i < nstacks - 1:
#                 _score.append(nn.Conv2d(21, ch, kernel_size=1, bias=True))
#
#         self.hgs = nn.ModuleList(hgs)  # hgs: hourglass stack
#         self.res = nn.ModuleList(res)
#         self.fc = nn.ModuleList(fc)  # change back to use the pre-trained
#         self.flux = nn.ModuleList(flux)
#         self.mask = nn.ModuleList(mask)
#         self.dis = nn.ModuleList(dis)
#         self.kp = nn.ModuleList(kp)
#
#     def _make_residual(self, block, nblocks, in_planes, out_planes):
#         layers = [block(in_planes, out_planes)]
#         self.in_planes = out_planes
#         for i in range(1, nblocks):
#             layers.append(block(self.in_planes, out_planes))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         fluxs = []
#         masks = []
#         diss = []
#         kps = []
#         # x: (B,3,256,256)
#         x = self.conv1(x)  # x: (B,64,128,128)
#         x = self.bn1(x)
#         x = self.relu(x)
#
#         x = self.layer1(x)  # x: (B,128,128,128)
#         x = self.maxpool(x)  # x: (B,128,64,64)
#         x = self.layer2(x)  # x: (B,256,64,64)
#         x = self.layer3(x)  # x: (B,256,64,64)
#         for i in range(self.nstacks):
#             y = self.hgs[i](x)
#             y = self.res[i](y)
#             y = self.fc[i](y)
#             flux = self.flux[i](y)
#             mask = self.mask[i](y)
#             dis = self.dis[i](y)
#             maps = torch.cat((flux, mask, dis), dim=1)
#             kp = self.kp[i](maps)
#             fluxs.append(flux)  # fluxs: (nstacks, B, 40, 64, 64), maps[-1] is the final prediction
#             masks.append(mask)  # masks: (nstacks, B, 20, 64, 64), maps[-1] is the final prediction
#             diss.append(dis)  # diss: (nstacks, B, 40, 64, 64), maps[-1] is the final prediction
#             kps.append(kp)  # kps: (nstacks, B, 21, 2), kps[-1] is the final prediction
#         return fluxs, masks, diss * 64, kps
