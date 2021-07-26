import torch.nn as nn
import torch
import torch.nn.functional as F

from models.bottleneck import BottleneckBlock


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
class Mapping(nn.Module):
    def __init__(self):
        super(Mapping, self).__init__()

    def get_direction_from_flux(self, flux, mask):
        direction = torch.clone(flux).detach()
        direction[:, :, :, 0] = flux[:, :, :, 1] * mask[:, :, :, 0]
        direction[:, :, :, 1] = -flux[:, :, :, 0] * mask[:, :, :, 0]
        return direction

    # Updating: Make this works for batching
    # 07/24/2021, 14:55 Maybe work now
    def maps_to_kp(self, flux_map, dis_map, res=64):
        batch_size = flux_map.shape[0]
        locs = torch.zeros([batch_size, res, res, 2])
        for y in range(res):
            locs[:, :, y, 0] = torch.FloatTensor(range(res))
            locs[:, :, y, 1] = y
        kp = torch.zeros([batch_size, 21, 2])

        # predicts the wrist keypoint 0
        kp[0] = 0
        for i in range(5):
            flux = flux_map[:, 4 * i, :, :, :2]
            masks = flux_map[:, 4 * i, :, :, 2, None]
            direction = self.get_direction_from_flux(flux, masks)
            j = 0
            for mask in masks:
                if torch.sum(torch.abs(mask)) != 0:
                    kp[j][0] += torch.sum(
                        torch.sum(torch.abs(mask)
                                  * (direction * dis_map[j, 4 * i, :, :, 0, None] + locs), dim=0),
                        dim=0) \
                        / torch.sum(torch.abs(mask))
                j = j + 1
        count = 5
        kp[0] = kp[0] / count

        for i in range(5):
            for j in range(4):
                as_second_point_flux = flux_map[:, 4 * i + j, :, :, :2]
                as_second_point_masks = flux_map[:, 4 * i + j, :, :, 2, None]  # (64, 64, 1)
                as_second_point_direction = self.get_direction_from_flux(as_second_point_flux, as_second_point_masks)
                as_first_point_flux = flux_map[:, 4 * i + j + 1, :, :, :2]
                as_first_point_masks = flux_map[:, 4 * i + j + 1, :, :, 2, None]
                as_first_point_direction = self.get_direction_from_flux(as_first_point_flux, as_first_point_masks)
                k = 0
                for as_second_point_mask in as_second_point_masks:
                    as_second_point_votes = torch.abs(as_second_point_mask) * (
                            -as_second_point_direction * dis_map[k, 4 * i + j, :, :, 1, None] + locs)
                    as_second_point_pred = as_second_point_votes / torch.sum(torch.abs(as_second_point_mask))
                    if j != 3:
                        # predict keypoint (4, 3, 2), (8, 7, 6), (12, 11, 10), (16, 15, 14), (20, 19, 18)
                        as_first_point_votes = torch.abs(as_first_point_masks[k]) * (
                                as_first_point_direction * dis_map[k, 4 * i + j + 1, :, :, 0, None] + locs)
                        as_first_point_pred = as_first_point_votes / torch.sum(torch.abs(as_first_point_masks[k]))
                        kp[k, 4 * i + 4 - j] = torch.sum(torch.sum(as_second_point_pred + as_first_point_pred, dim=0),
                                                         dim=0) / 2
                    else:
                        # predicts the location of fingertips 1, 5, 9, 13, 17
                        kp[k, 4 * i + 4 - j] = torch.sum(torch.sum(as_second_point_pred, dim=0), dim=0)
                    k = k + 1

        return kp

    def forward(self, x):
        flux_dimension = 3
        dis_dimension = 2
        count_phalanges = 20
        total_dimension = flux_dimension + dis_dimension  # 5

        x = x.reshape((x.shape[0], count_phalanges, -1, total_dimension))  # (B, 20, res*res, 5)
        # split it into (B, 20, res*res, 3) and (B, 20, res*res, 2)
        pred_maps = torch.split(x, split_size_or_sections=[3, 2], dim=-1)
        pred_flux = pred_maps[0]
        pred_dis = pred_maps[1]
        kps = self.maps_to_kp(pred_flux, pred_dis)
        return kps


# classical
class NetStackedHourglass(nn.Module):
    def __init__(
            self,
            nstacks=2,
            nblocks=1,
            nclasses=100,
            block=BottleneckBlock
    ):
        super(NetStackedHourglass, self).__init__()
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

        hgs, res, fc, _fc, score, kp, _score, _kp = [], [], [], [], [], [], [], []
        for i in range(nstacks):  # stacking the hourglass
            hgs.append(Hourglass(block, nblocks, ch, depth=4))
            res.append(self._make_residual(block, nblocks, ch, ch))
            fc.append(self._make_residual(block, nblocks, ch, ch))
            score.append(nn.Conv2d(ch, nclasses, kernel_size=1, bias=True))
            kp.append(Mapping())

            if i < nstacks - 1:
                _fc.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                _score.append(nn.Conv2d(nclasses, ch, kernel_size=1, bias=True))
                _kp.append(Mapping())

        self.hgs = nn.ModuleList(hgs)  # hgs: hourglass stack
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)  # change back to use the pre-trained
        self._fc = nn.ModuleList(_fc)
        self.score = nn.ModuleList(score)
        self._score = nn.ModuleList(_score)
        self.kp = nn.ModuleList(kp)
        self._kp = nn.ModuleList(_kp)

    def _make_residual(self, block, nblocks, in_planes, out_planes):
        layers = []
        layers.append(block(in_planes, out_planes))
        self.in_planes = out_planes
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, out_planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        maps = []
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
            score = self.score[i](y)
            kp = self.kp[i](y)
            maps.append(score)  # maps: (nstacks, B, 100, 64, 64), maps[-1] is the final prediction
            kps.append(kp)  # kps: (nstacks, B, 21, 2), kps[-1] is the final prediction
            if i < self.nstacks - 1:
                _fc = self._fc[i](y)
                _score = self._score[i](score)
                _kp = self._kp[i](kp)
        return maps, kps
