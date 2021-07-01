import torch.nn as nn
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

        hgs, res, fc, _fc, score, _score = [], [], [], [], [], []
        for i in range(nstacks):  # stacking the hourglass
            hgs.append(Hourglass(block, nblocks, ch, depth=4))
            res.append(self._make_residual(block, nblocks, ch, ch))
            fc.append(self._make_residual(block, nblocks, ch, ch))
            score.append(nn.Conv2d(ch, nclasses, kernel_size=1, bias=True))

            if i < nstacks - 1:
                _fc.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                _score.append(nn.Conv2d(nclasses, ch, kernel_size=1, bias=True))

        self.hgs = nn.ModuleList(hgs)  # hgs: hourglass stack
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)  # change back to use the pre-trained
        self._fc = nn.ModuleList(_fc)
        self.score = nn.ModuleList(score)
        self._score = nn.ModuleList(_score)

    def _make_residual(self, block, nblocks, in_planes, out_planes):
        layers = []
        layers.append(block(in_planes, out_planes))
        self.in_planes = out_planes
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, out_planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = []
        hm_enc = []  # heatmaps encoding
        # x: (B,3,256,256)
        x = self.conv1(x)  # x: (B,64,128,128)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # x: (B,128,128,128)
        x = self.maxpool(x)  # x: (B,128,64,64)
        x = self.layer2(x)  # x: (B,256,64,64)
        x = self.layer3(x)  # x: (B,256,64,64)
        hm_enc.append(x)
        for i in range(self.nstacks):
            y = self.hgs[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)  # out: (nstacks, B, 100, 64, 64), out[-1] is the final prediction
            if i < self.nstacks - 1:
                _fc = self._fc[i](y)
                _score = self._score[i](score)
                x = x + _fc + _score
                hm_enc.append(x)
            else:
                hm_enc.append(y)
        return out, hm_enc
