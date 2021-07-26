import os.path as osp
import torch
import torch.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calc_dists(preds, target, normalize, mask):
    preds = preds.float()
    target = target.float()
    dists = torch.zeros(preds.size(1), preds.size(0))  # (njoint, B)
    for b in range(preds.size(0)):
        for j in range(preds.size(1)):
            if mask[b][j] == 0:
                dists[j, b] = -1
            elif target[b, j, 0] < 1 or target[b, j, 1] < 1:
                dists[j, b] = -1
            else:
                dists[j, b] = torch.dist(preds[b, j, :], target[b, j, :]) / normalize[b]

    return dists


def MeanEPE(x, y):
    return torch.mean(F.pairwise_distance(x.permute(0, 2, 1), y.permute(0, 2, 1), p=2))


def dist_acc(dist, thr=0.5):
    """ Return percentage below threshold while ignoring values with a -1 """
    dist = dist[dist != -1]
    if len(dist) > 0:
        return 1.0 * (dist < thr).sum().item() / len(dist)
    else:
        return -1


def gen_heatmap(img, pt, sigma):
    """generate heatmap based on pt coord.

    :param img: original heatmap, zeros
    :type img: np (H,W) float32
    :param pt: keypoint coord.
    :type pt: np (2,) int32
    :param sigma: guassian sigma
    :type sigma: float
    :return
    - generated heatmap, np (H, W) each pixel values id a probability
    - flag 0 or 1: indicate wheather this heatmap is valid(1)

    """

    pt = pt.astype(np.int32)
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (
            ul[0] >= img.shape[1]
            or ul[1] >= img.shape[0]
            or br[0] < 0
            or br[1] < 0
    ):
        # If not, just return the image as is
        return img, 0

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img, 1


def get_heatmap_pred(heatmaps):
    """ get predictions from heatmaps in torch Tensor
        return type: torch.LongTensor
    """
    assert heatmaps.dim() == 4, 'Score maps should be 4-dim (B, nJoints, H, W)'
    maxval, idx = torch.max(heatmaps.view(heatmaps.size(0), heatmaps.size(1), -1), 2)

    maxval = maxval.view(heatmaps.size(0), heatmaps.size(1), 1)
    idx = idx.view(heatmaps.size(0), heatmaps.size(1), 1)

    preds = idx.repeat(1, 1, 2).float()  # (B, njoint, 2)

    preds[:, :, 0] = (preds[:, :, 0]) % heatmaps.size(3)  # + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / heatmaps.size(3))  # + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def accuracy_heatmap(output, target, mask, thr=0.5):
    """ Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First to be returned is average accuracy across 'idxs', Second is individual accuracies
    """
    preds = get_heatmap_pred(output).float()  # (B, njoint, 2)
    gts = get_heatmap_pred(target).float()
    norm = torch.ones(preds.size(0)) * output.size(3) / 10.0  # (B, ), all 6.4:(1/10 of heatmap side)
    dists = calc_dists(preds, gts, norm, mask)  # (njoint, B)

    acc = torch.zeros(mask.size(1))
    avg_acc = 0
    cnt = 0

    for i in range(mask.size(1)):  # njoint
        acc[i] = dist_acc(dists[i], thr)
        if acc[i] >= 0:
            avg_acc += acc[i]
            cnt += 1

    if cnt != 0:
        avg_acc /= cnt

    return avg_acc, acc


def get_measures(self, val_min, val_max, steps):
    """ Outputs the average mean and median error as well as the pck score. """
    thresholds = np.linspace(val_min, val_max, steps)
    thresholds = np.array(thresholds)
    norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

    # init mean measures
    epe_mean_all = list()
    epe_median_all = list()
    auc_all = list()
    pck_curve_all = list()

    # Create one plot for each part
    for part_id in range(self.num_kp):
        # mean/median error
        mean, median = self._get_epe(part_id)

        if mean is None:
            # there was no valid measurement for this keypoint
            continue

        epe_mean_all.append(mean)
        epe_median_all.append(median)

        # pck/auc
        pck_curve = list()
        for t in thresholds:
            pck = self._get_pck(part_id, t)
            pck_curve.append(pck)

        pck_curve = np.array(pck_curve)
        pck_curve_all.append(pck_curve)
        auc = np.trapz(pck_curve, thresholds)
        auc /= norm_factor
        auc_all.append(auc)
    # Display error per keypoint
    epe_mean_joint = epe_mean_all
    epe_mean_all = np.mean(np.array(epe_mean_all))
    epe_median_all = np.mean(np.array(epe_median_all))
    auc_all = np.mean(np.array(auc_all))
    pck_curve_all = np.mean(np.array(pck_curve_all), axis=0)  # mean only over keypoints

    return (
        epe_mean_all,
        epe_mean_joint,
        epe_median_all,
        auc_all,
        pck_curve_all,
        thresholds,
    )


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v, norm
    return v / norm, norm


def is_on_skeleton(res, direction, length, first_point, sigma=1):
    vertical_direction = np.array([direction[1], -direction[0]])
    pred = np.ndarray([res, res])
    for x in range(res):
        for y in range(res):
            pred_1 = 0 <= np.dot(direction, np.array([x, y]) - first_point) and np.dot(direction, np.array([x, y]) - first_point) <= length
            pred_2 = np.abs(np.dot(vertical_direction, np.array([x, y]) - first_point)) <= sigma
            pred[x, y] = pred_1 and pred_2
    return pred


def get_direction_from_flux(flux, mask):
    direction = np.copy(flux)
    direction[:, :, 0] = flux[:, :, 1] * mask[:, :, 0]
    direction[:, :, 1] = -flux[:, :, 0] * mask[:, :, 0]
    return direction


def maps_to_kp(flux_map, dis_map, res=64):
    locs = np.zeros([res, res, 2])
    for y in range(res):
        locs[:, y, 0] = np.array(range(res))
        locs[:, y, 1] = y
    kp = np.zeros([21, 2])

    # predicts the wrist keypoint 0
    kp[0] = 0
    for i in range(5):
        flux = flux_map[4 * i, :, :, :2]
        mask = flux_map[4 * i, :, :, 2, None]
        direction = get_direction_from_flux(flux, mask)
        count = 5
        if np.sum(np.abs(mask)) != 0:
            kp[0] += np.sum(np.sum(np.abs(mask) * (
                        direction * dis_map[4 * i, :, :, 0, None] + locs), axis=0), axis=0) / np.sum(np.abs(mask))
        # else:
        #     print("flag")
    kp[0] = kp[0] / count
    # print("kp[0] = ", kp[0])

    for i in range(5):
        for j in range(4):
            as_second_point_flux = flux_map[4 * i + j, :, :, :2]
            as_second_point_mask = flux_map[4 * i + j, :, :, 2, None]  # (64, 64, 1)
            as_second_point_direction = get_direction_from_flux(as_second_point_flux, as_second_point_mask)
            as_second_point_votes = np.abs(as_second_point_mask) * (
                    -as_second_point_direction * dis_map[4 * i + j, :, :, 1, None] + locs)
            as_second_point_pred = as_second_point_votes / np.sum(np.abs(as_second_point_mask))
            if j != 3:
                # predict the location of keypoint (4, 3, 2), (8, 7, 6), (12, 11, 10), (16, 15, 14), (20, 19, 18)
                as_first_point_flux = flux_map[4 * i + j + 1, :, :, :2]
                as_first_point_mask = flux_map[4 * i + j + 1, :, :, 2, None]
                as_first_point_direction = get_direction_from_flux(as_first_point_flux, as_first_point_mask)
                as_first_point_votes = np.abs(as_first_point_mask) * (
                        as_first_point_direction * dis_map[4 * i + j + 1, :, :, 0, None] + locs)
                as_first_point_pred = as_first_point_votes / np.sum(np.abs(as_first_point_mask))
                kp[4 * i + 4 - j] = np.sum(np.sum(as_second_point_pred + as_first_point_pred, axis=0), axis=0) / 2
            else:
                # predicts the location of fingertips 1, 5, 9, 13, 17
                kp[4 * i + 4 - j] = np.sum(np.sum(as_second_point_pred, axis=0), axis=0)
            # print("kp[", 4 * i + 4 - j, "] = ", kp[4 * i + 4 - j])

    return kp


def fill_maps(flux_map, dis_map, binary_skeleton, phalanx_idx, first_point, second_point,
              res=64, sigma=1):
    phalanx = second_point - first_point
    
    direction, length = normalize(phalanx)
    counter_clockwise_flux = np.array([-direction[1], direction[0]])
    clockwise_flux = np.array([direction[1], -direction[0]])
    binary_skeleton[phalanx_idx, :, :, 0] = is_on_skeleton(res, direction, length, first_point, sigma)
    for x in range(res):
        for y in range(res):
            if np.cross(first_point - np.array([x, y]), direction) < 0 and binary_skeleton[phalanx_idx, x, y, 0]:
                flux_map[phalanx_idx, x, y, :2] = counter_clockwise_flux
                flux_map[phalanx_idx, x, y, 2] = -1
                dis_map[phalanx_idx, x, y, 0] = normalize(second_point - np.array([x, y]))[1]
                dis_map[phalanx_idx, x, y, 1] = normalize(np.array([x, y]) - first_point)[1]

            elif np.cross(first_point - np.array([x, y]), direction) >= 0 and binary_skeleton[phalanx_idx, x, y, 0]:
                flux_map[phalanx_idx, x, y, :2] = clockwise_flux
                flux_map[phalanx_idx, x, y, 2] = 1
                dis_map[phalanx_idx, x, y, 0] = normalize(second_point - np.array([x, y]))[1]
                dis_map[phalanx_idx, x, y, 1] = normalize(np.array([x, y]) - first_point)[1]

            else:
                flux_map[phalanx_idx, x, y, :] = 0
                dis_map[phalanx_idx, x, y, :] = 0
    if binary_skeleton[phalanx_idx, :, :, 0].any() != True:
        # print("phalanx: ", phalanx)
        # print("direction: ", direction)
        # print("length: ", length)
        # print("first point: ", first_point)
        # print("second point: ", second_point)
        flux_map[phalanx_idx, int(first_point[0]), int(first_point[1]), 2] = 1
        flux_map[phalanx_idx, int(second_point[0]), int(second_point[1]), 2] = -1


def kp_to_maps(kp, res=64, sigma=1):
    binary_skeleton = np.zeros([20, res, res, 1])

    # maps
    flux_map = np.zeros([20, res, res, 3])
    dis_map = np.zeros([20, res, res, 2])

    wrist_coor = kp[0]
    # 0-4 4-3 3-2 2-1
    for j in range(5):

        # generates the maps of phalanges connected to wrist
        second_point = kp[4 * j + 4]
        first_point = wrist_coor
        fill_maps(flux_map, dis_map, binary_skeleton, 4 * j, first_point, second_point, res, sigma)

        # generates other 3 phalanges of each finger:
        # example: phalange 1 connects kp 4 and kp 3
        for i in range(3):
            curr = 4 * j + 4 - i
            first_point = kp[curr]
            second_point = kp[curr - 1]
            fill_maps(flux_map, dis_map, binary_skeleton, 4 * j + i + 1, first_point, second_point, res, sigma)

    return flux_map, dis_map


# Here the idx is the batch number
def draw_maps(flux, dis, img, idx):
    direction = np.zeros([flux.shape[1], flux.shape[2]])
    magnitude = np.zeros([flux.shape[1], flux.shape[2]])
    dis_to_first = np.zeros([dis.shape[1], dis.shape[2]])
    dis_to_second = np.zeros([dis.shape[1], dis.shape[2]])
    for bone in range(flux.shape[0]):
        for x in range(flux.shape[1]):
            for y in range(flux.shape[2]):
                if flux[bone, x, y, 0] != 0:
                    dis_to_first[y, x] = dis[bone, x, y, 0]
                    dis_to_second[y, x] = dis[bone, x, y, 1]
                    direction[y, x] = flux[bone, x, y, 1] / flux[bone, x, y, 0]
                    magnitude[y, x] = flux[bone, x, y, 2]
                    if flux[bone, x, y, 0] > 0:
                        direction[y, x] = np.degrees(np.arctan(direction[y, x]))
                    if flux[bone, x, y, 0] < 0 < flux[bone, x, y, 1]:
                        direction[y, x] = np.degrees(np.arctan(direction[y, x])) + 180
                    if flux[bone, x, y, 1] < 0 and flux[bone, x, y, 0] < 0:
                        direction[y, x] = np.degrees(np.arctan(direction[y, x])) - 180

    # 作图阶段
    fig = plt.figure()

    ax_1 = fig.add_subplot(221)
    plt.sca(ax_1)
    ax_1.set_title("flux_map_direction")
    sns.heatmap(data=direction,
                cmap=sns.diverging_palette(250, 15, sep=12, n=20, center="dark"),  # 区分度显著色盘：sns.diverging_palette()使用
                vmax=180,
                vmin=-180
                )

    ax_2 = fig.add_subplot(222)
    plt.sca(ax_2)
    ax_2.set_title("flux_map_magnitude")
    sns.heatmap(data=magnitude)

    ax_3 = fig.add_subplot(223)
    plt.sca(ax_3)
    ax_3.set_title("dis_map_to_the_first_keypoint")
    sns.heatmap(data=dis_to_first)

    ax_4 = fig.add_subplot(224)
    plt.sca(ax_4)
    ax_4.set_title("dis_map_to_the_second_keypoint")
    sns.heatmap(data=dis_to_second)

    maps_save_path = osp.join("sample", f"maps_{idx}.jpg")
    plt.savefig(maps_save_path)
    plt.close()

    img = img.reshape((256, 256, 3))
    plt.imshow(img)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    img_save_path = osp.join("sample", f"maps_{idx}.jpg")
    plt.savefig(img_save_path)
    plt.close()


