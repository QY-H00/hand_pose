import torch
import numpy as np


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


# def parse_kp2d_to_parse_skeleton(kp):
#     skeleton = np.zeros([21, 3])
#     wrist_coor = kp[0]
#     # 0-4 4-3 3-2 2-1
#     for j in range(5):
#         phalanx = kp[4 * j + 4] - wrist_coor
#         direction, length = normalize(phalanx)
#         skeleton[4 * j][:2] = direction
#         skeleton[4 * j][2] = length
#         for i in range(3):
#             curr_kp = 4 * j + 4 - i
#             phalanx = kp[curr_kp - 1] - kp[curr_kp]
#             direction, length = normalize(phalanx)
#             skeleton[4 * j + i + 1][:2] = direction
#             skeleton[4 * j + i + 1][2] = length
#     skeleton[20][:2] = wrist_coor
#     skeleton[20][2] = 0
#     return skeleton
#
#
# def parse_skeleton_to_parse_kp2d(skeleton):
#     kp = np.zeros([21, 2])
#     wrist_coor = skeleton[20][:2]
#     kp[0] = wrist_coor
#     for j in range(5):
#         kp[4 * j + 4] = wrist_coor + skeleton[4 * j][:2] * skeleton[4 * j][2]
#         kp[4 * j + 3] = kp[4 * j + 4] + skeleton[4 * j + 1][:2] * skeleton[4 * j + 1][2]
#         kp[4 * j + 2] = kp[4 * j + 3] + skeleton[4 * j + 2][:2] * skeleton[4 * j + 2][2]
#         kp[4 * j + 1] = kp[4 * j + 2] + skeleton[4 * j + 3][:2] * skeleton[4 * j + 3][2]
#     return kp
#
#
# def parse_kp2d_to_dense_skeleton(kp, res=64, sigma=1):
#     dense_skeleton = np.zeros([20, res, res, 5])
#     wrist_coor = kp[0]
#     pixels = np.zeros([res, res, 2])
#     pixels[:, :, 0] = range(res)
#     pixels[:, :, 1] = range(res)
#
#     # 0-4 4-3 3-2 2-1
#     for j in range(5):
#         second_point = kp[4 * j + 4]
#         first_point = wrist_coor
#         phalanx = second_point - first_point
#         direction, length = normalize(phalanx)
#         dense_skeleton[4 * j + 4, :, :, 0] = is_on_skeleton(pixels, direction, length, first_point, sigma)
#         dense_skeleton[4 * j + 4, :, :, 1:3] = direction if dense_skeleton[4 * j + 4, :, :, 0] else 0
#         dense_skeleton[4 * j + 4, :, :, 3] = normalize(second_point - pixels)[1] if dense_skeleton[4 * j + 4, :, :, 0] \
#             else 0
#         dense_skeleton[4 * j + 4, :, :, 4] = normalize(pixels - first_point)[1] if dense_skeleton[4 * j + 4, :, :, 0] \
#             else 0
#         for i in range(3):
#             curr = 4 * j + 4 - i
#             first_point = kp[curr]
#             second_point = kp[curr - 1]
#             phalanx = second_point - first_point
#             direction, length = normalize(phalanx)
#             dense_skeleton[4 * j + i + 1, :, :, 0] = is_on_skeleton(pixels, direction, length, first_point, sigma)
#             dense_skeleton[4 * j + i + 1, :, :, 1:3] = direction if dense_skeleton[4 * j + 4, :, :, 0] else 0
#             dense_skeleton[4 * j + i + 1, :, :, 3] = normalize(second_point - pixels)[1] \
#                 if dense_skeleton[4 * j + 4, :, :, 0] else 0
#             dense_skeleton[4 * j + i + 1, :, :, 4] = normalize(pixels - first_point)[1] \
#                 if dense_skeleton[4 * j + 4, :, :, 0] else 0
#     return dense_skeleton
#
#
# def dense_skeleton_to_parse_kp2d(dense_skeleton, res=64, sigma=1):
#     kp = np.zeros([21, 2])
#     count = 0
#     kp[0] = 0
#     for i in range(5):
#         count += np.sum(dense_skeleton[4 * i, :, :, 0])
#         kp[0] += dense_skeleton[4 * i, :, :, 0] * dense_skeleton[4 * i, :, :, 1:3] * dense_skeleton[4 * i, :, :, 4]
#     kp[0] = kp[0] / count
#
#     for j in range(5):
#         count = np.sum(dense_skeleton[4 * j + 0, :, :, 0]) + np.sum(dense_skeleton[4 * j + 1, :, :, 0])
#         kp[4 * j + 4] = (dense_skeleton[4 * j + 0, :, :, 0]
#                          * dense_skeleton[4 * j + 0, :, :, 1:3] * dense_skeleton[4 * j + 0, :, :, 3]
#                          + dense_skeleton[4 * j + 1, :, :, 0]
#                          * dense_skeleton[4 * j + 1, :, :, 1:3] * dense_skeleton[4 * j + 1, :, :, 4]) / count
#         count = np.sum(dense_skeleton[4 * j + 1, :, :, 0]) + np.sum(dense_skeleton[4 * j + 2, :, :, 0])
#         kp[4 * j + 3] = (dense_skeleton[4 * j + 1, :, :, 0]
#                          * dense_skeleton[4 * j + 1, :, :, 1:3] * dense_skeleton[4 * j + 1, :, :, 3]
#                          + dense_skeleton[4 * j + 2, :, :, 0]
#                          * dense_skeleton[4 * j + 2, :, :, 1:3] * dense_skeleton[4 * j + 2, :, :, 4]) / count
#         count = np.sum(dense_skeleton[4 * j + 2, :, :, 0]) + np.sum(dense_skeleton[4 * j + 3, :, :, 0])
#         kp[4 * j + 2] = (dense_skeleton[4 * j + 2, :, :, 0]
#                          * dense_skeleton[4 * j + 2, :, :, 1:3] * dense_skeleton[4 * j + 2, :, :, 3]
#                          + dense_skeleton[4 * j + 3, :, :, 0]
#                          * dense_skeleton[4 * j + 3, :, :, 1:3] * dense_skeleton[4 * j + 3, :, :, 4]) / count
#         count = np.sum(dense_skeleton[4 * j + 3, :, :, 0])
#         kp[4 * j + 1] = (dense_skeleton[4 * j + 3, :, :, 0]
#                          * dense_skeleton[4 * j + 3, :, :, 1:3] * dense_skeleton[4 * j + 3:, :, 3]) / count
#     return kp


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
    
    for idx in range(20):
        flux = flux_map[idx, :, :, :2]
        for x in range(64):
            for y in range(64):
                length = flux[x, y, 0] * flux[x, y, 0] + flux[x, y, 1] * flux[x, y, 1]
                # if length - 1 > 0.0001 and length - 0 > 0.0001:
                #     print(idx, x, y, flux[x, y])
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


