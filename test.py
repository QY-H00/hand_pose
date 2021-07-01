import argparse
import torch.backends.cudnn as cudnn
import torch.optim
import os.path as osp
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import sys
from data.RHD import RHD_DataReader
import eval_utils
from torchvision import transforms

from torch import FloatTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def transform_hm_to_kp2d(hm):
    hm = hm.detach().cpu().numpy()
    kp2d = np.zeros([21, 2])
    for i in range(21):
        hm[i] = hm[i] / np.sum(hm[i])
        for x in range(64):
            for y in range(64):
                kp2d[i][0] += hm[i][x][y] * x
                kp2d[i][1] += hm[i][x][y] * y
    return kp2d

def transform_hm_to_kp2d_with_highest_prob(hm):
    hm = hm.detach().cpu().numpy()
    kp2d = np.zeros([21, 2])
    for i in range(21):
        print(np.sum(hm[i]))
        prob = hm[i][0][0]
        # hm[i] = np.exp(hm[i]) / np.sum(np.exp(hm[i]))
        for x in range(64):
            for y in range(64):
                if hm[i][x][y] > prob:
                    kp2d[i][0] = x
                    kp2d[i][1] = y
    return kp2d


def vis(img, kp2d):
    # 参数
    index = 3
    filename = 'label_test/%d.txt' % index  #
    point_size = 2  # 点的尺寸
    color = (255, 255, 255)  # 点和文字的颜色，全白
    color_1 = (255, 0, 0)
    color_2 = (0, 255, 0)
    color_3 = (0, 0, 255)
    color_4 = (0, 255, 255)
    color_5 = (255, 0, 255)
    colors = [color_1, color_2, color_3, color_4, color_5]
    point_thickness = 2  # 点的圈圈的粗细
    line_thickness = 1
    text_thickness = 1
    text_size = 1  # 文字尺寸

    img1 = cv2.imread(img)
    b, g, r = cv2.split(img1)
    img1 = cv2.merge([r, g, b])

    # 标注图像
    plt.subplot(121)
    plt.imshow(img1)
    wrist_y, wrist_x = kp2d[0]
    wrist_x = int(wrist_x) * 4
    wrist_y = int(wrist_y) * 4
    wrist = (wrist_x, wrist_y)
    pre_point = wrist
    cv2.circle(img1, wrist, point_size, color, point_thickness)
    for i in range(5):
        for j in range(4):
            y, x = kp2d[4 * i + 4 - j]
            x = int(x) * 4
            y = int(y) * 4
            point = (x, y)
            cv2.circle(img1, point, point_size, colors[i], point_thickness)
            cv2.line(img1, pre_point, point, colors[i], line_thickness)
            pre_point = point
        pre_point = wrist
            # cv2.putText(img1, "%d" % (i + 1), point, cv2.FONT_HERSHEY_SIMPLEX, text_size, color, text_thickness)

    plt.imshow(img1)
    plt.show()


def draw_maps(flux, dis):
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
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch Test Hourglass On 2D Keypoint Detection')
    # Dataset setting
    # parser.add_argument(
    #     '-img',
    #     '--image_path',
    #     type=str,
    #     default="00003.png",
    #     help='input image'
    # )
    args = parser.parse_args()
    hand_crop, hand_flip, use_wrist, BL, root_id, rotate, uv_sigma = True, True, True, 'small', 12, 180, 0.0
    train_dataset = RHD_DataReader(path="RHD_published_v2", mode='training', hand_crop=hand_crop, use_wrist_coord=use_wrist,
                                   sigma=5,
                                   data_aug=False, uv_sigma=uv_sigma, rotate=rotate, BL=BL, root_id=root_id,
                                   right_hand_flip=hand_flip, crop_size_input=256)
    val_dataset = RHD_DataReader(path="RHD_published_v2", mode='training', hand_crop=hand_crop, use_wrist_coord=use_wrist,
                                 sigma=5,
                                 data_aug=False, uv_sigma=uv_sigma, rotate=rotate, BL=BL, root_id=root_id,
                                 right_hand_flip=hand_flip, crop_size_input=256)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        pin_memory=True
    )

    # assert osp.exists(args.image_path), RuntimeError(f'Can not read image! {args.image_path}')
    hm_example = []
    kp_example = None
    image = []
    for i, sample in enumerate(train_loader):
        if i == 1:
            image = sample["img_crop"][0]
            hm_example = sample["hm"][0]
            kp_example = sample["uv_crop"][0]
            print(kp_example)
            mask = sample["mask_crop"][0]
            flux_map = sample["flux_map"][0]
            dis_map = sample["dis_map"][0]
            break
        print("finish sample ", i)
    kp = kp_example.detach().cpu().numpy()
    flux_map = flux_map.numpy()
    dis_map = dis_map.numpy()
    # draw_maps(flux_map, dis_map)
    kp = eval_utils.maps_to_kp(flux_map, dis_map, res=64)
    
    kp = kp * 4
    print(kp.astype(int))
    # image = image.reshape(1, 3, 256, 256)
    # mask = FloatTensor(mask).to(device, non_blocking=True)
    # unloader = transforms.ToPILImage()
    # mask = unloader(mask)
    # mask.save('image.jpg')
    # print(image.shape)
    # trained_model = torch.load("hourglass_model.pkl")
    # trained_model.eval()
    # out, _ = trained_model(image)
    # hm = out[-1][0]
    # image_copy = hm_example[0].cpu().clone()  # clone the tensor
    # image_copy = image_copy.squeeze(0)  # remove the fake batch dimension
    # image_copy = unloader(image_copy)
    # image_copy.save('heatmap.jpg')
    # image = image.reshape(3, 256, 256)
    # image_copy = image.cpu().clone()  # clone the tensor
    # image_copy = image_copy.squeeze(0)  # remove the fake batch dimension
    # image_copy = unloader(image_copy)
    # image_copy.save('example.jpg')
    # kp2d = transform_hm_to_kp2d(hm_example)
    # print(kp2d[0])
    # kp_example = np.array(kp_example / 4, dtype=int)
    # print(kp_example[0])
    # vis('image.jpg', kp2d)
