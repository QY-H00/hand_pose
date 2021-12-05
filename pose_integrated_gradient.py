import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import numpy as np
import torch
import cv2
import pickle
from kornia.geometry.dsnt import spatial_softmax_2d, spatial_softargmax_2d
from IG.SaliencyModel.attributes import attr_grad
from IG.SaliencyModel.BackProp import attribution_objective
from IG.SaliencyModel.BackProp import saliency_map_PG as saliency_map
from IG.SaliencyModel.BackProp import GaussianBlurPath
from IG.SaliencyModel.utils import cv2_to_pil, pil_to_cv2, gini, grad_abs_norm, vis_saliency, vis_saliency_kde
from IG.SaliencyModel.utils import interpolation, isotropic_gaussian_kernel, make_pil_grid
from IG.ModelZoo.utils import _add_batch_one
from data.rhd_dataset import RHDDateset
from progress.bar import Bar


def regress25d(heatmaps, beta):
    bs = heatmaps.shape[0]
    betas = beta.clone().view(1, 21, 1).repeat(bs, 1, 1)
    uv_heatmaps = spatial_softmax_2d(heatmaps, betas)
    coord_out = spatial_softargmax_2d(uv_heatmaps, normalized_coordinates=False)
    coord_out_final = coord_out.clone()

    return coord_out_final.view(bs, 21, 2)


def Path_gradient(numpy_image, model, gt, attr_objective, path_interpolation_func, joint_number, noise = None, cuda=False):
    """
    :param path_interpolation_func:
        return \lambda(\alpha) and d\lambda(\alpha)/d\alpha, for \alpha\in[0, 1]
        This function return pil_numpy_images
    :return:
    """
    if cuda:
        model = model.cuda()

    cv_numpy_image = np.moveaxis(numpy_image, 0, 2)
    if noise is None:
        noise =  np.zeros_like(numpy_image)


    image_interpolation, lambda_derivative_interpolation = path_interpolation_func(cv_numpy_image)
    grad_accumulate_list = np.zeros_like(image_interpolation)
    result_list = []
    for i in range(image_interpolation.shape[0]):
        img_tensor = torch.from_numpy(image_interpolation[i]).cuda()
        img_tensor = img_tensor + torch.from_numpy(noise).float().cuda()
        img_tensor.requires_grad_(True)
        if cuda:
            result = model(_add_batch_one(img_tensor).cuda())
            # result = regress25d(result, beta)
            result = result[:, joint_number:joint_number+1]
            target = torch.exp(-0.3*torch.linalg.norm(result - gt.cuda()))
            target.backward()
            grad = img_tensor.grad.cpu().detach().numpy()
            if np.any(np.isnan(grad)):
                grad[np.isnan(grad)] = 0.0
        else:
            result = model(_add_batch_one(img_tensor))
            target = attr_objective(result)
            target.backward()
            grad = img_tensor.grad.cpu().detach().numpy()
            if np.any(np.isnan(grad)):
                grad[np.isnan(grad)] = 0.0

        grad_accumulate_list[i] = grad * lambda_derivative_interpolation[i]
        result_list.append(result.cpu().detach().numpy())
    results_numpy = np.asarray(result_list)
    return grad_accumulate_list, results_numpy, image_interpolation


def GaussianLinearPath(sigma, fold, l=5):
    def path_interpolation_func(cv_numpy_image):
        kernel = isotropic_gaussian_kernel(l, sigma)
        baseline_image = cv2.filter2D(cv_numpy_image, -1, kernel)
        image_interpolation = interpolation(cv_numpy_image, baseline_image, fold, mode='linear').astype(np.float32)
        lambda_derivative_interpolation = np.repeat(np.expand_dims(cv_numpy_image - baseline_image, axis=0), fold, axis=0)
        return np.moveaxis(image_interpolation, 3, 1).astype(np.float32), \
               np.moveaxis(lambda_derivative_interpolation, 3, 1).astype(np.float32)
    return path_interpolation_func


def invert_GaussianLinearPath(sigma, fold, l=5):
    def path_interpolation_func(cv_numpy_image):
        kernel = isotropic_gaussian_kernel(l, sigma)
        baseline_image = cv_numpy_image - cv2.filter2D(cv_numpy_image, -1, kernel)
        image_interpolation = interpolation(cv_numpy_image, baseline_image, fold, mode='linear').astype(np.float32)
        lambda_derivative_interpolation = np.repeat(np.expand_dims(cv_numpy_image - baseline_image, axis=0), fold, axis=0)
        return np.moveaxis(image_interpolation, 3, 1).astype(np.float32), \
               np.moveaxis(lambda_derivative_interpolation, 3, 1).astype(np.float32)
    return path_interpolation_func


def get_pose_integrated_gradient(model, uv_crop, img_crop, joint_number,
                                 window_size=4, sigma=1.2, fold=50, l=9, alpha=0.5, smooth=True):
    img_crop = img_crop.permute(1, 2, 0)

    min = torch.min(torch.min(img_crop, dim=0)[0], dim=0)[0]
    max = torch.max(torch.max(img_crop, dim=0)[0], dim=0)[0]
    img_show = (img_crop - min) / (max - min)
    img_show = np.array(img_show)
    img_show = img_show * 255

    uv_crop_zero = uv_crop[joint_number] * 4
    w = uv_crop_zero[0].type(torch.IntTensor)
    h = uv_crop_zero[1].type(torch.IntTensor)
    draw_img = pil_to_cv2(img_show)
    cv2.rectangle(draw_img, (int(w - window_size // 2), int(h - window_size // 2)),
                  (int(w + window_size // 2), int(h + window_size // 2)),
                  (0, 0, 1), 2)
    w = w // 4
    h = h // 4
    window_size = window_size // 4
    position_pil = cv2_to_pil(draw_img)

    attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
    gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)
    img_tensor = img_crop.permute(2, 0, 1)
    image_numpy = img_tensor.cpu().detach().numpy()

    if smooth:
        stdev = 0.01 * (np.max(image_numpy) - np.min(image_numpy))
        total_gradients = np.zeros_like(image_numpy, dtype=np.float32)
        for _ in range(25):
            noise = np.random.normal(0, stdev, image_numpy.shape)
            interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(image_numpy, model,
                                                                                      uv_crop[joint_number],
                                                                                      attr_objective,
                                                                                      gaus_blur_path_func,
                                                                                      joint_number,
                                                                                      noise=noise,
                                                                                      cuda=True)
            grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)
            total_gradients += (grad_numpy * grad_numpy)
        grad_numpy_final = total_gradients / 25
    else:
        interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(image_numpy, model,
                                                                                  uv_crop[joint_number], attr_objective,
                                                                                  gaus_blur_path_func, joint_number,
                                                                                  cuda=True)
        grad_numpy_final, result = saliency_map(interpolated_grad_numpy, result_numpy)

    abs_normed_grad_numpy = grad_abs_norm(grad_numpy_final)
    saliency_image_abs = vis_saliency(abs_normed_grad_numpy, zoomin=1)
    saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy)
    saliency_image_kde = saliency_image_kde.resize(position_pil.size)

    blend_abs_and_input = cv2_to_pil(pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_show) * alpha)
    blend_kde_and_input = cv2_to_pil(pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_show) * alpha)
    pil = make_pil_grid(
        [position_pil,
         saliency_image_abs,
         saliency_image_kde,
         blend_abs_and_input,
         blend_kde_and_input]
    )
    return abs_normed_grad_numpy, pil


parser = argparse.ArgumentParser(
        description='PyTorch Train Hourglass On 2D Keypoint Detection')
parser.add_argument(
    '-m',
    '--mode',
    type=str,
    default='regression',
    help='regression/softargmax/detection'
)
parser.add_argument(
    '-s',
    '--start_idx',
    type=int,
    default=0,
    help='regression/softargmax/detection'
)
parser.add_argument(
    '-e',
    '--end_idx',
    type=int,
    default=31,
    help='regression/softargmax/detection'
)
args = parser.parse_args()
mode = args.mode
start_sample_size = args.start_idx
end_sample_size = args.end_idx
batch_size = 32
model = torch.load(f'trained_model_{mode}/model.pkl')
beta = torch.mul(torch.ones(21), 5).cuda()

val_dataset = RHDDateset('RHD_published_v2/', 'evaluation', input_size=256, output_full=True, aug=False)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True
)

'''Store the cropped image and corresponding heatmap'''
inner_index = end_sample_size % batch_size
batch_index = end_sample_size // batch_size
out = []
for i, sample in enumerate(val_loader):
    if i < start_sample_size / batch_size:
        continue
    for j in range(batch_size):
        sample_num = i * batch_size + j
        if sample_num >= end_sample_size:
            break
        img_crop = sample["img_crop"][j]
        uv_crop = sample["uv_crop"][j]
        bar = Bar(f'\033[31m PIG_{sample_num} \033[0m', max=21)
        for joint_number in range(21):
            grad_np, pil = get_pose_integrated_gradient(model, uv_crop, img_crop, joint_number)
            path = os.path.join("sample", f"IG_{sample_num}_joint_{joint_number}")
            if not os.path.exists(path):
                os.mkdir(path)
            pil.save(f"{path}/{mode}.jpg")
            with open(f'{path}/{mode}_grad.pickle', 'wb') as f:
                pickle.dump(grad_np, f)
            bar.next()
        bar.finish()
    if i > end_sample_size / batch_size + 1:
        break
