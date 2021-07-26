import argparse
import torch.backends.cudnn as cudnn
import torch.optim
from data.eval_utils import draw_maps
from data.RHD import RHD_DataReader_With_File
from models.hourglass import NetStackedHourglass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch Test Hourglass On 2D Keypoint Detection')
    args = parser.parse_args()
    hand_crop, hand_flip, use_wrist, BL, root_id, rotate, uv_sigma = True, True, True, 'small', 12, 180, 0.0

    val_dataset = RHD_DataReader_With_File(mode="training", path=None)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=True
    )

    # assert osp.exists(args.image_path), RuntimeError(f'Can not read image! {args.image_path}')
    hm_example = []
    kp_example = None
    image = []
    flux = []
    dis = []
    for i, sample in enumerate(val_loader):
        if i == 0:
            image = sample["img_crop"]
            flux = sample["flux_map"]
            dis = sample["dis_map"]
            break

    # Create the model
    model = torch.load("skeleton_model_after_95_epochs.pkl")
    model = model.to(device)
    preds = model(image)
    final_pred = preds[0][-1]
    pred_maps = final_pred.reshape((32, 20, -1, 5))  # (B, 20, res*res, 5)
    # split it into (B, 20, res*res, 3) and (B, 20, res*res, 2)
    pred_maps = torch.split(pred_maps, split_size_or_sections=[3, 2], dim=-1)
    pred_flux = pred_maps[0].reshape((32, 20, 64, 64, 3))
    pred_dis = pred_maps[1].reshape((32, 20, 64, 64, 2))
    draw_maps(pred_flux[0], pred_dis[0], "pred")
    draw_maps(flux[0], dis[0], "test")
