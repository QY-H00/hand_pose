import data.eval_utils as eval_utils


class KeypointLoss:
    def compute_loss(self, preds, targs, infos):
        vis = infos['vis'][:, :, None]
        pred_kps = preds.reshape(preds.shape[0], 21, 2)
        targ_kps = targs['kp2d']  # targ_kps resolution is 256 * 256
        kp_loss = eval_utils.MeanEPE(pred_kps * vis, targ_kps * vis)
        return kp_loss
