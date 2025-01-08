import numpy as np
from skimage.metrics import structural_similarity as cal_ssim


def rescale(x):
    return (x - x.max()) / (x.max() - x.min()) * 2 - 1


def MAE(pred, label):
    return np.mean(np.abs(pred-label))


def MRE(pred, label):
    return np.mean(np.abs(pred-label) / (np.abs(label) + 1e-20))


def MSE(pred, label):
    return np.mean((pred-label)**2)


def RMSE(pred, label):
    return np.sqrt(np.mean((pred-label)**2))


def MXRE(pred, label):
    return np.max(np.abs(pred-label) / (np.abs(label) + 1e-20))


def SSIM(pred, label):
    try:
        B, T, H, W = label.shape
        pred = pred.reshape(B, T, H, W)
        label = label.reshape(B, T, H, W)
        ssim = []
        for b in range(B):
            for t in range(T):
                ssim.append(cal_ssim(pred[b, t], label[b, t], data_range = label[b, t].max() - label[b, t].min()))
    except:
        B, H, W = label.shape
        pred = pred.reshape(B, H, W)
        label = label.reshape(B, H, W)
        ssim = []
        for b in range(B):
            ssim.append(cal_ssim(pred[b], label[b], data_range = label[b].max() - label[b].min()))

    return np.mean(ssim)


def metric(pred, label, scaler = None, metrics=['mae', 'mse', 'mre'],
         return_log=True, mode = None):
    """The evaluation function to output metrics.

    Args:
        pred (tensor): The prediction values.
        label (tensor): The Original values.
        scaler (sklearn.preprocessing): Denormalize values
        metric (str | list[str]): Metrics to be evaluated.
        return_log (bool): Whether to return the log string.

    Returns:
        dict: evaluation results
    """
    if scaler :
        if len(label.shape)==4:
            B, T, H, W = label.shape
            for b in range(B):
                for t in range(T):
                    pred[b, t] = scaler.inverse_transform(pred[b, t])
                    label[b, t] = scaler.inverse_transform(label[b, t])
        elif len(label.shape)==3:
            B, H, W = label.shape
            for b in range(B):
                pred[b] = scaler.inverse_transform(pred[b])
                label[b] = scaler.inverse_transform(label[b])


    eval_log = ""
    allowed_metrics = ['mae', 'mse', 'rmse', 'ssim', 'mre', 'mxre']
    invalid_metrics = set(metrics) - set(allowed_metrics)
    if len(invalid_metrics) != 0:
        raise ValueError(f'metric {invalid_metrics} is not supported.')
    
    eval_res = {}
    if 'mse' in metrics:
        eval_res['mse'] = MSE(pred, label)

    if 'mae' in metrics:
        eval_res['mae'] = MAE(pred, label)

    if 'rmse' in metrics:
        eval_res['rmse'] = RMSE(pred, label)
    
    if 'mre' in metrics:
        eval_res['mre'] = MRE(pred, label)
    
    if 'mxre' in metrics:
        eval_res['mxre'] = MXRE(pred, label)

    if 'ssim' in metrics:
        eval_res['ssim'] = SSIM(pred, label)

    if return_log:
        for k, v in eval_res.items():
            eval_str = f"{k}:{v:.5e}" if len(eval_log) == 0 else f", {k}:{v:.5e}"
            eval_log += eval_str

        if mode:
            eval_log = f"{mode}:\n" + eval_log

    return eval_res, eval_log
