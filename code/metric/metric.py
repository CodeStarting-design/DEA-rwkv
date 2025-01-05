import math
from math import exp
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.cuda.amp import autocast


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    # 使用autocast确保半精度计算
    with autocast():
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # 计算 SSIM map
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # 最终计算平均值
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
    # 防止值超出范围
    img1 = torch.clamp(img1, min=0, max=1)
    img2 = torch.clamp(img2, min=0, max=1)

    (_, channel, _, _) = img1.size()

    # 创建一个卷积窗口
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    # 调用 _ssim 来计算结果
    return _ssim(img1, img2, window, window_size, channel, size_average)


def psnr(pred, gt):
    # 确保输入为张量，并限制在[0, 1]之间
    pred = pred.clamp(0, 1).cpu().numpy()
    gt = gt.clamp(0, 1).cpu().numpy()
    
    # 计算每个图像的均方根误差 (RMSE)
    imdff = pred - gt
    rmse = np.sqrt(np.mean(imdff ** 2, axis=(1, 2, 3)))  # 针对每个图像计算 RMSE
    
    # 计算 PSNR
    psnr_values = np.where(rmse == 0, 100, 20 * np.log10(1.0 / rmse))
    
    # 返回均值 PSNR
    return np.mean(psnr_values)