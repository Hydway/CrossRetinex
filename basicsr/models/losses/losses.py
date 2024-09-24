import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss   #把 l1_loss 作为 weighted_loss 的输入
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss   #把 mse_loss 作为 weighted_loss 的输入
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


# class L1Loss(nn.Module):
#     """L1 (mean absolute error, MAE) loss.
#
#     Args:
#         loss_weight (float): Loss weight for L1 loss. Default: 1.0.
#         reduction (str): Specifies the reduction to apply to the output.
#             Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
#     """
#
#     def __init__(self, loss_weight=1.0, reduction='mean'):
#         super(L1Loss, self).__init__()
#         if reduction not in ['none', 'mean', 'sum']:
#             raise ValueError(f'Unsupported reduction mode: {reduction}. '
#                              f'Supported ones are: {_reduction_modes}')
#
#         self.loss_weight = loss_weight
#         self.reduction = reduction
#
#     def forward(self, pred, target, weight=None, **kwargs):
#         """
#         Args:
#             pred (Tensor): of shape (N, C, H, W). Predicted tensor.
#             target (Tensor): of shape (N, C, H, W). Ground truth tensor.
#             weight (Tensor, optional): of shape (N, C, H, W). Element-wise
#                 weights. Default: None.
#         """
#         return self.loss_weight * l1_loss(
#             pred, target, weight, reduction=self.reduction)

initial_matrix = torch.tensor([[0.299, 0.587, 0.114],
                               [-0.14713, -0.28886, 0.436],
                               [0.615, -0.51499, -0.10001]], dtype=torch.float32).to('cuda')


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss with regularization.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        lambda_reg (float): Weight for the regularization term. Default: 0.1.
        initial_matrix (Tensor): Initial transformation matrix for regularization.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', lambda_reg=0.0, initial_matrix=initial_matrix):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.lambda_reg = lambda_reg
        self.initial_matrix = initial_matrix

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            transform_matrix (Tensor): Transformation matrix used in the model.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        # Calculate L1 loss
        l1 = l1_loss(pred, target, weight, reduction=self.reduction)

        # Calculate regularization loss
        # if self.initial_matrix is not None and transform_matrix is not None:
        #     reg_loss = torch.norm(transform_matrix - self.initial_matrix, p='fro')
        # else:
        #     reg_loss = 0

        # Combine L1 loss and regularization loss
        total_loss = self.loss_weight * l1
        # print("self.loss_weight:", self.loss_weight)
        # print("l1:", l1)
        # print("self.lambda_reg:", self.lambda_reg )
        # print("reg_loss：", reg_loss)
        # print("total_loss:", total_loss)
        # print('='*50)

        return total_loss

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target, transform_matrix=None, weight=None, **kwargs):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class NEWLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False, content_weight=1, style_weight=1e2, tv_weight=10):
        super(NEWLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        # self.base_loss = L1Loss()
        self.base_loss = PSNRLoss()
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight

    def gram(self, X):
        num_channels, n = X.shape[1], X.numel() // X.shape[1]
        X = X.reshape((num_channels, n))
        return torch.matmul(X, X.T) / (num_channels * n)

    def style_loss(self, Y_hat, gram_Y):
        return torch.square(self.gram(Y_hat) - gram_Y.detach()).mean()

    def tv_loss(self, Y_hat):
        return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                      torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())

    def forward(self, pred, target, transform_matrix=None, weight=None, **kwargs):

        # PSNR Loss
        # psnr_loss = self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        # psnr_loss = self.loss_weight * self.scale * torch.log( 1 / (((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8)).mean()
        # psnr_loss = self.loss_weight * self.scale * (((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

        # Style Loss
        gram_target = self.gram(target)
        style_loss = self.style_weight * self.style_loss(pred, gram_target)

        # Total Variation Loss
        # tv_loss = self.tv_weight * self.tv_loss(pred)

        # Total Loss
        # l1loss = self.base_loss(pred, target)
        base_loss = self.base_loss(pred, target)

        # style_loss = torch.log(style_loss)
        # tv_loss = torch.log(tv_loss)

        # total_loss = l1loss + style_loss
        total_loss = base_loss + style_loss

        # print("base_loss:", base_loss)
        # print("l1loss:", l1loss)
        # print("style_loss:", style_loss)
        # print("tv_loss:", tv_loss)
        # print("total_loss:", total_loss)
        # print(">"*50)

        return total_loss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

# def gradient(input_tensor, direction):
#     smooth_kernel_x = torch.reshape(torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32), [2, 2, 1, 1])
#     smooth_kernel_y = torch.transpose(smooth_kernel_x, 0, 1)
#     if direction == "x":
#         kernel = smooth_kernel_x
#     elif direction == "y":
#         kernel = smooth_kernel_y
#     gradient_orig = torch.abs(torch.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
#     grad_min = torch.min(gradient_orig)
#     grad_max = torch.max(gradient_orig)
#     grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
#     return grad_norm

# class SmoothLoss(nn.Moudle):
#     """ illumination smoothness"""

#     def __init__(self, loss_weight=0.15, reduction='mean', eps=1e-2):
#         super(SmoothLoss,self).__init__()
#         self.loss_weight = loss_weight
#         self.eps = eps
#         self.reduction = reduction
    
#     def forward(self, illu, img):
#         # illu: b×c×h×w   illumination map
#         # img:  b×c×h×w   input image
#         illu_gradient_x = gradient(illu, "x")
#         img_gradient_x  = gradient(img, "x")
#         x_loss = torch.abs(torch.div(illu_gradient_x, torch.maximum(img_gradient_x, 0.01)))

#         illu_gradient_y = gradient(illu, "y")
#         img_gradient_y  = gradient(img, "y")
#         y_loss = torch.abs(torch.div(illu_gradient_y, torch.maximum(img_gradient_y, 0.01)))

#         loss = torch.mean(x_loss + y_loss) * self.loss_weight

#         return loss

# class MultualLoss(nn.Moudle):
#     """ Multual Consistency"""

#     def __init__(self, loss_weight=0.20, reduction='mean'):
#         super(MultualLoss,self).__init__()

#         self.loss_weight = loss_weight
#         self.reduction = reduction
    

#     def forward(self, illu):
#         # illu: b x c x h x w
#         gradient_x = gradient(illu,"x")
#         gradient_y = gradient(illu,"y")

#         x_loss = gradient_x * torch.exp(-10*gradient_x)
#         y_loss = gradient_y * torch.exp(-10*gradient_y)

#         loss = torch.mean(x_loss+y_loss) * self.loss_weight
#         return loss




