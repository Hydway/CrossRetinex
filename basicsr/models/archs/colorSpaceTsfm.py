import torch
import torch.nn as nn
import torch.nn.functional as F


class RGB_HVI(nn.Module):
    def __init__(self, input_dim=3, output_dim=3):
        super(RGB_HVI, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 初始化一个随机矩阵并进行正交化
        random_matrix = torch.randn(input_dim, output_dim)
        u, _, v = torch.svd(random_matrix)
        self.transform_matrix = nn.Parameter(u)

        # PHVIT的逆变换矩阵
        self.inverse_transform_matrix = nn.Parameter(v.t())

    def HVIT(self, img):
        # RGB转换到新色彩空间
        # img shape: (batch_size, 3, height, width)
        batch_size, _, height, width = img.shape
        img_flat = img.view(batch_size, 3, -1)  # shape: (batch_size, 3, height * width)

        # 使用正交转换矩阵将RGB转换到新色彩空间
        transformed_flat = torch.matmul(self.transform_matrix,
                                        img_flat)  # shape: (batch_size, output_dim, height * width)

        transformed = transformed_flat.view(batch_size, self.output_dim, height,
                                            width)  # shape: (batch_size, output_dim, height, width)
        return transformed

    def PHVIT(self, img):
        # 从新色彩空间转换回RGB
        # img shape: (batch_size, output_dim, height, width)
        batch_size, _, height, width = img.shape
        img_flat = img.view(batch_size, self.output_dim, -1)  # shape: (batch_size, output_dim, height * width)

        # 使用逆变换矩阵将新色彩空间转换回RGB
        inverse_transformed_flat = torch.matmul(self.inverse_transform_matrix,
                                                img_flat)  # shape: (batch_size, input_dim, height * width)

        inverse_transformed = inverse_transformed_flat.view(batch_size, self.input_dim, height,
                                                            width)  # shape: (batch_size, input_dim, height, width)
        return inverse_transformed
