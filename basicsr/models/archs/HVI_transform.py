import torch
import torch.nn as nn
import torch.nn.functional as F

class RGB_HVI(nn.Module):
    def __init__(self, input_dim=3, output_dim=3):
        super(RGB_HVI, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 从已知色彩空间初始化转换矩阵（如RGB到YUV）
        initial_matrix = torch.tensor([[0.299, 0.587, 0.114],
                                       [-0.14713, -0.28886, 0.436],
                                       [0.615, -0.51499, -0.10001]], dtype=torch.float32)
        self.initial_matrix = initial_matrix  # 保存初始矩阵
        self.weight_matrix = nn.Parameter(initial_matrix.clone())  # 可学习矩阵作为参数存储

        self.transform_matrix = None  # 用于存储正交化后的转换矩阵
        self.inverse_transform_matrix = None  # 用于存储正交化后的逆转换矩阵

    def orthogonalize(self, matrix):
        # 使用SVD分解确保矩阵的正交性
        u, _, v = torch.svd(matrix)
        return u, v.t()

    def HVIT(self, img):
        # RGB转换到新色彩空间
        # img shape: (batch_size, 3, height, width)
        batch_size, _, height, width = img.shape
        img_flat = img.view(batch_size, 3, -1)  # shape: (batch_size, 3, height * width)

        # 通过SVD分解确保转换矩阵的正交性
        self.transform_matrix, self.inverse_transform_matrix = self.orthogonalize(self.weight_matrix)

        # 使用正交转换矩阵将RGB转换到新色彩空间
        transformed_flat = torch.matmul(self.transform_matrix, img_flat)  # shape: (batch_size, output_dim, height * width)

        transformed = transformed_flat.view(batch_size, self.output_dim, height, width)  # shape: (batch_size, output_dim, height, width)
        return transformed

    def PHVIT(self, img):
        # 从新色彩空间转换回RGB
        # img shape: (batch_size, output_dim, height, width)
        batch_size, _, height, width = img.shape
        img_flat = img.view(batch_size, self.output_dim, -1)  # shape: (batch_size, output_dim, height * width)

        # 使用逆变换矩阵将新色彩空间转换回RGB
        inverse_transformed_flat = torch.matmul(self.inverse_transform_matrix, img_flat)  # shape: (batch_size, input_dim, height * width)

        inverse_transformed = inverse_transformed_flat.view(batch_size, self.input_dim, height, width)  # shape: (batch_size, input_dim, height, width)
        return inverse_transformed

    def get_transform_matrix(self):
        return self.transform_matrix
