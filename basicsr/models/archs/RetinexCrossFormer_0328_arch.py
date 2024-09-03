import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from .dce_model import enhance_net_nopool
# from .dcepp_model import enhance_net_nopool
from pdb import set_trace as stx
import os
from .HVI_transform import RGB_HVI
from datetime import datetime

# import cv2
# print('Executing file:', __file__)

import torch
import torch.nn as nn
import torch.nn.functional as F

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# input [bs,28,256,310]  output [bs, 28, 256, 256]
def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]


class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  # __init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()
        input = torch.cat([img, mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super(Estimator, self).__init__()

        self.conv1_fea = nn.Conv2d(n_fea_in, n_fea_in, kernel_size=3, padding=1, bias=True)
        self.conv2_fea = nn.Conv2d(n_fea_in, n_fea_in, kernel_size=3, padding=1, bias=True)
        self.norm1 = nn.LayerNorm(n_fea_in)

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            # groups=4
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=4)
        self.norm2 = nn.LayerNorm(n_fea_middle)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)
        self.norm3 = nn.LayerNorm(n_fea_out)


    def forward(self, img):
        
        fea = self.conv1_fea(img)
        fea = self.conv2_fea(fea)
        fea = fea.permute(0, 2, 3, 1)  # Change shape from (N, C, H, W) to (N, H, W, C)
        fea = self.norm1(fea)
        fea = fea.permute(0, 3, 1, 2)  # Change shape back to (N, C, H, W)

        x_1 = self.conv1(img)
        illu_fea = self.depth_conv(x_1)
        illu_fea = illu_fea.permute(0, 2, 3, 1)
        illu_fea = self.norm2(illu_fea)
        illu_fea = illu_fea.permute(0, 3, 1, 2)

        illu_map = self.conv2(illu_fea)
        illu_map = illu_map.permute(0, 2, 3, 1)
        illu_map = self.norm3(illu_map)
        illu_map = illu_map.permute(0, 3, 1, 2)

        return fea, illu_fea, illu_map



class Cross_Attention_MSA(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, dim_k=3, dim_head_k=3):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.dim_k = dim_k
        self.dim_head_k = dim_head_k
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim_k, dim_head_k * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head_k * heads, dim_k, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim_k, dim_k, 3, 1, 1, bias=False, groups=dim_head_k),
            GELU(),
            nn.Conv2d(dim_k, dim_k, 3, 1, 1, bias=False, groups=dim_head_k),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans, illu_map_trans):

        b, h, w, c = x_in.shape
        b1, h1, w1, c1 = illu_map_trans.shape
        # print("x_in shape: ", x_in.size())

        x_q = x_in.reshape(b, h * w, c)
        # 修改为 k v，分别为 illu_map illu_fea
        # print("illu_fea_trans size: ", illu_fea_trans.size())
        # print("illu_map_trans size: ", illu_map_trans.size())
        x_k = illu_map_trans.reshape(b1, h1 * w1, c1)
        x_v = illu_fea_trans.reshape(b, h * w, c)
        # print("x shape: ", x_q.size())
        q_inp = self.to_q(x_q)
        # print("dim: ", self.dim)
        # print("num_heads: ", self.num_heads)
        # print("dim heads: ", self.dim_head)
        k_inp = self.to_k(x_k)
        v_inp = self.to_v(x_v)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                      (q_inp, k_inp, v_inp))

        # normalization
        q = q / self.dim ** 0.5
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        # k = F.normalize(k, dim=-1, p=2)
        # v = F.normalize(v, dim=-1, p=2)

        # print('*'*50)
        # print("q_size: ", q.size())
        # print("k_size: ", k.size())
        # print("v_size: ", v.size())
        # print("k__transpose_size: ", k.transpose(-2, -1).size())

        # attention
        # print("attn size: ", (k @ q.transpose(-2, -1)).size())
        # print("heads num:", self.num_heads)

        attn = k @ q.transpose(-2, -1)
        # print("attn = k @ q: ", attn.size())
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        # print("out = attn @ v:", out.size())
        out = out.permute(0, 3, 2, 1)
        # print("out size: ", out.size())
        out = out.reshape(b1, h1 * w1, self.num_heads * self.dim_head_k)

        # output projection

        # print(illu_map_trans.shape)
        out = self.proj(out)

        out = out.view(illu_map_trans.shape)
        # print("out: ", out.size())
        # print(b1, h1, w1, c1)
        # print("k_inp: ", k_inp.size())
        out_p = self.pos_emb(k_inp.reshape(b1, h1, w1, c1).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)

        out = out + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class FFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        hidden_dim = dim * mult

        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            GELU(),
            nn.Linear(hidden_dim, dim, bias=False)
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x.shape
        # print("b c h w:", x.shape)
        x = x.view(b * h * w, c)  # 展平为二维张量
        x = self.ffn(x)
        x = x.view(b, h, w, c)  # 恢复为四维张量
        return x


class IGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            dim_k=3,
            dim_head_k=3,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.dim_k = dim_k
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                Cross_Attention_MSA(dim=dim, dim_head=dim_head, heads=heads, dim_k=dim_k, dim_head_k=dim_head_k),
                PreNorm(dim, FFN(dim=dim))
            ]))

    def forward(self, x, illu_fea, illu_map):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            # print("x in attn: ", x.size())
            x_attn = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1), illu_map_trans=illu_map.permute(0, 2, 3, 1))
            # print("x_attn size: ", x_attn.size())
            # x[:,:,:,:self.dim_k] = x_attn + x[:,:,:,:self.dim_k].clone()
            result = x_attn + x[:, :, :, :self.dim_k].clone()
            x = torch.cat((result, x[:, :, :, self.dim_k:]), dim=-1)
            # print("x out size: ", x.size())
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        # print("IGAB out:", out.size())
        return out


class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, n_feat=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser, self).__init__()
        self.dim = dim # n_feat
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        # print("dim_level:", dim_level)
        n_fea_out = 10 # illu_map 初始输出 channel 数
        for i in range(level):
            k_dim = (i + 1) * n_fea_out
            # print("k_dim", k_dim)
            # print(dim_level, k_dim, k_dim)
            self.encoder_layers.append(nn.ModuleList([
                Estimator(n_fea_middle=dim_level, n_fea_in=dim_level, n_fea_out=k_dim),
                IGAB(
                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim, dim_k=k_dim, dim_head_k=k_dim // (dim_level // dim)),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),  # FeaDownSample
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),  # IlluFeaDownsample
                nn.Conv2d(k_dim, k_dim * 2, 4, 2, 1, bias=False),  # IlluMapDownsample
            ]))
            dim_level = dim_level * 2

        # Bottleneck
        # print("k_dim in bottleneck: ", k_dim)
        self.bottleneck = IGAB(
            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1], dim_k=k_dim*2, dim_head_k=k_dim*2//(dim_level // dim))

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            k_dim_decoder = (level - i) * n_fea_out
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                Estimator(n_fea_middle=dim_level // 2, n_fea_in=dim_level // 2, n_fea_out=k_dim_decoder),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),  # fea
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False), # illu_fea_Fution
                nn.Conv2d(k_dim_decoder * 2, k_dim_decoder, 1, 1, bias=False), # illu_map_Fution
                IGAB(
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim, dim_k=k_dim_decoder, dim_head_k=k_dim_decoder // ((dim_level // 2) // dim)),
            ]))
            dim_level = dim_level // 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea=None, illu_map=None):
        """
        x:          [b,c,h,w]         x是feature, 不是image
        illu_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)
        # fea = x

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        illu_map_list = []
        for (Estimator, IGAB, FeaDownSample, IlluFeaDownsample, IlluMapDownsample) in self.encoder_layers:
            # print(">"*50)
            # print("fea size:", fea.size())
            # print("illu_fea size:", illu_fea.size())
            # print("illu_map size:", illu_map.size())

            fea, illu_fea, illu_map = Estimator(fea)
            # print("illu_fea in size: ", illu_fea.size())
            # print("illu_map in size: ", illu_map.size())

            fea = IGAB(fea, illu_fea, illu_map)  # bchw

            illu_fea_list.append(illu_fea)
            illu_map_list.append(illu_map)
            fea_encoder.append(fea)

            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)
            illu_map = IlluMapDownsample(illu_map)

        # print(">"*50)
        # print("Bottleneck")
        # Bottleneck
        fea = self.bottleneck(fea, illu_fea, illu_map)
        # print(">" * 50)
        # print("1 fea size:", fea.size())

        # Decoder
        for i, (FeaUpSample, Estimator, Fution, illu_fea_Fution, illu_map_Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            # print(">" * 50)
            fea = FeaUpSample(fea)
            fea, illu_fea, illu_map = Estimator(fea)
            # print("2 fea size:", fea.size())

            # 转置卷积
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))

            # print("illu_fea size:", illu_fea.size())
            # print("illu_fea_list[self.level - 1 - i] size:", illu_fea_list[self.level - 1 - i].size())
            illu_fea = illu_fea_Fution(
                torch.cat([illu_fea, illu_fea_list[self.level - 1 - i]], dim=1))
            # print("illu_map size:", illu_map.size())
            # print("illu_map_list[self.level - 1 - i] size:", illu_map_list[self.level - 1 - i].size())
            illu_map = illu_map_Fution(
                torch.cat([illu_map, illu_map_list[self.level - 1 - i]], dim=1))

            # print("3 fea size:", fea.size())
            fea = LeWinBlcok(fea, illu_fea, illu_map)
            # print("4 fea size:", fea.size())

        # Mapping
        out = self.mapping(fea) + x

        return out


class RetinexFormer_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1]):
        super(RetinexFormer_Single_Stage, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level,
                                 num_blocks=num_blocks)  #### 将 Denoiser 改为 img2img

    def forward(self, img):
        # img:        b,c=3,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        # illu_fea, illu_map = self.estimator(img)
        # input_img = img * illu_map + img
        # output_img = self.denoiser(input_img, illu_fea, illu_map)

        output_img = self.denoiser(img)

        return output_img


class RetinexCrossFormer_0328(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=[1, 1, 1]):
        super(RetinexCrossFormer_0328, self).__init__()
        self.stage = stage
        self.trans = RGB_HVI().cuda()
        modules_body = [
            RetinexFormer_Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2,
                                       num_blocks=num_blocks)
            for _ in range(stage)]

        self.body = nn.Sequential(*modules_body)


        # self.DCE_net = enhance_net_nopool(scale_factor = 12).cuda()
        self.DCE_net = enhance_net_nopool().cuda()
        base_dir = os.path.dirname(__file__)
        snapshot_path = os.path.join(base_dir, 'snapshots', 'Epoch99.pth')
        self.DCE_net.load_state_dict(torch.load(snapshot_path))

        # print(">"*50)
        print("training start:", str(datetime.now()))
        # print("<"*50)


    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        with torch.no_grad():
            _, enhanced_image, _ = self.DCE_net(x)

        # adj_x = self.trans.HVIT(enhanced_image)

        out_rgb = self.body(enhanced_image)

        return out_rgb

# if __name__ == '__main__':
#     from fvcore.nn import FlopCountAnalysis
#     model = RetinexCrossFormer_0328(stage=1,n_feat=40,num_blocks=[1,2,2]).cuda()
#     print(model)
#     inputs = torch.randn((1, 3, 256, 256)).cuda()
#     flops = FlopCountAnalysis(model,inputs)
#     n_param = sum([p.nelement() for p in model.parameters()])  # 所有参数数量
#     print(f'GMac:{flops.total()/(1024*1024*1024)}')
#     print(f'Params:{n_param}')