import torch,os
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
from torch.nn.modules import transformer
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.linear import Linear
from x_transformers import Encoder
from timm.models.layers import DropPath, trunc_normal_
from thop import profile
from torch.autograd import Variable
import torchvision.transforms.functional as TF
from torchvision import transforms, utils
import numpy as np
from data_loader import pCRDataset
from torch.utils.data import DataLoader
import copy
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

# helpers


def exists(val):
    return val is not None


def conv_output_size(image_size, kernel_size, stride, padding=0):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)


def cast_tuple(val, num):
    return val if isinstance(val, tuple) else (val,) * num


# classes

# Custom transformation to convert a tensor to double type
class ToDoubleTensor(object):
    def __call__(self, pic):
        return torch.tensor(pic, dtype=torch.float32)

class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h=int(math.sqrt(x.shape[1])))


def pair(t):
    # 把t变成一对输出
    return t if isinstance(t, tuple) else (t, t)

class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor
        gamma:
        ignore_index:
        reduction:
    """

    def __init__(self, num_class, alpha=None, gamma=2, ignore_index=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.ignore_index = ignore_index
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, )
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, logit, target):
        # assert isinstance(self.alpha,torch.Tensor)\
        N, C = logit.shape[:2]
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)
        if prob.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        ori_shp = target.shape
        target = target.view(-1, 1)  # [N,d1,d2,...]->[N*d1*d2*...,1]
        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            target = target * valid_mask

        # ----------memory saving way--------
        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.view(-1))
        alpha_class = alpha[target.squeeze().long()]
        class_weight = -alpha_class * torch.pow(torch.sub(1.0, prob), self.gamma)
        loss = class_weight * logpt
        if valid_mask is not None:
            loss = loss * valid_mask.squeeze()

        if self.reduction == 'mean':
            loss = loss.mean()
            if valid_mask is not None:
                loss = loss.sum() / valid_mask.sum()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)
        return loss
# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=None, dropout=0.):
        super().__init__()
        # 这个前传过程其实就是几层全连接
        if out_dim is None:
            out_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, out_dim=None, dropout=0., se=0):
        super().__init__()
        # dim_head是每个头的特征维度
        # 多个头的特征是放在一起计算的
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.out_dim = out_dim
        self.dim = dim

        self.attend = nn.Softmax(dim=-1)
        # 这个就是产生QKV三组向量因此要乘以3
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if out_dim is None:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
            ) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, out_dim),
            ) if project_out else nn.Identity()

        self.to_out_dropout = nn.Dropout(dropout)

        self.se = se
        if self.se > 0:
            self.se_layer = SE(dim)

    def forward(self, x):
        # b是batch size h 是注意力头的数目 n 是图像块的数目
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.to_out(out)

        if self.se:
            out = self.se_layer(out)

        out = self.to_out_dropout(out)

        if self.out_dim is not None and self.out_dim != self.dim:
            # 这个时候需要特殊处理，提前做一个残差
            out = out + v.squeeze(1)

        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, attn_out_dim=None, ff_out_dim=None, dropout=0., se=0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.attn_out_dim = attn_out_dim
        self.dim = dim
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads,
                                       dim_head=dim_head, out_dim=attn_out_dim, dropout=dropout, se=se)),
                PreNorm(dim if not attn_out_dim else attn_out_dim,
                        FeedForward(dim if not attn_out_dim else attn_out_dim, mlp_dim, out_dim=ff_out_dim,
                                    dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            # 都是残差学习
            if self.attn_out_dim is not None and self.dim != self.attn_out_dim:
                x = attn(x)
            else:
                x = attn(x) + x
            x = ff(x) + x
        return x


class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, N, C]
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        return x


class T2TViT(nn.Module):
    def __init__(self, *,
                 image_size,
                 num_classes,
                 dim,
                 depth=None,
                 heads=None,
                 mlp_dim=None,
                 pool='cls',
                 channels=3,
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.,
                 transformer=None,
                 t2t_layers=((7, 4), (3, 2), (3, 2))):
        super().__init__()
        assert pool in {
            'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        layers = []
        layer_dim = [channels * 7 * 7, 64 * 3 * 3]
        output_image_size = image_size

        for i, (kernel_size, stride) in enumerate(t2t_layers):
            # layer_dim *= kernel_size ** 2
            is_first = i == 0
            is_last = i == (len(t2t_layers) - 1)
            output_image_size = conv_output_size(
                output_image_size, kernel_size, stride, stride // 2)
            layers.extend([
                RearrangeImage() if not is_first else nn.Identity(),
                nn.Unfold(kernel_size=kernel_size,
                          stride=stride, padding=stride // 2),
                Rearrange('b c n -> b n c'),
                Transformer(dim=layer_dim[i], heads=1, depth=1, dim_head=64,
                            mlp_dim=64, attn_out_dim=64, ff_out_dim=64,
                            dropout=dropout) if not is_last else nn.Identity(),
            ])

        layers.append(nn.Linear(layer_dim[1], dim))
        self.to_patch_embedding = nn.Sequential(*layers)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, output_image_size ** 2 + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        if not exists(transformer):
            assert all([exists(depth), exists(heads), exists(mlp_dim)]
                       ), 'depth, heads, and mlp_dim must be supplied'
            self.transformer = Transformer(
                dim, depth, heads, dim_head, mlp_dim, dropout)
        else:
            self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n + 1]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        # return self.mlp_head(x)
        return x


class DiT(nn.Module):
    def __init__(self, *,
                 image_size,
                 num_classes,
                 dim,
                 depth=None,
                 heads=None,
                 mlp_dim=None,
                 pool='cls',
                 channels=3,
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.,
                 patch_emb='share',
                 time_emb=False,
                 pos_emb='share',  # share or isolated # 共享或独立或不加
                 use_scale=False,
                 transformer=None,
                 t2t_layers=((7, 4), (3, 2), (3, 2)),

                 ):
        super().__init__()
        assert pool in {
            'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        layers = []
        # layer_dim = channels
        layer_dim = [channels * 7 * 7, 64 * 3 * 3]
        output_image_size = image_size

        for i, (kernel_size, stride) in enumerate(t2t_layers):
            # layer_dim *= kernel_size ** 2
            is_first = i == 0
            is_last = i == (len(t2t_layers) - 1)
            output_image_size = conv_output_size(
                output_image_size, kernel_size, stride, stride // 2)
            layers.extend([
                RearrangeImage() if not is_first else nn.Identity(),
                nn.Unfold(kernel_size=kernel_size,
                          stride=stride, padding=stride // 2),
                Rearrange('b c n -> b n c'),
                Transformer(dim=layer_dim[i], heads=1, depth=1, dim_head=64,
                            mlp_dim=64, attn_out_dim=64, ff_out_dim=64,
                            dropout=dropout) if not is_last else nn.Identity(),
            ])

        layers.append(nn.Linear(layer_dim[1], dim))

        self.patch_emb = patch_emb
        if self.patch_emb == 'share':
            self.to_patch_embedding = nn.Sequential(*layers)  # 共用
        elif self.patch_emb == 'isolated':
            layers_before = copy.deepcopy(layers)
            layers_after = copy.deepcopy(layers)
            self.to_patch_embedding_before = nn.Sequential(*layers_before)  # 不共用
            self.to_patch_embedding_after = nn.Sequential(*layers_after)

        self.pos_emb = pos_emb
        if self.pos_emb == 'share':
            self.pos_embedding_before_and_after = nn.Parameter(
                torch.randn(1, output_image_size ** 2, dim))
        elif self.pos_emb == 'sin':
            self.pos_embedding_before_and_after = self.get_sinusoid_encoding(output_image_size ** 2, dim)
        elif self.pos_emb == 'isolated':
            self.pos_embedding_before = nn.Parameter(
                torch.randn(1, output_image_size ** 2, dim))
            self.pos_embedding_after = nn.Parameter(
                torch.randn(1, output_image_size ** 2, dim))

        self.time_emb = time_emb
        if self.time_emb:
            self.time_embedding = nn.Parameter(torch.randn(2, dim))

        self.use_scale = use_scale
        if self.use_scale:
            # self.scale = nn.Parameter(torch.randn(1, 2)) # 当前最佳
            self.scale = nn.Sequential(
                nn.Linear(2 * output_image_size ** 2, 2)
            )
            self.softmax = nn.Softmax(dim=1)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        if not exists(transformer):
            assert all([exists(depth), exists(heads), exists(mlp_dim)]
                       ), 'depth, heads, and mlp_dim must be supplied'
            self.transformer = Transformer(
                dim, depth, heads, dim_head, mlp_dim, dropout)
        else:
            self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
            
        
        )


    def get_sinusoid_encoding(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0).cuda()

    def forward(self, before_x, after_x):

        if self.patch_emb == 'share':
            before_x = self.to_patch_embedding(before_x)
            after_x = self.to_patch_embedding(after_x)
        elif self.patch_emb == 'isolated':
            before_x = self.to_patch_embedding_before(before_x)
            after_x = self.to_patch_embedding_after(after_x)

        b, n, _ = before_x.shape

        # 把cls token弄进去
        if self.pos_emb == 'share' or self.pos_emb == 'sin':
            before_x += self.pos_embedding_before_and_after
            after_x += self.pos_embedding_before_and_after
        elif self.pos_emb == 'isolated':
            before_x += self.pos_embedding_before
            after_x += self.pos_embedding_after

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        if not self.use_scale:
            x = torch.cat((cls_tokens, before_x, after_x), dim=1)
        else:
            x = torch.cat((before_x, after_x), dim=1)

        if self.time_emb:
            if not self.use_scale:
                x[:, 1:(n + 1)] += self.time_embedding[0]
                x[:, (n + 1):] += self.time_embedding[1]
            else:
                x[:, :n] += self.time_embedding[0]
                x[:, n:] += self.time_embedding[1]

        x = self.dropout(x)

        x = self.transformer(x)

        if not self.use_scale:
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        else:
            # scale = self.softmax(self.scale)  # 之前最佳
            # x = scale[0, 1] * x[:, n:, :].mean(dim=1) + scale[0, 0] * x[:, :n, :].mean(dim=1) # 之前最佳
            scale = self.scale(x.mean(dim=-1))
            scale = self.softmax(scale)
            # scale = scale.view(scale.shape[0], scale.shape[1], 1)
            x = scale[0, 1] * x[:, n:, :].mean(dim=1) + scale[0, 0] * x[:, :n, :].mean(dim=1)

        x = self.to_latent(x)
            
        # return self.mlp_head(x)
        return x


class ViT_v2(nn.Module):
    def __init__(self,
                 image_size,
                 patch_size,
                 num_classes,
                 dim=None,
                 depth=None,
                 heads=None,
                 mlp_dim=None,
                 pool='cls',
                 channels=3,
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.,
                 time_emb=False,
                 pos_emb='share',
                 use_scale=False,
                 transformer=None):
        super().__init__()
        # 图像的长宽和每个Patch的长宽
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # 会有多少个patch
        num_patches = (image_height // patch_height) * \
                      (image_width // patch_width)
        # 图像的维数
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # 编码每一个Patch的信息
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_emb = pos_emb

        if self.pos_emb == 'share':
            self.pos_embedding_before_and_after = nn.Parameter(
                torch.randn(1, num_patches, dim))
        elif self.pos_emb == 'isolated':
            self.pos_embedding_before = nn.Parameter(
                torch.randn(1, num_patches, dim))
            self.pos_embedding_after = nn.Parameter(
                torch.randn(1, num_patches, dim))

        self.time_emb = time_emb
        if self.time_emb:
            self.time_embedding = nn.Parameter(torch.randn(2, dim))

        self.use_scale = use_scale
        if self.use_scale:
            # self.scale = nn.Parameter(torch.randn(1, 2)) # 当前最佳
            self.scale = nn.Sequential(
                nn.Linear(2 * num_patches, 2)
            )
            self.softmax = nn.Softmax(dim=1)

        # 类别token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        if not exists(transformer):
            assert all([exists(depth), exists(heads), exists(mlp_dim)]
                       ), 'depth, heads, and mlp_dim must be supplied'
            self.transformer = Transformer(
                dim, depth, heads, dim_head, mlp_dim, dropout)
        else:
            self.transformer = transformer

        # self.transformer = Transformer(
        #     dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # 最后的层我们自己融合
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )

    def forward(self, img_before, img_after):
        before_x = self.to_patch_embedding(img_before)
        after_x = self.to_patch_embedding(img_after)

        b, n, _ = before_x.shape

        # 把cls token弄进去
        if self.pos_emb == 'share':
            before_x += self.pos_embedding_before_and_after
            after_x += self.pos_embedding_before_and_after
        elif self.pos_emb == 'isolated':
            before_x += self.pos_embedding_before
            after_x += self.pos_embedding_after

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        if not self.use_scale:
            x = torch.cat((cls_tokens, before_x, after_x), dim=1)
        else:
            x = torch.cat((before_x, after_x), dim=1)

        if self.time_emb:
            if not self.use_scale:
                x[:, 1:(n + 1)] += self.time_embedding[0]
                x[:, (n + 1):] += self.time_embedding[1]
            else:
                x[:, :n] += self.time_embedding[0]
                x[:, n:] += self.time_embedding[1]

        # dropout操作
        x = self.dropout(x)
        # 开始transformer
        x = self.transformer(x)

        # 如果是mean模式，则对图像块所有的输出作为平均从而进行下一步分类
        # 如果是cls，则用token的输出作为特征来进行下一步的分类
        if not self.use_scale:
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        else:
            # scale = self.softmax(self.scale)  # 之前最佳
            # x = scale[0, 1] * x[:, n:, :].mean(dim=1) + scale[0, 0] * x[:, :n, :].mean(dim=1) # 之前最佳
            scale = self.scale(x.mean(dim=-1))
            scale = self.softmax(scale)
            # scale = scale.view(scale.shape[0], scale.shape[1], 1)
            x = scale[0, 1] * x[:, n:, :].mean(dim=1) + scale[0, 0] * x[:, :n, :].mean(dim=1)

        x = self.to_latent(x)

        # return self.mlp_head(x)
        return x


class DiT_basic(nn.Module):
    def __init__(self,
                 basic_model='t2t',  # 使用vit还是t2t vit只支持both 因为其他的不需要消融探究
                 patch_emb='isolated',
                 time_emb=True,
                 pos_emb='share',
                 use_scale=False,
                 pool='cls',
                 loss_f='focal',  # focal of ce
                 input_type='both',
                 label_type='MP',
                 output_type='probability',
                 dataset='US',
                 pretrain=False,
                 ):
        super(DiT_basic, self).__init__()
        print('basic_model: %s\n'
              'input_type: %s\n'
              'label_type: %s\n'
              'loss_f: %s\n'
              'patch_emb: %s\n'
              'time_emb: %d\n'
              'use_scale: %d\n'
              'pos_emb: %s\n'
              'pool: %s\n'
              % (
                  basic_model, input_type, label_type, loss_f, patch_emb, time_emb, use_scale, pos_emb, pool))

        self.input_type = input_type
        self.label_type = label_type
        self.output_type = output_type
        self.dataset=dataset
        self.feature_net = nn.Sequential()
        self.pretrain=pretrain
        if self.dataset=='US': 
            self.channelnum=3
            self.isize=128
        elif self.dataset=='DOT': 
            self.channelnum=7
            self.isize=33

        # baseline模型是普通resnet18不做任何修改
        
        if basic_model == 't2t':
            if input_type == 'both':
                self.net = DiT(image_size=self.isize,
                               num_classes=2,
                               dim=256,
                               depth=None,
                               heads=None,
                               mlp_dim=None,
                               pool=pool,
                               channels=self.channelnum, # for DOT it is 7 for US it is 3
                               dim_head=64,
                               dropout=0.4,
                               emb_dropout=0.,
                               patch_emb=patch_emb,
                               pos_emb=pos_emb,
                               time_emb=time_emb,
                               use_scale=use_scale,
                               transformer=Transformer(dim=256,
                                                       depth=16,
                                                       heads=16,
                                                       dim_head=64,
                                                       mlp_dim=512),
                               t2t_layers=((7, 4), (3, 2), (3, 2)),
                               
                               )
            else:
                self.net = T2TViT(image_size=224,
                                  num_classes=1,
                                  dim=256,
                                  pool=pool,
                                  channels=3,
                                  dim_head=64,
                                  dropout=0.,
                                  emb_dropout=0.,
                                  transformer=Transformer(dim=256,
                                                          depth=16,
                                                          heads=16,
                                                          dim_head=64,
                                                          mlp_dim=512),
                                  t2t_layers=((7, 4), (3, 2), (3, 2)))
        elif basic_model == 'vit':
            self.net = ViT_v2(image_size=224,
                              patch_size=16,
                              num_classes=2,
                              dim=256,
                              pool=pool,
                              channels=3, 
                              dim_head=64,
                              dropout=0.,
                              emb_dropout=0.,
                              time_emb=time_emb,
                              pos_emb=pos_emb,
                              use_scale=use_scale,
                              transformer=Transformer(dim=256,
                                                      depth=16,
                                                      heads=16,
                                                      dim_head=64,
                                                      mlp_dim=512))

        if self.input_type == 'both':
            self.fc = nn.Sequential(nn.LayerNorm(256),
                                    nn.Linear(256, 2))
        else:
            self.fc = nn.Sequential(nn.LayerNorm(256),
                                    nn.Linear(256, 2))

        # loss函数和softmax
        if label_type == 'MP':
            weight = torch.tensor([1.0, 1.0])
        elif label_type == 'LNM':
            weight = torch.tensor([1.0, 1.0])

        if loss_f == 'ce':
            self.loss = nn.CrossEntropyLoss(weight=weight)
        elif loss_f == 'focal':
            self.loss = FocalLoss(class_num=2, alpha=weight, gamma=2)
        self.softmax = nn.Softmax(dim=1)

        self._initialize_weights()

    def _initialize_weights(self):
        print("initialize weights for network!")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                # nn.init.xavier_normal_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, before_x, after_x, labels=None):
        before_x = self.feature_net(before_x)
        after_x = self.feature_net(after_x)

        if self.input_type == 'before':
            before_out = self.net(before_x)
            out = before_out
        elif self.input_type == 'after':
            after_out = self.net(after_x)
            out = after_out
        elif self.input_type == 'both':
            # 两个都用before_net精度还不错
            out = self.net(before_x, after_x)

        out = out.view(out.shape[0], -1)
        
        out = self.fc(out)
        if self.pretrain==True:
            return out
        if labels is not None:  # training or validation process
            # output loss and acc
            labels = labels.view(-1)
            #labels_onehot = torch.nn.functional.one_hot(labels,2).type('torch.FloatTensor').to(device)
            prob = self.softmax(out)
            # cls_loss = self.loss(out, labels_MP.float())
            cls_loss = self.loss(out, labels)
            #cls_loss = self.loss(out, labels_onehot)

            _, predicted = torch.max(prob.data, 1)
            # predicted = prob.data > 0.5
            acc = (predicted == labels).sum().item() / out.size(0)


            return cls_loss, acc
        else:  # test process
            # output probability of each class
            if self.output_type == 'probability':
                out = self.softmax(out)
                return out[:, 0]
            elif self.output_type == 'score':
                out = out
                return out

            return None


if __name__ == '__main__':
    
    
    net = DiT_basic(basic_model='t2t',  # 使用vit还是t2t vit只支持both 因为其他的不需要消融探究
                    patch_emb='isolated',
                    time_emb=True,
                    pos_emb='share',
                    use_scale=True,
                    loss_f='ce',  # focal of ce
                    input_type='both',
                    pool='mean',
                    output_type='probability',
                    dataset='DOT',
                    )
    # net = ViT_basic(basic_model='t2t')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    net.to(device)
    img_size=33
    
    train_dataset = pCRDataset(datatype='DOT',
                                info_file='./1Patient_US_train.ods',
                                           root_dir='./Dataset/',
                                           cyc_num='1',
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               ToDoubleTensor(),
                                               transforms.RandomHorizontalFlip(p=0.5),
                                               transforms.RandomRotation(degrees=(-45,45)),
                                               transforms.Resize([img_size,img_size]),
                                           ]))
    train_loader = DataLoader(train_dataset, batch_size=48,
                        shuffle=True, num_workers=0)
    
    test_dataset = pCRDataset(datatype='DOT',
                                info_file='./1Patient_US_test.ods',
                                           root_dir='./Dataset/',
                                           cyc_num='1',
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               ToDoubleTensor(),
                                               transforms.Resize([img_size,img_size]),
                                           ]))
    test_loader = DataLoader(test_dataset, batch_size=48,
                        shuffle=False, num_workers=0)
    test_loader1 = DataLoader(test_dataset, batch_size=48,
                        shuffle=True, num_workers=0)
    
    Loss, Test_Acc_All = [],[]
    label_p,prob_p,box_prob=[],[],[]
    
    num_epochs = 15
    learning_rate = 6e-6
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=50,threshold=0.004, threshold_mode='abs',verbose =True)

    best_pred = []
    best_acc = 0
    
    for epoch in range(num_epochs):
        for i, i_batch  in enumerate(train_loader):
            net.train()
            image_batch = [i.to(device) for i in i_batch['image']]
            label_batch = i_batch['labels'].to(device)
    
            optimizer.zero_grad()
            loss,acc = net(image_batch[0], image_batch[1],labels=label_batch)
            
            Loss.append(loss.item())
            loss.backward()
            
            optimizer.step()  
            
            
            '''
            # test loss
            with torch.no_grad():
                net.eval()
                # test data
                test_batch = next(iter(test_loader1))
                test_image = [i.to(device) for i in test_batch['image']]
                test_label = test_batch['labels'].to(device)
                test_loss,test_acc = net(test_image[0], test_image[1],labels=test_label)            
                
                if (i+1)%20==0:
                    print('Epoch: {}. Batch: {}. Loss: {}. Accuracy: {}. Test_loss: {}. Test_acc: {}.'
                          .format(epoch, i+1, loss.item(), acc, test_loss.item(), test_acc))
            '''
                
            if (i+1)%20==0:
                scheduler.step(loss)
                print('Epoch: {}. Batch: {}. Loss: {}. Accuracy: {}.'
                      .format(epoch, i+1, loss.item(), acc))
            
        Prob, Predict, Test_label, Acc_test = [],[],[],[]
        with torch.no_grad():
            for i1, i_batch1 in enumerate(test_loader):
                net.eval()
                test_batch = [i.to(device) for i in i_batch1['image']]
                test_label = i_batch1['labels'].to(device)
                prob = net(test_batch[0], test_batch[1])
                predicted = prob.data > 0.5
                
                test_acc = (predicted == test_label).sum().item() / predicted.size(0)
                
                prob=prob.tolist()
                predicted=predicted.tolist()
                test_label=test_label.tolist()
                
                Prob.extend(prob)
                Predict.extend(predicted)
                Test_label.extend(test_label)
                Acc_test.append(test_acc)
            
            
        print('Epoch: {}.  Test_acc: {}.'.format(epoch, np.mean(Acc_test)))
            
        
        # test acc for epoch
        Test_Acc_All.append(np.mean(Acc_test))
        
        
        if epoch>=5:
            if np.mean(Acc_test)>best_acc:
                best_acc=np.mean(Acc_test)
                best_pred=Prob
                print('model saved')
                save_path = '/media/whitaker-160/bigstorage/DiT/breast_dit/model/'
                save_mode_path = os.path.join(save_path, 'US.pth')
                torch.save(net, save_mode_path)
            cnts=test_dataset.us_num
            label_p+=[np.mean(Test_label[cnts[i]:cnts[i+1]]) for i in range(len(cnts)-1)]
            prob_p+=[np.mean(Prob[cnts[i]:cnts[i+1]]) for i in range(len(cnts)-1)] 
            plt.figure()
            plt.plot(Prob)
            plt.plot(Test_label)
    

    plt.figure()
    plt.plot(Test_Acc_All)
    
    fig = plt.figure()
    # calculate the fpr and tpr for all thresholds of the classification
    fpr, tpr, threshold = metrics.roc_curve(Test_label, np.array(best_pred))
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    fig = plt.figure()
    box_prob=np.reshape(prob_p,(10,9))
    bp = plt.boxplot(box_prob)
    ax = plt.gca()
    ax.set_xticklabels(['P2','P4','P15','P16','P19','P25','P30','P36','P38','P41'])
