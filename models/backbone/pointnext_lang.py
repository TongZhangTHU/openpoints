from typing import List, Type
import logging
import torch
import torch.nn as nn
from ..build import MODELS
from ..layers import create_convblock1d, create_convblock2d, create_act, CHANNEL_MAP, \
    create_grouper, furthest_point_sample, random_sample, three_interpolation, get_aggregation_feautres
from helpers.network_utils import DenseBlock
from .pointnext import InvResMLP, get_reduction_fn

# This SetAbstraction class is slightly different from pointnext version
class SetAbstraction(nn.Module):
    """The modified set abstraction module in PointNet++ with residual connection support
    """

    def __init__(self,
                 in_channels, out_channels,
                 layers=1,
                 stride=1,
                 group_args={'NAME': 'ballquery',
                             'radius': 0.1, 'nsample': 16},
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 conv_args=None,
                 sampler='fps',
                 feature_type='dp_fj',
                 use_res=False,
                 is_head=False,
                 use_skip=False,
                 **kwargs, 
                 ):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        self.all_aggr = not is_head and stride == 1
        self.use_res = use_res and not self.all_aggr and not self.is_head
        self.feature_type = feature_type

        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_channels] + [mid_channel] * \
                   (layers - 1) + [out_channels]
        channels[0] = in_channels if is_head else CHANNEL_MAP[feature_type](channels[0])

        if self.use_res:
            self.skipconv = create_convblock1d(
                in_channels, channels[-1], norm_args=None, act_args=None) if in_channels != channels[
                -1] or use_skip else nn.Identity()
            self.act = create_act(act_args)

        # actually, one can use local aggregation layer to replace the following
        create_conv = create_convblock1d if is_head else create_convblock2d
        convs = []
        for i in range(len(channels) - 1):
            convs.append(create_conv(channels[i], channels[i + 1],
                                     norm_args=norm_args if not is_head else None,
                                     act_args=None if i == len(channels) - 2
                                                      and (self.use_res or is_head) else act_args,
                                     **conv_args)
                         )
        self.convs = nn.Sequential(*convs)
        if not is_head:
            if self.all_aggr:
                group_args.nsample = None
                group_args.radius = None
            self.grouper = create_grouper(group_args)
            #self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
            self.pool = get_reduction_fn('max')
            if sampler.lower() == 'fps':
                self.sample_fn = furthest_point_sample
            elif sampler.lower() == 'random':
                self.sample_fn = random_sample

    def forward(self, pf):
        p, f = pf
        if self.is_head:
            f = self.convs(f)  # (n, c)
        else:
            # (1) subsample
            if not self.all_aggr:
                idx = self.sample_fn(p, p.shape[1] // self.stride).long()
                new_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            else:
                new_p = p
            """ DEBUG neighbor numbers. 
            query_xyz, support_xyz = new_p, p
            radius = self.grouper.radius
            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            logging.info(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
            DEBUG end """
            if self.use_res or 'df' in self.feature_type:
                fi = torch.gather(
                    f, -1, idx.unsqueeze(1).expand(-1, f.shape[1], -1))
                if self.use_res:
                    identity = self.skipconv(fi)
            else:
                fi = None
            # (2) grouping
            dp, fj = self.grouper(new_p, p, f)
            fj = get_aggregation_feautres(new_p, dp, fi, fj, feature_type=self.feature_type)
            # (3) mlps (4) Reduction (max-pooling)
            f = self.pool(self.convs(fj))
            if self.use_res:
                f = self.act(f + identity)
            p = new_p
        return p, f



@MODELS.register_module()
class PointNextLangEncoder(nn.Module):
    r"""The Encoder for PointNext 
    `"PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies".
    <https://arxiv.org/abs/2206.04670>`_.
    .. note::
        For an example of using :obj:`PointNextEncoder`, see
        `examples/segmentation/main.py <https://github.com/guochengqian/PointNeXt/blob/master/cfgs/s3dis/README.md>`_.
    Args:
        in_channels (int, optional): input channels . Defaults to 4.
        width (int, optional): width of network, the output mlp of the stem MLP. Defaults to 32.
        blocks (List[int], optional): # of blocks per stage (including the SA block). Defaults to [1, 4, 7, 4, 4].
        strides (List[int], optional): the downsampling ratio of each stage. Defaults to [4, 4, 4, 4].
        block (strorType[InvResMLP], optional): the block to use for depth scaling. Defaults to 'InvResMLP'.
        nsample (intorList[int], optional): the number of neighbors to query for each block. Defaults to 32.
        radius (floatorList[float], optional): the initial radius. Defaults to 0.1.
        aggr_args (_type_, optional): the args for local aggregataion. Defaults to {'feature_type': 'dp_fj', "reduction": 'max'}.
        group_args (_type_, optional): the args for grouping. Defaults to {'NAME': 'ballquery'}.
        norm_args (_type_, optional): the args for normalization layer. Defaults to {'norm': 'bn'}.
        act_args (_type_, optional): the args for activation layer. Defaults to {'act': 'relu'}.
        expansion (int, optional): the expansion ratio of the InvResMLP block. Defaults to 4.
        sa_layers (int, optional): the number of MLP layers to use in the SA block. Defaults to 1.
        sa_use_res (bool, optional): wheter to use residual connection in SA block. Set to True only for PointNeXt-S. 
    """

    def __init__(self,
                 in_channels: int = 4,
                 width: int = 32,
                 blocks: List[int] = [1, 4, 7, 4, 4],
                 strides: List[int] = [4, 4, 4, 4],
                 block: str or Type[InvResMLP] = 'InvResMLP',
                 nsample: int or List[int] = 32,
                 radius: float or List[float] = 0.1,
                 aggr_args: dict = {'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args: dict = {'NAME': 'ballquery'},
                 sa_layers: int = 1,
                 sa_use_res: bool = False,
                 # language added
                 lang_index: List[int] = [],
                 lang_feat_dim: int = 1024,
                 fusion_type: str = 'mult',
                 **kwargs
                 ):
        super().__init__()
        if isinstance(block, str):
            block = eval(block)
        self.blocks = blocks
        self.strides = strides
        self.in_channels = in_channels
        self.aggr_args = aggr_args
        self.norm_args = kwargs.get('norm_args', {'norm': 'bn'}) 
        self.act_args = kwargs.get('act_args', {'act': 'relu'}) 
        self.conv_args = kwargs.get('conv_args', None)
        self.sampler = kwargs.get('sampler', 'fps')
        self.expansion = kwargs.get('expansion', 4)
        self.sa_layers = sa_layers
        self.sa_use_res = sa_use_res
        self.lang_index = lang_index
        #self.lang_feat_dim = lang_feat_dim
        self.fusion_type = fusion_type
        self.use_res = kwargs.get('use_res', True)
        radius_scaling = kwargs.get('radius_scaling', 2)
        nsample_scaling = kwargs.get('nsample_scaling', 1)

        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        logging.info(f'radius: {self.radii},\n nsample: {self.nsample}')

        # double width after downsampling.
        channels = []
        for stride in strides:
            if stride != 1:
                width *= 2
            channels.append(width)
        encoder = []
        for i in range(len(blocks)):
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            encoder.append(self._make_enc(
                block, channels[i], blocks[i], stride=strides[i], group_args=group_args,
                is_head=i == 0 and strides[i] == 1
            ))
            if self.fusion_type == 'concat' and i in self.lang_index:
                self.in_channels = self.in_channels * 2

        self.encoder = nn.Sequential(*encoder)
        self.out_channels = channels[-1]
        self.channel_list = channels

        self._lang_proj = nn.ModuleDict()
        for index in self.lang_index:
            self._lang_proj.update({str(index): DenseBlock(lang_feat_dim, channels[index], None, None)})

    def _to_full_list(self, param, param_scaling=1):
        # param can be: radius, nsample
        param_list = []
        if isinstance(param, List):
            # make param a full list
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:  # radius is a scalar (in this case, only initial raidus is provide), then create a list (radius for each block)
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append(
                        [param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def _make_enc(self, block, channels, blocks, stride, group_args, is_head=False):
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        group_args.radius = radii[0]
        group_args.nsample = nsample[0]
        layers.append(SetAbstraction(self.in_channels, channels,
                                     self.sa_layers if not is_head else 1, stride,
                                     group_args=group_args,
                                     sampler=self.sampler,
                                     norm_args=self.norm_args, act_args=self.act_args, conv_args=self.conv_args,
                                     is_head=is_head, use_res=self.sa_use_res, use_skip=(self.fusion_type == 'concat'), **self.aggr_args 
                                     ))
        self.in_channels = channels
        for i in range(1, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(block(self.in_channels,
                                aggr_args=self.aggr_args,
                                norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
                                conv_args=self.conv_args, expansion=self.expansion,
                                use_res=self.use_res
                                ))
        return nn.Sequential(*layers)

    def _proj_feature(self, x, spatial_size, proj_fn):
        x = proj_fn(x)
        x = x.unsqueeze(2)
        x = x.repeat(1, 1, spatial_size)
        return x

    def forward_cls_feat(self, p0, f0=None):
        # p0: (bs, N, 3) f0: (bs, k, N)
        if hasattr(p0, 'keys'):
            p0, f0, lang_goal_emb = p0['pos'], p0.get('x', None), p0.get('lang_goal_emb', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        for i in range(0, len(self.encoder)):
            p0, f0 = self.encoder[i]([p0, f0])
            if i in self.lang_index : 
                l = self._proj_feature(lang_goal_emb, f0.shape[-1], self._lang_proj[str(i)])
                if self.fusion_type == 'mult':
                    f0 = f0 * l
                elif self.fusion_type == 'add':
                    f0 = f0 + l
                elif self.fusion_type == 'max':
                    f0 = torch.max(f0, l)
                elif self.fusion_type == 'concat':
                    f0 = torch.cat([f0, l], dim=1)
                else:
                    raise NotImplementedError
        return f0.squeeze(-1)

