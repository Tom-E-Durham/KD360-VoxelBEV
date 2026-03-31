import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")
from efficientnet_pytorch import EfficientNet
import utils.basic
import utils.vox

from torchvision.models.resnet import resnet18, resnet50, ResNet50_Weights, resnet101, ResNet101_Weights


class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        if x_to_upsample.shape[2:] != x.shape[2:]:
            x_to_upsample = F.interpolate(
                x_to_upsample,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False,
            )
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)

class UpsamplingAdd(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor=2,
        dropout_rate=0.0,
        spatial_dropout=False,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.spatial_dropout = spatial_dropout
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
        )
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout2d(p=self.dropout_rate) if self.spatial_dropout else nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = None

    def forward(self, x, x_skip):
        x = self.upsample_layer(x)
        if x.shape[2:] != x_skip.shape[2:]:
            x = F.interpolate(
                x,
                size=x_skip.shape[2:],
                mode='bilinear',
                align_corners=False,
            )
        if self.dropout is not None and self.training:
            x_skip = self.dropout(x_skip)
        return x + x_skip


class Encoder_res101(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        weights = ResNet101_Weights.DEFAULT
        resnet = resnet101(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)
        return x

class Encoder_res50(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x
    
class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        n_classes,
        predict_future_flow,
        dropout_rate=0.0,
        spatial_dropout=False,
        decoder_dropout=0.0,
    ):
        super().__init__()
        backbone = resnet18(weights=None, zero_init_residual=True)
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.predict_future_flow = predict_future_flow
        self.decoder_dropout = decoder_dropout

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingAdd(
            256,
            128,
            scale_factor=2,
            dropout_rate=dropout_rate,
            spatial_dropout=spatial_dropout,
        )
        self.up2_skip = UpsamplingAdd(
            128,
            64,
            scale_factor=2,
            dropout_rate=dropout_rate,
            spatial_dropout=spatial_dropout,
        )
        self.up1_skip = UpsamplingAdd(
            64,
            shared_out_channels,
            scale_factor=2,
            dropout_rate=dropout_rate,
            spatial_dropout=spatial_dropout,
        )
        self.feature_dropout = nn.Dropout2d(p=decoder_dropout) if decoder_dropout > 0 else None

        self.feat_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=1, padding=0),
        )
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, n_classes, kernel_size=1, padding=0),
        )
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

        if self.predict_future_flow:
            self.instance_future_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
            )

    def forward(self, x, bev_flip_indices=None):
        b, c, h, w = x.shape
        skip_x = {'1': x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        skip_x['2'] = x
        x = self.layer2(x)
        skip_x['3'] = x
        x = self.layer3(x)
        encoder_feature = x
        
        multi_scale = {
            'b1': skip_x['2'],         # output of layer2 (≈ student B1)
            'b2': skip_x['3'],     # output of layer3 (≈ student B2)
            'h': encoder_feature      # final BEV before decoding (≈ student H)
        }

        x = self.up3_skip(x, skip_x['3'])
        x = self.up2_skip(x, skip_x['2'])
        x = self.up1_skip(x, skip_x['1'])

        if bev_flip_indices is not None:
            bev_flip1_index, bev_flip2_index = bev_flip_indices
            x[bev_flip2_index] = torch.flip(x[bev_flip2_index], [-2]) # note [-2] instead of [-3], since Y is gone now
            x[bev_flip1_index] = torch.flip(x[bev_flip1_index], [-1])
        if self.feature_dropout is not None and self.training:
            x = self.feature_dropout(x)
        feat_output = self.feat_head(x)
        segmentation_output = self.segmentation_head(x)
        instance_center_output = self.instance_center_head(x)
        instance_offset_output = self.instance_offset_head(x)
        instance_future_output = self.instance_future_head(x) if self.predict_future_flow else None

        return {
            'raw_feat': x,
            'encoder_feature': encoder_feature,
            'feat': feat_output.view(b, *feat_output.shape[1:]),
            'segmentation': segmentation_output.view(b, *segmentation_output.shape[1:]),
            'instance_center': instance_center_output.view(b, *instance_center_output.shape[1:]),
            'instance_offset': instance_offset_output.view(b, *instance_offset_output.shape[1:]),
            'instance_flow': instance_future_output.view(b, *instance_future_output.shape[1:])
            if instance_future_output is not None else None,
            'multi_scale': multi_scale
        }
        
def XYZ2xy(X,Y,Z):
    theta = torch.arctan2(Z, X)
    phi = torch.arctan(Y / torch.sqrt(X**2 + Z**2))

    x = theta / np.pi
    y = phi * 2 / np.pi
    return x, y
def rotate_y_axis(xyz_fix, angle):
    # Equivalent to xyz @ Ry(angle), but avoids constructing a matrix every call.
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    x = xyz_fix[..., 0]
    y = xyz_fix[..., 1]
    z = xyz_fix[..., 2]
    x_new = x * cos_angle - z * sin_angle
    z_new = x * sin_angle + z * cos_angle
    return torch.stack([x_new, y, z_new], dim=-1)

def unproject_image_to_mem(feat_img, xyz_fix, Z, Y, X):
    B, N, _ = xyz_fix.shape

    xyz_fix_rotated = rotate_y_axis(xyz_fix, angle = torch.tensor(np.pi / 2))

    X_w = xyz_fix_rotated[:,:,0]
    Y_w = xyz_fix_rotated[:,:,1]
    Z_w = xyz_fix_rotated[:,:,2]
    
    x, y = XYZ2xy(X_w, Y_w, Z_w)
    z = torch.zeros_like(x)
    xyz_fix = torch.stack([x, y, z], axis=2)

    xyz_fix = torch.reshape(xyz_fix, [B, Z, Y, X, 3])
    feat_mem = nn.functional.grid_sample(feat_img.unsqueeze(2), xyz_fix, 
                                   padding_mode = 'zeros', 
                                   align_corners=False)
    return feat_mem

class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]), reduction='none')

    def forward(self, ypred, ytgt, valid = None):
        loss = self.loss_fn(ypred, ytgt)
        if valid is None:
            loss = utils.basic.reduce_masked_mean(loss, torch.ones_like(loss))
        else:
            loss = utils.basic.reduce_masked_mean(loss, valid)
        return loss

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', trainable=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.trainable = trainable
        if trainable:
            self.gamma_param = nn.Parameter(torch.tensor(0, dtype=torch.float32))
        else:
            self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        if self.trainable:
            gamma = F.softplus(self.gamma_param)
        else:
            gamma = self.gamma

        pt = torch.exp(-BCE_loss)

        F_loss = self.alpha * (1-pt)**gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
class Encoder_eff_ori(nn.Module):
    def __init__(self, C, version='b0'):
        super().__init__()
        assert version in ['b0', 'b4'], "Only b0 and b4 are supported."
        self.backbone = EfficientNet.from_pretrained(f'efficientnet-{version}')
        self.channel_proj = nn.Conv2d(1280, C, kernel_size=1)
    def forward(self, x):
        features = self.backbone.extract_features(x)
        return self.channel_proj(features)    

class Encoder_eff(nn.Module):
    def __init__(self, C, version='b4'):
        super().__init__()
        self.C = C
        self.downsample = 8
        self.version = version

        if self.version == 'b0':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        elif self.version == 'b4':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.delete_unused_layers()

        if self.downsample == 16:
            if self.version == 'b0':
                upsampling_in_channels = 320 + 112
            elif self.version == 'b4':
                upsampling_in_channels = 448 + 160
            upsampling_out_channels = 512
        elif self.downsample == 8:
            if self.version == 'b0':
                upsampling_in_channels = 112 + 40
            elif self.version == 'b4':
                upsampling_in_channels = 160 + 56
            upsampling_out_channels = 128
        else:
            raise ValueError(f'Downsample factor {self.downsample} not handled.')

        self.upsampling_layer = UpsamplingConcat(upsampling_in_channels, upsampling_out_channels)
        self.depth_layer = nn.Conv2d(upsampling_out_channels, self.C, kernel_size=1, padding=0)

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features(self, x):
        endpoints = dict()

        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        if self.downsample == 16:
            input_1, input_2 = endpoints['reduction_5'], endpoints['reduction_4']
        elif self.downsample == 8:
            input_1, input_2 = endpoints['reduction_4'], endpoints['reduction_3']
        x = self.upsampling_layer(input_1, input_2)
        return x

    def forward(self, x):
        x = self.get_features(x)  # get feature vector
        x = self.depth_layer(x)  # feature and depth head
        return x
 
class AdaptationLayer(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_scale=1):
        super().__init__()
        self.scale = spatial_scale
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if spatial_scale != 1:
            self.upsample = nn.Upsample(scale_factor=spatial_scale, mode='bilinear', align_corners=False)
        else:
            self.upsample = nn.Identity()

    def forward(self, x):
        x = self.upsample(x)
        return self.proj(x)

def build_encoder(encoder_type, feat2d_dim):
    if encoder_type == "res101":
        return Encoder_res101(feat2d_dim)
    if encoder_type == "res50":
        return Encoder_res50(feat2d_dim)
    if encoder_type == "effb0":
        return Encoder_eff(feat2d_dim, version='b0')
    if encoder_type == "effb4":
        return Encoder_eff(feat2d_dim, version='b4')
    if encoder_type == "effb0_ori":
        return Encoder_eff_ori(feat2d_dim, version='b0')
    if encoder_type == "effb4_ori":
        return Encoder_eff_ori(feat2d_dim, version='b4')
    raise ValueError(f"Unsupported encoder_type: {encoder_type}")


class Segnet(nn.Module):
    def __init__(self, Z, Y, X, vox_util=None,
                 use_lidar=False,
                 do_rgbcompress=True,
                 rand_flip=False,
                 latent_dim=128,
                 encoder_type="res101",
                 if_KL=True):
        super(Segnet, self).__init__()
        assert (encoder_type in ["res101", "res50", "effb0", "effb4", "effb0_ori", "effb4_ori"])

        self.Z, self.Y, self.X = Z, Y, X
        self.use_lidar = use_lidar
        self.do_rgbcompress = do_rgbcompress   
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type

        self.mean = torch.as_tensor([0.2941, 0.3056, 0.3148]).reshape(1,3,1,1).float().cuda()
        self.std = torch.as_tensor([0.3561, 0.3668, 0.3749]).reshape(1,3,1,1).float().cuda()
        
        self.feat2d_dim = feat2d_dim = latent_dim
        self.encoder = build_encoder(encoder_type, feat2d_dim)

        if self.use_lidar:
            self.bev_compressor = nn.Sequential(
                nn.Conv2d(feat2d_dim*Y+Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(latent_dim),
                nn.GELU(),
            )
        else:
            if self.do_rgbcompress:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
            else:
                pass

        self.decoder = Decoder(
            in_channels=latent_dim,
            n_classes=1,
            predict_future_flow=False
        )
        self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        if if_KL:
            self.kl1_weight = nn.Parameter(torch.tensor(0.0))  
            self.kl2_weight = nn.Parameter(torch.tensor(0.0))
            self.kl3_weight = nn.Parameter(torch.tensor(0.0))

        if vox_util is not None:
            self.xyz_mem = utils.basic.gridcloud3d(1, Z, Y, X, norm=False)
            self.xyz_fix = vox_util.Mem2Ref(self.xyz_mem, Z, Y, X, assert_cube=False)
        else:
            self.xyz_camA = None
        
    def forward(self, imgs, rad_occ_mem0=None):
        '''
        B = batch size, C = 3, H = img height, W = img width
        rgb_camXs: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cam0_T_camXs: (B,S,4,4)
        vox_util: vox util object
        rad_occ_mem0:
            - None when use_radar = False, use_lidar = False
            - (B, 1, Z, Y, X) when use_radar = True, use_metaradar = False
            - (B, 16, Z, Y, X) when use_radar = True, use_metaradar = True
            - (B, 1, Z, Y, X) when use_lidar = True
        '''

        B, C, H, W = imgs.shape
        assert(C==3)

        device = imgs.device
        imgs = (imgs + 0.5 - self.mean.to(device)) / self.std.to(device)
        if self.rand_flip:
            B0, _, _, _ = imgs.shape
            self.rgb_flip_index = np.random.choice([0,1], B0).astype(bool)
            imgs[self.rgb_flip_index] = torch.flip(imgs[self.rgb_flip_index], [-1])
        
        feat_imgs = self.encoder(imgs)
        if self.rand_flip:
            feat_imgs[self.rgb_flip_index] = torch.flip(feat_imgs[self.rgb_flip_index], [-1])

        _, C, Hf, Wf = feat_imgs.shape

        Z, Y, X = self.Z, self.Y, self.X
        xyz_fix = self.xyz_fix.to(feat_imgs.device).repeat(B,1,1)
        feat_mem = unproject_image_to_mem(feat_imgs, xyz_fix, Z, Y, X)

        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0,1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0,1], B).astype(bool)
            feat_mem[self.bev_flip1_index] = torch.flip(feat_mem[self.bev_flip1_index], [-1])
            feat_mem[self.bev_flip2_index] = torch.flip(feat_mem[self.bev_flip2_index], [-3])

            if rad_occ_mem0 is not None:
                rad_occ_mem0[self.bev_flip1_index] = torch.flip(rad_occ_mem0[self.bev_flip1_index], [-1])
                rad_occ_mem0[self.bev_flip2_index] = torch.flip(rad_occ_mem0[self.bev_flip2_index], [-3])

        if self.use_lidar:
            assert(rad_occ_mem0 is not None)
            feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
            rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, Y, Z, X)
            feat_bev_ = torch.cat([feat_bev_, rad_bev_], dim=1)
            feat_bev = self.bev_compressor(feat_bev_)
        else: # rgb only
            if self.do_rgbcompress:
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
                feat_bev = self.bev_compressor(feat_bev_)
            else:
                feat_bev = torch.sum(feat_mem, dim=3)

        out_dict = self.decoder(feat_bev, (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)

        raw_e = out_dict['raw_feat']
        feat_e = out_dict['feat']
        u_net_feature = out_dict['encoder_feature']
        seg_e = out_dict['segmentation']
        center_e = out_dict['instance_center']
        offset_e = out_dict['instance_offset']

        return raw_e, feat_e, seg_e, center_e, offset_e, feat_bev, u_net_feature