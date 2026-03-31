import torch
import torch.nn as nn
import numpy as np
import utils.basic

from nets.segnet_equi import (
    Decoder,
    SimpleLoss,
    FocalLoss,
    AdaptationLayer,
    build_encoder,
    unproject_image_to_mem,
)
from nets.gate_fusion import GatedFusion


class Segnet(nn.Module):
    def __init__(
        self,
        Z,
        Y,
        X,
        vox_util=None,
        use_lidar=False,
        do_rgbcompress=True,
        rand_flip=False,
        latent_dim=128,
        encoder_type="res101",
        if_KL=True,
    ):
        super(Segnet, self).__init__()
        assert encoder_type in ["res101", "res50", "effb0", "effb4", "effb0_ori", "effb4_ori"]

        self.Z, self.Y, self.X = Z, Y, X
        self.use_lidar = use_lidar
        self.do_rgbcompress = do_rgbcompress
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type

        self.mean = torch.as_tensor([0.2941, 0.3056, 0.3148]).reshape(1, 3, 1, 1).float().cuda()
        self.std = torch.as_tensor([0.3561, 0.3668, 0.3749]).reshape(1, 3, 1, 1).float().cuda()

        self.feat2d_dim = feat2d_dim = latent_dim
        self.encoder = build_encoder(encoder_type, feat2d_dim)

        if self.use_lidar:
            self.bev_compressor = nn.Sequential(
                nn.Conv2d(feat2d_dim * Y + Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(latent_dim),
                nn.GELU(),
            )
        else:
            if self.do_rgbcompress:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim * Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
            else:
                pass

        self.gatefusion = GatedFusion(feat2d_dim)

        self.decoder = Decoder(
            in_channels=latent_dim,
            n_classes=1,
            predict_future_flow=False,
        )
        self.decoder_ta = Decoder(
            in_channels=latent_dim,
            n_classes=1,
            predict_future_flow=False,
        )

        self.adapt_feat = AdaptationLayer(in_channels=latent_dim, out_channels=latent_dim, spatial_scale=4)
        self.adapt_b1 = AdaptationLayer(in_channels=int(latent_dim / 2), out_channels=latent_dim, spatial_scale=1)
        self.adapt_b2 = AdaptationLayer(in_channels=latent_dim, out_channels=latent_dim, spatial_scale=1)
        self.adapt_H = AdaptationLayer(in_channels=int(latent_dim * 2), out_channels=int(latent_dim * 2), spatial_scale=1)

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

    def forward(self, imgs, lidar_bev, rad_occ_mem0=None):
        B, C, H, W = imgs.shape
        assert C == 3

        device = imgs.device
        imgs = (imgs + 0.5 - self.mean.to(device)) / self.std.to(device)
        if self.rand_flip:
            self.rgb_flip_index = np.random.choice([0, 1], B).astype(bool)
            imgs[self.rgb_flip_index] = torch.flip(imgs[self.rgb_flip_index], [-1])

        feat_imgs = self.encoder(imgs)
        if self.rand_flip:
            feat_imgs[self.rgb_flip_index] = torch.flip(feat_imgs[self.rgb_flip_index], [-1])

        Z, Y, X = self.Z, self.Y, self.X
        xyz_fix = self.xyz_fix.to(feat_imgs.device).repeat(B, 1, 1)
        feat_mem = unproject_image_to_mem(feat_imgs, xyz_fix, Z, Y, X)

        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0, 1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0, 1], B).astype(bool)
            feat_mem[self.bev_flip1_index] = torch.flip(feat_mem[self.bev_flip1_index], [-1])
            feat_mem[self.bev_flip2_index] = torch.flip(feat_mem[self.bev_flip2_index], [-3])

            if rad_occ_mem0 is not None:
                rad_occ_mem0[self.bev_flip1_index] = torch.flip(rad_occ_mem0[self.bev_flip1_index], [-1])
                rad_occ_mem0[self.bev_flip2_index] = torch.flip(rad_occ_mem0[self.bev_flip2_index], [-3])

        if self.use_lidar:
            assert rad_occ_mem0 is not None
            feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim * Y, Z, X)
            rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, Y, Z, X)
            feat_bev_ = torch.cat([feat_bev_, rad_bev_], dim=1)
            feat_bev = self.bev_compressor(feat_bev_)
        else:
            if self.do_rgbcompress:
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim * Y, Z, X)
                feat_bev = self.bev_compressor(feat_bev_)
            else:
                feat_bev = torch.sum(feat_mem, dim=3)

        out_s = self.decoder(feat_bev, (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)
        feature_fused_ta = self.gatefusion(lidar_bev, feat_bev)
        out_ta = self.decoder_ta(feature_fused_ta, (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)

        return {
            "student": {
                **out_s,
                "feat_bev": feat_bev,
            },
            "ta": {
                **out_ta,
                "feat_fused": feature_fused_ta,
            },
        }
