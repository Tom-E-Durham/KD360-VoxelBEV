import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")

import utils.basic
import utils.vox

from torchvision.models.resnet import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
from nets.segnet_equi import (
    unproject_image_to_mem,
    Encoder_eff,
    Encoder_eff_ori,
    UpsamplingConcat,
    Decoder,
    SimpleLoss,
    FocalLoss,
)
from nets.gate_fusion import GatedFusion

class ChannelAdapter(nn.Module):
    """Channel adapter: converts any input channels to 3 channels using learnable 1x1 convolution"""
    def __init__(self, input_channels):
        super().__init__()
        # Learnable 1x1 convolution to project input channels to 3 channels
        self.adapter = nn.Conv2d(input_channels, 3, kernel_size=1, bias=False)
    
    def forward(self, x):
        # Use learnable projection to convert to 3 channels
        return self.adapter(x)


class Encoder_res101(nn.Module):
    def __init__(self, C, input_channels=3):
        super().__init__()
        self.C = C
        self.input_channels = input_channels
        # Add channel adapter if input is not 3 channels
        if input_channels != 3:
            self.channel_adapter = ChannelAdapter(input_channels)
        else:
            self.channel_adapter = None
        
        # Always use pretrained weights since we adapter non-3-channel inputs to 3 channels
        weights = ResNet101_Weights.DEFAULT
        resnet = resnet101(weights=weights)

        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        # Apply channel adaptation if needed
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        
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
    
def XYZ2xy_fov(X,Y,Z, vfov=(-22.5, 22.5)):
    theta = torch.arctan2(-X, Z)
    phi = torch.arctan(Y / torch.sqrt(X**2 + Z**2))

    x = (theta) / np.pi

    vfov_min, vfov_max = np.deg2rad(vfov[0]), np.deg2rad(vfov[1])
    y = (phi - vfov_min) / (vfov_max - vfov_min) * 2 - 1
    return x, -y

def build_fusion_encoder(encoder_type, feat2d_dim, input_channels=3):
    if encoder_type == "res101":
        return Encoder_res101(feat2d_dim, input_channels=input_channels)
    if encoder_type == "res50":
        if input_channels != 3:
            raise ValueError("res50 encoder in fusion expects 3-channel input.")
        return Encoder_res50(feat2d_dim)
    if encoder_type == "effb0":
        if input_channels != 3:
            raise ValueError("efficientnet encoder in fusion expects 3-channel input.")
        return Encoder_eff(feat2d_dim, version='b0')
    if encoder_type == "effb4":
        if input_channels != 3:
            raise ValueError("efficientnet encoder in fusion expects 3-channel input.")
        return Encoder_eff(feat2d_dim, version='b4')
    if encoder_type == "effb0_ori":
        if input_channels != 3:
            raise ValueError("efficientnet encoder in fusion expects 3-channel input.")
        return Encoder_eff_ori(feat2d_dim, version='b0')
    if encoder_type == "effb4_ori":
        if input_channels != 3:
            raise ValueError("efficientnet encoder in fusion expects 3-channel input.")
        return Encoder_eff_ori(feat2d_dim, version='b4')
    raise ValueError(f"Unsupported encoder type: {encoder_type}")


class Segnet_fusion(nn.Module):
    def __init__(self, Z_l, Y_l, X_l, 
                 Z_e, Y_e, X_e, 
                 vox_util_l=None,
                 vox_util_e=None,
                 use_lidar=False,
                 do_rgbcompress=True,
                 rand_flip=False,
                 latent_dim=128,
                 teacher_encoder_type="res101",
                 student_encoder_type="effb0_ori",
                 input_vfov=(-22.5, 22.5),
                 input_channels=3,
                 skip_dropout=0,
                 spatial_dropout=False, 
                 decoder_dropout=0):
        super(Segnet_fusion, self).__init__()
        assert (teacher_encoder_type in ["res101", "res50", "effb0", "effb4", "effb0_ori", "effb4_ori"])
        assert (student_encoder_type in ["res101", "res50", "effb0", "effb4", "effb0_ori", "effb4_ori"])

        self.Z_l, self.Y_l, self.X_l = Z_l, Y_l, X_l
        self.Z_e, self.Y_e, self.X_e = Z_e, Y_e, X_e
        self.use_lidar = use_lidar
        self.do_rgbcompress = do_rgbcompress   
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.input_vfov = input_vfov
        self.input_channels = input_channels
        self.teacher_encoder_type = teacher_encoder_type
        self.student_encoder_type = student_encoder_type
        
        self.vox_util_l = vox_util_l
        self.vox_util_e = vox_util_e
        self.mean = torch.as_tensor([0.2941, 0.3056, 0.3148]).reshape(1,3,1,1).float().cuda()
        self.std = torch.as_tensor([0.3561, 0.3668, 0.3749]).reshape(1,3,1,1).float().cuda()
        
        self.skip_dropout = skip_dropout
        self.spatial_dropout = spatial_dropout
        self.decoder_dropout = decoder_dropout
        
        self.feat2d_dim = feat2d_dim = latent_dim
        self.encoder_l = build_fusion_encoder(teacher_encoder_type, feat2d_dim, input_channels=input_channels)
        self.encoder_e = build_fusion_encoder(student_encoder_type, feat2d_dim, input_channels=3)
        self.gatefusion = GatedFusion(feat2d_dim) 

        if self.do_rgbcompress:
            self.bev_compressor_l = nn.Sequential(
                nn.Conv2d(feat2d_dim*Y_l, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(latent_dim),
                nn.GELU(),
            )
            self.bev_compressor_e = nn.Sequential(
                nn.Conv2d(feat2d_dim*Y_e, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(latent_dim),
                nn.GELU(),
            )
        else:
            pass

        
        self.decoder = Decoder(
            in_channels=latent_dim,
            n_classes=1,
            predict_future_flow=False,
            dropout_rate=skip_dropout,      # Skip connection dropout
            spatial_dropout=spatial_dropout, # Spatial dropout
            decoder_dropout=decoder_dropout  # Decoder feature dropout
        ) 

        self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            
        if vox_util_e is not None:
            self.xyz_mem = utils.basic.gridcloud3d(1, Z_e, Y_e, X_e, norm=False)
            self.xyz_fix = vox_util_e.Mem2Ref(self.xyz_mem, Z_e, Y_e, X_e, assert_cube=False)
        else:
            self.xyz_camA = None

        
    def prepare_voxelise_pcd(self, pcd):
        device = pcd.device

        # --- Coordinate Reordering ---
        point_cloud_transformed = torch.empty_like(pcd, device=device)
        point_cloud_transformed[... , 0] = pcd[... , 1]  # new x (left)
        point_cloud_transformed[... , 1] = pcd[... , 2]  # new y (up)
        point_cloud_transformed[... , 2] = pcd[... , 0]  # new z (forward)

        # --- Shift by the grid center ---
        # Subtract the grid_center so that our coordinates are relative to it.
        grid_center = self.vox_util_l['grid_center'].to(device)
        point_cloud_transformed = point_cloud_transformed - grid_center.unsqueeze(0).unsqueeze(0)  # broadcast subtraction
        
        # Filter Points Within the Grid Boundaries
        bounds = self.vox_util_l['bounds']
        XMIN_shifted, XMAX_shifted = [torch.tensor(b, device=device) for b in bounds[0]]
        YMIN_shifted, YMAX_shifted = [torch.tensor(b, device=device) for b in bounds[1]]
        ZMIN_shifted, ZMAX_shifted = [torch.tensor(b, device=device) for b in bounds[2]]
        
        B, N_points, _ = point_cloud_transformed.shape
        # Flatten from [B, N_points, 3] to [B*N_points, 3]
        pts_flat = point_cloud_transformed.reshape(-1, 3)  # shape: [B * N_points, 3]
        # -------------------------------------
        # Create a Validity Mask for Points within Boundaries
        # -------------------------------------
        mask_flat = (
            (pts_flat[:, 0] >= XMIN_shifted) & (pts_flat[:, 0] <= XMAX_shifted) &
            (pts_flat[:, 1] >= YMIN_shifted) & (pts_flat[:, 1] <= YMAX_shifted) &
            (pts_flat[:, 2] >= ZMIN_shifted) & (pts_flat[:, 2] <= ZMAX_shifted)
        )  # shape: [B * N_points]

        # -------------------------------------
        # Compute Voxel Indices for All Points
        # -------------------------------------
        voxel_x, voxel_y, voxel_z = self.vox_util_l['voxel_size']
        res_x, res_y, res_z = self.vox_util_l['res']
        indices_x_all = ((pts_flat[:, 0] - XMIN_shifted) / voxel_x).floor().long()
        indices_y_all = ((pts_flat[:, 1] - YMIN_shifted) / voxel_y).floor().long()
        indices_z_all = ((pts_flat[:, 2] - ZMIN_shifted) / voxel_z).floor().long()

        # Clamp indices to valid ranges (this is a safety check)
        indices_x_all = indices_x_all.clamp(0, res_x - 1)
        indices_y_all = indices_y_all.clamp(0, res_y - 1)
        indices_z_all = indices_z_all.clamp(0, res_z - 1)

        # Now select only valid points (those within the boundaries)
        indices_x_valid = indices_x_all[mask_flat]
        indices_y_valid = indices_y_all[mask_flat]
        indices_z_valid = indices_z_all[mask_flat]

        # -------------------------------------
        # Compute Corresponding Batch Indices
        # -------------------------------------
        # Create a tensor of batch indices repeated for each point.
        batch_indices = torch.arange(B, device=point_cloud_transformed.device).unsqueeze(1).expand(B, N_points)
        batch_indices_flat = batch_indices.reshape(-1)
        batch_indices_valid = batch_indices_flat[mask_flat]

        # -------------------------------------
        # Create the Occupancy Grid for All Batches at Once
        # -------------------------------------
        # The occupancy grid has shape [B, res_z, res_y, res_x]
        occupancy_batch = torch.zeros(B, res_z, res_y, res_x, dtype=torch.bool, device=point_cloud_transformed.device)

        # Use advanced indexing to mark occupied voxels:
        occupancy_batch[batch_indices_valid, indices_z_valid, indices_y_valid, indices_x_valid] = True

        index_matrix = occupancy_batch.unsqueeze(-1)
        
        # Build the Dense Voxel Grid (Voxel Center Coordinates)
        # Here, we generate the centers for each voxel in the shifted coordinate system.
        x_centers = torch.linspace(XMIN_shifted + voxel_x/2, XMAX_shifted - voxel_x/2, res_x, device=device)
        y_centers = torch.linspace(YMIN_shifted + voxel_y/2, YMAX_shifted - voxel_y/2, res_y, device=device)
        z_centers = torch.linspace(ZMIN_shifted + voxel_z/2, ZMAX_shifted - voxel_z/2, res_z, device=device)

        # Create a meshgrid.
        # We want the grid arranged so that the first dimension corresponds to depth (z),
        # the second to height (y), and the third to width (x). This matches the ordering for grid_sample.
        zz, yy, xx = torch.meshgrid(z_centers, y_centers, x_centers, indexing='ij')
        # Stack in (x, y, z) order for grid_sample.
        dense_grid = torch.stack((zz, yy, xx), dim=-1)  # shape: [res_z, res_y, res_x, 3]

        return index_matrix, dense_grid
    
    def sparse_grid_sample_fov(self, feat, index_matrix, dense_grid):
        B, C = feat.shape[:2]
        assert B == index_matrix.shape[0] == dense_grid.shape[0]
        dense_grid_zyx = dense_grid.reshape(B,-1, 3)
        #dense_grid_xyz = rotate_y_axis(dense_grid_xyz, angle = torch.tensor(np.pi / 2))
        Z_w,Y_w,X_w = dense_grid_zyx[:, :, 0], dense_grid_zyx[:, :, 1], dense_grid_zyx[:, :, 2]
        x, y = XYZ2xy_fov(X_w,Y_w,Z_w, vfov=self.input_vfov)
        z = torch.zeros_like(x)

        grid_for_sample = torch.stack([x, y, z], axis=2)
        grid_for_sample = torch.reshape(grid_for_sample, [B, self.X_l, self.Y_l, self.Z_l, 3])
        sampled_features = F.grid_sample(feat.unsqueeze(2), grid_for_sample, mode='bilinear',
                                 padding_mode='zeros', align_corners=False)
        
        filtered_sampled_features = sampled_features * index_matrix.permute(0,4,1,2,3)

        return filtered_sampled_features.flip(dims=[2, 4])
    
    def forward(self, lidar_imgs, equi_imgs, pcds, rad_occ_mem0=None):
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

        B, C, H, W = lidar_imgs.shape

        # rgb encoder
        device = lidar_imgs.device
        imgs = (lidar_imgs + 0.5 - self.mean.to(device)) / self.std.to(device)
        device = equi_imgs.device
        equi_imgs = (equi_imgs + 0.5 - self.mean.to(device)) / self.std.to(device)
        if self.rand_flip:
            B0, _, _, _ = imgs.shape
            self.rgb_flip_index = np.random.choice([0,1], B0).astype(bool)
            imgs[self.rgb_flip_index] = torch.flip(imgs[self.rgb_flip_index], [-1])
            equi_imgs[self.rgb_flip_index] = torch.flip(equi_imgs[self.rgb_flip_index], [-1])
        
        
        feat_imgs_l = self.encoder_l(imgs)
        feat_imgs_e = self.encoder_e(equi_imgs)
        if self.rand_flip:
            feat_imgs_l[self.rgb_flip_index] = torch.flip(feat_imgs_l[self.rgb_flip_index], [-1])
            feat_imgs_e[self.rgb_flip_index] = torch.flip(feat_imgs_e[self.rgb_flip_index], [-1])


        Z_l, Y_l, X_l = self.Z_l, self.Y_l, self.X_l

        index_matrix, dense_grid = self.prepare_voxelise_pcd(pcds)
        index_matrix = index_matrix.to(device)
        dense_grid = dense_grid.to(device).repeat(B,1,1,1,1)
        feat_mem_l = self.sparse_grid_sample_fov(feat_imgs_l, 
                           index_matrix, 
                           dense_grid)

        
        Z_e, Y_e, X_e = self.Z_e, self.Y_e, self.X_e
        xyz_fix = self.xyz_fix.to(feat_imgs_e.device).repeat(B,1,1)
        feat_mem_e = unproject_image_to_mem(feat_imgs_e, xyz_fix, Z_e, Y_e, X_e)
        
        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0,1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0,1], B).astype(bool)
            feat_mem_l[self.bev_flip1_index] = torch.flip(feat_mem_l[self.bev_flip1_index], [-1])
            feat_mem_l[self.bev_flip2_index] = torch.flip(feat_mem_l[self.bev_flip2_index], [-3])
            
            feat_mem_e[self.bev_flip1_index] = torch.flip(feat_mem_e[self.bev_flip1_index], [-1])
            feat_mem_e[self.bev_flip2_index] = torch.flip(feat_mem_e[self.bev_flip2_index], [-3])

            if rad_occ_mem0 is not None:
                rad_occ_mem0[self.bev_flip1_index] = torch.flip(rad_occ_mem0[self.bev_flip1_index], [-1])
                rad_occ_mem0[self.bev_flip2_index] = torch.flip(rad_occ_mem0[self.bev_flip2_index], [-3])

        if self.do_rgbcompress:
            feat_bev_l_ = feat_mem_l.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y_l, Z_l, X_l)
            feat_bev_l = self.bev_compressor_l(feat_bev_l_)
            feat_bev_e_ = feat_mem_e.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y_e, Z_e, X_e)
            feat_bev_e = self.bev_compressor_e(feat_bev_e_)
        else:
            feat_bev_l = torch.sum(feat_mem_l, dim=3)
            feat_bev_e = torch.sum(feat_mem_e, dim=3)

        feature_fused = self.gatefusion(feat_bev_l, feat_bev_e)  

        out_teacher = self.decoder(feature_fused, (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)
        
        return {
            'teacher': {
                **out_teacher,
                'lidar_bev': feat_bev_l,
                'feat_fused': feature_fused,         
            }
        }