import numpy as np
import time
import os
from pathlib import Path
from fire import Fire

from Dur360BEV_dataset import dur360bev_dataset
from nets.segnet_fusion import Segnet_fusion
import utils.vox
import utils.sw

import torch
from utils.criterion import CriterionCWD, CriterionBMSE, CriterionFL, CriterionAD
from nets.segnet_equi import Segnet as SegnetEqui
from tensorboardX import SummaryWriter

def compute_iou_car(seg_e, bev_seg_g):
    seg_e_round = torch.sigmoid(seg_e).round()
    seg_e_car = seg_e_round[:, 0]  # Car prediction
    bev_seg_g_car = bev_seg_g[:, 0]  # Car ground truth

    intersection_car = (seg_e_car * bev_seg_g_car).sum(dim=[1, 2])
    union_car = (seg_e_car + bev_seg_g_car).clamp(0, 1).sum(dim=[1, 2])

    # iou_car = (intersection_car / (1e-4 + union_car)).mean()
    return intersection_car, union_car


def fetch_optimizer_onecycle(lr, wdecay, epsilon, num_steps, params):
    """ Create the optimizer and learning rate scheduler """
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, num_steps+100,
        pct_start=0.1, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


def fetch_optimizer_cosine(lr, wdecay, epsilon, num_steps, params):
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)
    # T_max: maximum number of iterations for the scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_steps,
        eta_min=lr*0.01
    )
    return optimizer, scheduler

# Set up checkpoint saver:
def save_checkpoint(global_step, model, optimizer, loss, checkpoint_path, scheduler=None):
    checkpoint = {
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(checkpoint, checkpoint_path)


def _resolve_optional_path(path_value):
    if path_value is None:
        return None
    p = Path(path_value).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def _default_dataset_dir():
    env_dir = os.environ.get("DUR360BEV_DATASET_DIR")
    if env_dir:
        return str(Path(env_dir).expanduser())
    # Portable default: repository-local data directory.
    return str((Path(__file__).resolve().parent / "data" / "Dur360BEV_Dataset_Complete").resolve())

def fusion_imgs_generation(sample, device='cuda:0'):
    lidar_imgs = torch.cat((sample['lidar_images']['range_img'], 
                            sample['lidar_images']['ambient_img'], 
                            sample['lidar_images']['intensity_img']), 
                            dim=1).to(device)

    equi_imgs = sample['image'].to(device)
        
    return lidar_imgs, equi_imgs

def visual_img_tensorboard(sw, sample, seg_e, bev_seg_g, offset_e, bev_offset_g, center_e, bev_center_g):
    lidar_imgs, equi = fusion_imgs_generation(sample, device='cuda:0')
    if sw is not None and sw.img_save:
        print(f"[DEBUG INFO]: Saved images at step {sw.global_step}")
        lidar_imgs = lidar_imgs.float() - 0.5
        sw.rgb_img('0_inputs/lidar', lidar_imgs+0.5) 
        sw.rgb_img('0_inputs/equi', equi) 
        sw.bin_img('1_outputs/bev_seg_car_e', torch.sigmoid(seg_e[:,0]))
        sw.bin_img('1_outputs/bev_seg_car_g', bev_seg_g[:,0])
        sw.bin_img('1_outputs/bev_offset_x_e', offset_e[:,0]) 
        sw.bin_img('1_outputs/bev_offset_x_g', bev_offset_g[:,0]) 
        sw.bin_img('1_outputs/bev_offset_y_e', offset_e[:,1]) 
        sw.bin_img('1_outputs/bev_offset_y_g', bev_offset_g[:,1]) 
        sw.bin_img('1_outputs/bev_center_e', center_e[:,0])
        sw.bin_img('1_outputs/bev_center_g', bev_center_g[:,0])

def _select_stage_features(stage, feat_stage1, feat_stage2, feat_stage3):
    stage_map = {
        'stage1': (0,),
        'stage2': (1,),
        'stage3': (2,),
        'stage12': (0, 1),
        'stage13': (0, 2),
        'stage23': (1, 2),
        'stage123': (0, 1, 2),
    }
    if stage not in stage_map:
        raise ValueError(f"Unsupported stage: {stage}")
    feats = (feat_stage1, feat_stage2, feat_stage3)
    selected = tuple(feats[i] for i in stage_map[stage])
    return selected[0] if len(selected) == 1 else selected


def run_model_forward(model, sample, device='cuda:0', stage='stage1'):
    lidar_imgs, equi_imgs = fusion_imgs_generation(sample)
    pcds = sample['pcd'][:, :, :3].to(device)

    lidar_imgs = lidar_imgs.float() - 0.5
    equi_imgs = equi_imgs.float() - 0.5

    _, _, seg_logits, _, _, feature_fused, u_net_feature = model(lidar_imgs, equi_imgs, pcds)
    return _select_stage_features(stage, feature_fused, u_net_feature, seg_logits)


def run_model_S_forward(model, sample, device='cuda:0', stage='stage1'):
    _, equi_imgs = fusion_imgs_generation(sample)
    equi_imgs = equi_imgs.float() - 0.5
    bev_seg_g = sample['bev_seg'].to(device)
    bev_center_g = sample['center'].to(device)
    bev_offset_g = sample['offset'].to(device)

    _, _, seg_e, center_e, offset_e, feat_bev, u_net_feature = model(equi_imgs)
    feat_S = _select_stage_features(stage, feat_bev, u_net_feature, seg_e)
    return feat_S, seg_e, bev_seg_g, bev_center_g, bev_offset_g, center_e, offset_e

def compute_stages_loss(
    model, stage, pred_S, bev_seg_g,
    feat_S, feat_T,
    criterion_kl,
    criterion_ad,
    criterion_bmse, criterion_fl,
    center_e, offset_e, bev_center_g, bev_offset_g,
    dis_type = 'ad',
    device='cuda:0'
):
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)
    loss_fl = criterion_fl(pred_S, bev_seg_g[:, 0].unsqueeze(1))
    ce_factor = 1 / torch.exp(model.ce_weight)
    loss_fl = loss_fl * ce_factor * 10
    fl_uncertainty_loss = 0.5 * model.ce_weight

    center_loss = criterion_bmse(center_e, bev_center_g)
    offset_loss = torch.abs(offset_e - bev_offset_g).sum(dim=1, keepdim=True).mean()
    offset_loss = utils.basic.reduce_masked_mean(offset_loss, bev_seg_g.sum(dim=1, keepdim=True))

    center_factor = 1 / (2 * torch.exp(model.center_weight))
    center_loss = center_factor * center_loss
    center_uncertainty_loss = 0.5 * model.center_weight

    offset_factor = 1 / (2 * torch.exp(model.offset_weight))
    offset_loss = offset_factor * offset_loss
    offset_uncertainty_loss = 0.5 * model.offset_weight

    if stage in ('stage1', 'stage2', 'stage3'):
        pairs = [(stage, feat_S, feat_T)]
    elif stage == 'stage12':
        pairs = [('stage1', feat_S[0], feat_T[0]), ('stage2', feat_S[1], feat_T[1])]
    elif stage == 'stage13':
        pairs = [('stage1', feat_S[0], feat_T[0]), ('stage3', feat_S[1], feat_T[1])]
    elif stage == 'stage23':
        pairs = [('stage2', feat_S[0], feat_T[0]), ('stage3', feat_S[1], feat_T[1])]
    elif stage == 'stage123':
        pairs = [
            ('stage1', feat_S[0], feat_T[0]),
            ('stage2', feat_S[1], feat_T[1]),
            ('stage3', feat_S[2], feat_T[2]),
        ]
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    multi_stage = (len(pairs) > 1)
    total_kd_loss = 0.0
    total_kd_uncertainty = 0.0

    for idx, (sub_stage, feat_s, feat_t) in enumerate(pairs):
        if sub_stage == 'stage1':
            kd_loss = criterion_ad(feat_s, feat_t)
        elif sub_stage == 'stage2' or sub_stage == 'stage3':
            kd_loss = criterion_kl(feat_s, feat_t, sub_stage)

        if not multi_stage:
            total_kd_loss = total_kd_loss + kd_loss
        else:
            if idx == 0 and hasattr(model, 'kl1_weight'):
                factor = 1 / (2 * torch.exp(model.kl1_weight))
                kd_loss = factor * kd_loss
                uncertainty = 0.5 * model.kl1_weight
            elif idx == 1 and hasattr(model, 'kl2_weight'):
                factor = 1 / (2 * torch.exp(model.kl2_weight))
                kd_loss = factor * kd_loss
                uncertainty = 0.5 * model.kl2_weight
            elif idx == 2 and hasattr(model, 'kl3_weight'):
                factor = 1 / (2 * torch.exp(model.kl3_weight))
                kd_loss = factor * kd_loss
                uncertainty = 0.5 * model.kl3_weight
            else:
                uncertainty = 0.0

            total_kd_loss = total_kd_loss + kd_loss
            total_kd_uncertainty = total_kd_uncertainty + uncertainty

    total_loss = (
        loss_fl +
        total_kd_loss +
        total_kd_uncertainty +
        center_loss + offset_loss +
        center_uncertainty_loss + offset_uncertainty_loss +
        fl_uncertainty_loss
    )

    return loss_fl, total_kd_loss, center_loss, offset_loss, total_loss



def prepare_voxels():
    voxel_x = 0.5
    voxel_y = 0.5
    voxel_z = 0.5

    XMIN, XMAX = -50.0, 50.0
    YMIN, YMAX = -2.0, 2.0
    ZMIN, ZMAX = -50.0, 50.0

    grid_center = torch.tensor([0.0, 0.0, 0.0])

    res_x = int((XMAX - XMIN) / voxel_x)
    res_y = int((YMAX - YMIN) / voxel_y)
    res_z = int((ZMAX - ZMIN) / voxel_z)

    print(f"Voxel grid resolution (in voxels): X={res_x}, Y={res_y}, Z={res_z}")

    XMIN_shifted, XMAX_shifted = XMIN, XMAX
    YMIN_shifted, YMAX_shifted = YMIN - grid_center[1].item(), YMAX - grid_center[1].item()
    ZMIN_shifted, ZMAX_shifted = ZMIN, ZMAX

    return {'voxel_size': (voxel_x, voxel_y, voxel_z),
            'grid_center': grid_center,
            'bounds': ((XMIN_shifted, XMAX_shifted),
                       (YMIN_shifted, YMAX_shifted),
                       (ZMIN_shifted, ZMAX_shifted)),
            'res': (res_x, res_y, res_z)}

def main(
        max_iters=25000,
        batch_size=6,
        nworkers=4,
        lr=5e-4,
        weight_decay=1e-7,
        grad_acc=5, 
        load_ckpt_dir=None,
        use_scheduler=True,
        scheduler_type='cosine',
        backbone='effb0_ori',
        norm_type='channel', 
        temperature=4,
        dis_type='ad',
        stage='stage1',
        dataset_name='Dur360BEV',
        dataset_dir=None,
        dataset_version='mini', 
        input_type_S='equi_imgs',
        map_r=100, 
        map_scale=2, 
        log_freq =10,
        img_freq=100,
        do_val=False,
        gamma=1,
        alpha=1,
        skip_dropout=0,
        spatial_dropout=False, 
        decoder_dropout=0.0
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG INFO]: Current using device:{device}, Batch_size:{batch_size}, num_workers:{nworkers}.")
    dataset_dir = _resolve_optional_path(dataset_dir) or Path(_default_dataset_dir())
    dataset_dir = dataset_dir.resolve()

    if not use_scheduler:
        model_name = f"fusion_{dis_type}_{stage}_{dataset_version}_{max_iters}_{batch_size}x{nworkers}_{lr:.0e}_{time.strftime('%m-%d_%H:%M')}"
    else:
        model_name = f"fusion_{dis_type}_{stage}_{dataset_version}_{max_iters}_{batch_size}x{nworkers}_{lr:.0e}s_{time.strftime('%m-%d_%H:%M')}"

    assert dataset_name in ['Dur360BEV', 'nuscenes'], "Dataset name not found."
    if dataset_name == 'Dur360BEV':
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset dir not found: {dataset_dir}. "
                "Set --dataset_dir or env DUR360BEV_DATASET_DIR."
            )
        train_loader, val_loader = dur360bev_dataset.compile_data(str(dataset_dir),
                                                        img_type='equi_img',
                                                        batch_size=batch_size, 
                                                        num_workers=nworkers, 
                                                        map_r=map_r, 
                                                        map_scale=map_scale, 
                                                        do_shuffle=True, 
                                                        is_train=True,
                                                        dataset_version=dataset_version)   # type: ignore
    elif dataset_name == 'nuscenes':
        import nuscenesdataset
        scene_centroid_x = 0.0
        scene_centroid_y = 1.0
        scene_centroid_z = 0.0

        scene_centroid_py = np.array([scene_centroid_x,
                                    scene_centroid_y,
                                    scene_centroid_z]).reshape([1, 3])
        scene_centroid = torch.from_numpy(scene_centroid_py).float()

        XMIN, XMAX = -50, 50
        ZMIN, ZMAX = -50, 50
        YMIN, YMAX = -5, 5
        bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

        Z, Y, X = 200, 8, 200

        res_scale = 2
        ncams=6
        final_dim = (900, 1600)
        resize_lim = [1,1]
        crop_offset = int(final_dim[0]*(1-resize_lim[0]))

        data_aug_conf = {
                'crop_offset': crop_offset,
                'resize_lim': resize_lim,
                'final_dim': final_dim,
                'H': 900, 'W': 1600,
                'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                'ncams': ncams,
                'img_type': 'equi'
            }
        
        dset='trainval'
        shuffle = True
        use_radar_filters = False
        nsweeps = 1
        do_shuffle_cams = False
        train_loader, val_loader = nuscenesdataset.compile_data(
                                            dset,
                                            str(dataset_dir),
                                            data_aug_conf=data_aug_conf,
                                            centroid=scene_centroid_py,
                                            bounds=bounds,
                                            res_3d=(Z,Y,X),
                                            bsz=batch_size,
                                            nworkers=nworkers,
                                            shuffle=shuffle,
                                            use_radar_filters=use_radar_filters,
                                            seqlen=1,
                                            nsweeps=nsweeps,
                                            do_shuffle_cams=do_shuffle_cams,
                                            get_tids=True,
                                        )
                  
    train_iterloader = iter(train_loader)
    
    vox_util_T = prepare_voxels()
    X_T,Y_T,Z_T = vox_util_T['res']
    
    scene_centroid_x = 0.0
    scene_centroid_y = 1.0
    scene_centroid_z = 0.0

    scene_centroid_py = np.array([scene_centroid_x,
                                  scene_centroid_y,
                                  scene_centroid_z]).reshape([1, 3])
    scene_centroid = torch.from_numpy(scene_centroid_py).float()
    XMIN, XMAX = -50, 50
    ZMIN, ZMAX = -50, 50
    YMIN, YMAX = -5, 5
    bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)
    Z_S, Y_S, X_S= 200, 8, 200

    vox_util_S = utils.vox.Vox_util(
             Z_S, Y_S, X_S,
            scene_centroid=scene_centroid.to(device),
            bounds=bounds,
            assert_cube=False)
       
    model_T = Segnet_fusion(Z_T, Y_T, X_T, 
                            Z_S, Y_S, X_S, 
                            vox_util_T, 
                            vox_util_S, 
                            rand_flip=False, 
                            teacher_encoder_type='res101', 
                            student_encoder_type="effb0_ori",
                            input_vfov=(-22.5, 22.5),
                            skip_dropout=skip_dropout,
                            spatial_dropout=spatial_dropout, 
                            decoder_dropout=decoder_dropout)
    
    model_T = model_T.to(device)
    
    model_S = SegnetEqui(Z_S, Y_S, X_S, vox_util_S, rand_flip=True, encoder_type=backbone)
    model_S = model_S.to(device)
    
    if use_scheduler:
        if scheduler_type=='onecycle':
            optimizer, scheduler = fetch_optimizer_onecycle(lr, 
                                        weight_decay, 
                                        1e-8, 
                                        max_iters, 
                                        model_S.parameters())
        elif scheduler_type=='cosine':
            optimizer, scheduler = fetch_optimizer_cosine(lr, 
                                        weight_decay, 
                                        1e-8, 
                                        max_iters, 
                                        model_S.parameters())
        else:
            raise ValueError(f"Unsupported scheduler_type: {scheduler_type}. Choose from ['onecycle', 'cosine']")
    else:
        optimizer = torch.optim.Adam(model_S.parameters(), lr=lr, weight_decay=weight_decay)
        
    print(f"[STATUS INFO] Teacher Total_params: {sum(p.numel() for p in model_T.parameters() if p.requires_grad)}")

    print(f"[STATUS INFO] Student Total_params: {sum(p.numel() for p in model_S.parameters() if p.requires_grad)}")

    criterion_bmse = CriterionBMSE().cuda()
    criterion_fl = CriterionFL(trainable=False, alpha=alpha, gamma=gamma).to(device)
    
    criterion_ad= CriterionAD(window_size=(100,100), reduction='mean',divide_by_batch=True).cuda()
    criterion_kl = CriterionCWD(norm_type,temperature).cuda()
    if dataset_name == 'Dur360BEV':
        writer_t = SummaryWriter(f'logs_dur360bev_{dataset_version}/distill/{model_name}/t')
        if do_val:
            writer_v = SummaryWriter(f'logs_dur360bev_{dataset_version}/distill/{model_name}/v')
        checkpoint_dir = f'./{dataset_version}_checkpoints_FL/distill/{model_name}'
    elif dataset_name == 'nuscenes':
        writer_t = SummaryWriter(f'logs_nuscenes_KD/distill/{model_name}/t')
        if do_val:
            writer_v = SummaryWriter(f'logs_nuscenes_KD/distill/{model_name}/v')
        checkpoint_dir = f'./checkpoints_nusc_KD/distill/{model_name}'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Folder '{checkpoint_dir}' created.")
    checkpoint_pattern = 'checkpoint_epoch_{epoch}.pth'
    global_step = 0
    if load_ckpt_dir is not None:
        checkpoint_path = _resolve_optional_path(load_ckpt_dir)
        checkpoint = torch.load(str(checkpoint_path))
        model_T.load_state_dict(checkpoint['model_state_dict'])
        loss = checkpoint['loss']
    else:
        
        print("No teacher's checkpoints found!")

    print(f"[DEBUG INFO]: Start to train {max_iters-global_step} steps from the global step: {global_step}.")

    model_T.eval()
    model_S.train()

    while global_step < max_iters:
        iter_start_time = time.time()

        global_step += 1
        iter_read_time = 0.0

        total_intersection_car = 0.0
        total_union_car = 0.0
        sw_t = None
        total_loss = total_kl = total_fl = total_center = total_offset = 0.0
        optimizer.zero_grad()  
        
        for internal_step in range(grad_acc):
            inter_start_time = time.time()

            if internal_step == grad_acc-1:
                sw_t = utils.sw.TensorBoardLogger(
                    writer = writer_t,
                    global_step = global_step,
                    log_freq = log_freq,
                    img_freq = img_freq 
                )

            try:
                sample = next(train_iterloader)
            except StopIteration:
                train_iterloader = iter(train_loader)
                sample = next(train_iterloader)
            if sample['image'].size(0) != batch_size:
                continue

            read_time = time.time()-inter_start_time
            iter_read_time += read_time
            with torch.no_grad():
                if dataset_name == 'Dur360BEV':
                    feat_T = run_model_forward(model_T, sample, device, stage)

            feat_S, pred_S, bev_seg_g, bev_center_g, bev_offset_g, center_e, offset_e = run_model_S_forward(
                model_S, sample, device, stage
            )
            visual_img_tensorboard(sw_t, sample, pred_S, bev_seg_g, offset_e, bev_offset_g, center_e, bev_center_g)    

            loss_fl, loss_kl, loss_center, loss_offset, loss = compute_stages_loss(
                model_S, stage, pred_S, bev_seg_g,
                feat_S, feat_T,
                criterion_kl, criterion_ad, criterion_bmse, criterion_fl,
                center_e, offset_e, bev_center_g, bev_offset_g, dis_type, device
            )

            intersection_car, union_car = compute_iou_car(pred_S, bev_seg_g)
            total_intersection_car += intersection_car.sum().item()
            total_union_car += union_car.sum().item()
            total_loss += loss.item()
            total_kl += loss_kl.item()
            total_fl += loss_fl.item()
            total_center += loss_center.item()
            total_offset += loss_offset.item()
            (loss / grad_acc).backward()
        
        iou_car = total_intersection_car / (total_union_car + 1e-4)
        sw_t.scalar('stats/fn_loss', total_fl / grad_acc)
        sw_t.scalar('stats/kl_loss', total_kl / grad_acc)
        sw_t.scalar('stats/loss_center', total_center / grad_acc)
        sw_t.scalar('stats/loss_offset', total_offset / grad_acc)
        sw_t.scalar('stats/total_loss', total_loss / grad_acc)
        sw_t.scalar('stats/iou_car', iou_car)
        
        optimizer.step()
        if use_scheduler:
            scheduler.step()

        if do_val and (global_step % 500 == 0):
            print(f"[VALIDATION]: Running validation at step {global_step}.")

            torch.cuda.empty_cache()
            model_S.eval()

            val_accumulated_fl_loss = 0.0
            val_accumulated_kl_loss = 0.0
            val_accumulated_center_loss = 0.0
            val_accumulated_offset_loss = 0.0
            val_accumulated_total_loss = 0.0
            val_accumulated_intersection = 0.0
            val_accumulated_union = 0.0
            val_batch_count = 0
            
            val_total_batches = len(val_loader)
            print(f"[VALIDATION]: Processing {val_total_batches} validation batches.")

            for val_step, sample in enumerate(val_loader):
                if sample['image'].size(0) != batch_size:
                    continue
                
                sw_v = None
                if val_step < 3 or val_step == val_total_batches - 1:
                    sw_v = utils.sw.TensorBoardLogger(
                        writer=writer_v,
                        global_step=global_step,
                        log_freq=1,
                        img_freq=1
                    )
                    sw_v.img_save = True
                
                with torch.no_grad():
                    if dataset_name == 'Dur360BEV':
                        feat_T = run_model_forward(model_T, sample, device, stage)
                        feat_val, pred_val_S, bev_seg_g_val, bev_center_g, bev_offset_g, center_e_val, offset_e_val = run_model_S_forward(
                            model_S, sample, device, stage
                        )
                        loss_fl_val, loss_kl_val, loss_center_val, loss_offset_val, loss_val = compute_stages_loss(
                            model_S, stage, pred_val_S, bev_seg_g_val,
                            feat_val, feat_T,
                            criterion_kl, criterion_ad, criterion_bmse, criterion_fl,
                            center_e_val, offset_e_val, bev_center_g, bev_offset_g, dis_type, device
                        )

                        intersection, union = compute_iou_car(pred_val_S, bev_seg_g_val)
                        val_accumulated_fl_loss += loss_fl_val.item()
                        val_accumulated_kl_loss += loss_kl_val.item()
                        val_accumulated_center_loss += loss_center_val.item()
                        val_accumulated_offset_loss += loss_offset_val.item()
                        val_accumulated_total_loss += loss_val.item()
                        val_accumulated_intersection += intersection.sum().item()
                        val_accumulated_union += union.sum().item()
                        
                        val_batch_count += 1
                        
                        if sw_v is not None:
                            visual_img_tensorboard(sw_v, sample, pred_val_S, bev_seg_g_val, offset_e_val, bev_offset_g, center_e_val, bev_center_g)

            avg_fl_loss = val_accumulated_fl_loss / val_batch_count
            avg_kl_loss = val_accumulated_kl_loss / val_batch_count
            avg_center_loss = val_accumulated_center_loss / val_batch_count
            avg_offset_loss = val_accumulated_offset_loss / val_batch_count
            avg_total_loss = val_accumulated_total_loss / val_batch_count
            avg_iou = (val_accumulated_intersection/ (val_accumulated_union + 1e-4))
            sw_v_metrics = utils.sw.TensorBoardLogger(
                writer=writer_v,
                global_step=global_step,
                log_freq=1,
                img_freq=1
            )
            sw_v_metrics.scalar('stats/fn_loss_val', avg_fl_loss)
            sw_v_metrics.scalar('stats/kl_loss_val', avg_kl_loss)
            sw_v_metrics.scalar('stats/total_loss_val', avg_total_loss)
            sw_v_metrics.scalar('stats/loss_center_val', avg_center_loss)
            sw_v_metrics.scalar('stats/loss_offset_val', avg_offset_loss)
            sw_v_metrics.scalar('stats/iou_car_val', avg_iou)
            
            print(f"[VALIDATION RESULTS]: Avg Loss: {avg_total_loss:.4f}, Avg IoU: {avg_iou:.4f}")
            
            model_S.train()
                


        current_lr = optimizer.param_groups[0]['lr']
        sw_t.scalar('_/current_lr', current_lr)

        if np.mod(global_step, 1000) == 0 or global_step == max_iters:
            checkpoint_path = f'{checkpoint_dir}/{checkpoint_pattern.format(epoch=global_step)}'
            save_checkpoint(global_step, model_S, optimizer, loss.item(), checkpoint_path, scheduler)
            print(f'[CHECKPOINT SAVED]: Iter: {global_step}, at {checkpoint_path}.')
        
        iter_end_time = time.time()
        iter_time = iter_end_time - iter_start_time
        remaining_time_sec = iter_time*(max_iters-global_step)
        hours, rem = divmod(remaining_time_sec, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"[STATUS INFO]: Iter: {global_step}/{max_iters}, loss: {loss.item():.4f}, IoU: [Car:{float(iou_car):.4f}], Iter Time: {iter_time:.2f}s, \n"
              f"Remaining Time: {int(hours)}h {int(minutes)}m {int(seconds)}s"  )

    writer_t.close()
    if do_val:
        writer_v.close()

if __name__ == '__main__':
    Fire(main)
