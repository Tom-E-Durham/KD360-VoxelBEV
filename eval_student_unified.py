import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.utils as vutils
from fire import Fire
from tensorboardX import SummaryWriter

from Dur360BEV_dataset import dur360bev_dataset
from nets.segnet_equi import Segnet as SegStudent
import utils.sw
import utils.vox


def _resolve_optional_path(path_value):
    if path_value is None or path_value == "":
        return None
    p = Path(path_value).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def _default_dataset_dir():
    env_dir = os.environ.get("DUR360BEV_DATASET_DIR")
    if env_dir:
        return str(Path(env_dir).expanduser())
    return str((Path(__file__).resolve().parent / "data" / "Dur360BEV_Dataset_Complete").resolve())


def strip_prefix_if_present(state_dict, prefix_list=("module.", "model.", "student.")):
    new_sd = {}
    for k, v in state_dict.items():
        new_k = k
        for p in prefix_list:
            if new_k.startswith(p):
                new_k = new_k[len(p):]
        new_sd[new_k] = v
    return new_sd


def load_partial_weights(model, checkpoint_path, device="cpu", key="model_state_dict"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and key in ckpt:
        sd = ckpt[key]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    sd = strip_prefix_if_present(sd, ("module.", "model.", "student."))
    model_sd = model.state_dict()

    filtered = {}
    shape_mismatch = []
    unexpected = []
    for k, v in sd.items():
        if k in model_sd:
            if model_sd[k].shape == v.shape:
                filtered[k] = v
            else:
                shape_mismatch.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
        else:
            unexpected.append(k)
    missing = [k for k in model_sd.keys() if k not in filtered]
    model.load_state_dict(filtered, strict=False)

    print("=== Partial load report ===")
    print(f"Loaded params: {len(filtered)} / {len(model_sd)}")
    if missing:
        print(f"Missing (not in ckpt): {len(missing)} e.g. {missing[:8]}")
    if unexpected:
        print(f"Unexpected (only in ckpt): {len(unexpected)} e.g. {unexpected[:8]}")
    if shape_mismatch:
        print(f"Shape mismatch: {len(shape_mismatch)}")
        for k, s_ckpt, s_model in shape_mismatch[:8]:
            print(f"  {k}: ckpt{s_ckpt} vs model{s_model}")


def _build_vox_util(device):
    scene_centroid = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32, device=device)
    bounds = (-50, 50, -5, 5, -50, 50)
    zyx = (200, 8, 200)
    return utils.vox.Vox_util(
        zyx[0],
        zyx[1],
        zyx[2],
        scene_centroid=scene_centroid,
        bounds=bounds,
        assert_cube=False,
    ), zyx


def _build_val_loader(dataset_name, dataset_dir, dataset_version, batch_size, nworkers, map_r, map_scale):
    if dataset_name != "Dur360BEV":
        raise ValueError("Only dataset_name='Dur360BEV' is supported in this script.")
    _, val_loader = dur360bev_dataset.compile_data(
        str(dataset_dir),
        img_type="equi_img",
        batch_size=batch_size,
        num_workers=nworkers,
        map_r=map_r,
        map_scale=map_scale,
        do_shuffle=False,
        is_train=True,
        dataset_version=dataset_version,
    )
    return val_loader


def _compute_iou_metrics(seg_logits, bev_seg):
    seg_pred = torch.sigmoid(seg_logits).round()
    seg_car = seg_pred[:, 0]
    gt_car = bev_seg[:, 0]

    def _inter_union(a, b):
        inter = (a * b).sum(dim=[1, 2]).sum().item()
        union = (a + b).clamp(0, 1).sum(dim=[1, 2]).sum().item()
        return inter, union

    inter, union = _inter_union(seg_car, gt_car)
    inter_25, union_25 = _inter_union(seg_car[:, 50:150, 50:150], gt_car[:, 50:150, 50:150])
    inter_10, union_10 = _inter_union(seg_car[:, 80:120, 80:120], gt_car[:, 80:120, 80:120])

    return {
        "intersection": inter,
        "union": union,
        "intersection_25": inter_25,
        "union_25": union_25,
        "intersection_10": inter_10,
        "union_10": union_10,
    }, seg_car


def _save_pred(seg_car, save_dir, step):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    vutils.save_image(seg_car.unsqueeze(1), str(save_dir / f"{step:06d}.png"))


def run_eval(model, val_loader, device, log_freq, img_freq, save_pred, pred_dir):
    writer = SummaryWriter(log_dir=f"results_dur360bev/equi/{int(time.time())}")
    total = {"intersection": 0.0, "union": 0.0, "intersection_25": 0.0, "union_25": 0.0, "intersection_10": 0.0, "union_10": 0.0}

    model.eval()
    max_iters = len(val_loader)
    with torch.no_grad():
        for step, sample in enumerate(val_loader, start=1):
            iter_start = time.time()
            if sample["image"].size(0) == 0:
                continue

            sw = utils.sw.TensorBoardLogger(writer=writer, global_step=step, log_freq=log_freq, img_freq=img_freq)
            equi = sample["image"].to(device).float() - 0.5
            bev_seg_g = sample["bev_seg"].to(device)

            _, _, seg_e, center_e, offset_e, _, _ = model(equi)
            metrics, seg_car = _compute_iou_metrics(seg_e, bev_seg_g)

            for k, v in metrics.items():
                total[k] += v

            iou = total["intersection"] / (total["union"] + 1e-4)
            iou25 = total["intersection_25"] / (total["union_25"] + 1e-4)
            iou10 = total["intersection_10"] / (total["union_10"] + 1e-4)
            sw.scalar("results/mean_iou", iou)
            sw.scalar("results/mean_iou_25", iou25)
            sw.scalar("results/mean_iou_10", iou10)

            if sw.img_save:
                sw.rgb_img("0_inputs/image", equi + 0.5)
                sw.bin_img("1_outputs/bev_seg_car_e", torch.sigmoid(seg_e[:, 0]))
                sw.bin_img("1_outputs/bev_seg_car_g", bev_seg_g[:, 0])
                sw.rgb_img("1_outputs/bev_offset_e", sw.offset2color(offset_e))
                sw.rgb_img("1_outputs/bev_offset_g", sw.offset2color(sample["offset"].to(device)))
                sw.bin_img("1_outputs/bev_center_e", center_e[:, 0])
                sw.bin_img("1_outputs/bev_center_g", sample["center"].to(device)[:, 0])

            if save_pred:
                _save_pred(seg_car, pred_dir, step)

            iter_time = time.time() - iter_start
            remain = iter_time * (max_iters - step)
            h, rem = divmod(remain, 3600)
            m, s = divmod(rem, 60)
            print(
                f"[EVAL] Iter {step}/{max_iters} | "
                f"IoU {iou:.4f} | IoU_25 {iou25:.4f} | IoU_10 {iou10:.4f} | "
                f"Iter {iter_time:.2f}s | Remaining {int(h)}h {int(m)}m {int(s)}s"
            )

    writer.close()
    final_iou = total["intersection"] / (total["union"] + 1e-4)
    print(f"[EVAL DONE] Final IoU: {final_iou:.4f}")


def run_speed(model, val_loader, device, repeats, warmup, batch_size):
    model.eval()
    sample = None
    for s in val_loader:
        if s["image"].size(0) == batch_size:
            sample = s
            break
    if sample is None:
        raise RuntimeError("No validation batch matches requested batch_size.")

    equi = sample["image"].to(device).float() - 0.5
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(equi)
    if device.type == "cuda":
        torch.cuda.synchronize()

    best_sec = math.inf
    with torch.no_grad():
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(repeats):
                torch.cuda.synchronize()
                start.record()
                _ = model(equi)
                end.record()
                torch.cuda.synchronize()
                best_sec = min(best_sec, start.elapsed_time(end) / 1000.0)
        else:
            for _ in range(repeats):
                t0 = time.perf_counter()
                _ = model(equi)
                best_sec = min(best_sec, time.perf_counter() - t0)

    best_fps = batch_size / best_sec if best_sec > 0 else float("inf")
    print(f"[SPEED] Best forward latency: {best_sec*1000:.3f} ms | Batch={batch_size} | FPS={best_fps:.2f}")


def main(
    mode="eval",  # eval or speed
    batch_size=6,
    nworkers=6,
    encoder_type="effb0_ori",
    checkpoint_dir=None,
    dataset_name="Dur360BEV",
    dataset_dir=None,
    dataset_version="extended",
    map_r=100,
    map_scale=2,
    log_freq=1,
    img_freq=10,
    save_pred=False,
    pred_dir="./results_dur360bev/equi/pred",
    warmup=5,
    repeats=20,
):
    if checkpoint_dir is None:
        raise ValueError("Please provide checkpoint_dir.")
    if mode not in ("eval", "speed"):
        raise ValueError("mode must be 'eval' or 'speed'.")
    if dataset_version not in ("extended", "mini"):
        raise ValueError("dataset_version must be 'extended' or 'mini'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device={device} | batch_size={batch_size} | nworkers={nworkers}")

    dataset_dir = _resolve_optional_path(dataset_dir) or Path(_default_dataset_dir())
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset dir not found: {dataset_dir}. "
            "Set --dataset_dir or env DUR360BEV_DATASET_DIR."
        )
    ckpt_path = _resolve_optional_path(checkpoint_dir)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    val_loader = _build_val_loader(
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        dataset_version=dataset_version,
        batch_size=batch_size,
        nworkers=nworkers,
        map_r=map_r,
        map_scale=map_scale,
    )

    vox_util_s, (z_s, y_s, x_s) = _build_vox_util(device)
    model = SegStudent(z_s, y_s, x_s, vox_util_s, rand_flip=False, encoder_type=encoder_type).to(device)
    load_partial_weights(model, str(ckpt_path), device=device, key="model_state_dict")
    print(f"[INFO] Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if mode == "eval":
        run_eval(model, val_loader, device, log_freq, img_freq, save_pred, pred_dir)
    else:
        run_speed(model, val_loader, device, repeats=repeats, warmup=warmup, batch_size=batch_size)


if __name__ == "__main__":
    Fire(main)
