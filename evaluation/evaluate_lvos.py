import argparse
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import glob
import json
import numpy as np
import torch
from PIL import Image
import shutil
import tempfile
from sam2.build_sam import build_sam2_video_predictor
from utils import list_frame_files, save_overlay_grid, save_label_mask

"""
Reference points are provided here for evaluation. Using manual clicks may further improve performance.
"""

def make_temp_video_folder(video_dir, video_name, start_idx):
    frame_files = list_frame_files(video_dir)
    tmp_dir = os.path.join("/tmp", f"lvos_{video_name}_f{start_idx}")
    if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir); 
    os.makedirs(tmp_dir)
    for new_i, old_i in enumerate(range(start_idx, len(frame_files))):
        src = os.path.abspath(os.path.join(video_dir, frame_files[old_i]))
        ext = os.path.splitext(frame_files[old_i])[1].lower()
        dst = os.path.join(tmp_dir, f"{new_i:05d}{ext}")
        os.symlink(src, dst)
    return tmp_dir


def load_points_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        rec = json.load(f)
    groups = []
    for g in rec["groups"]:
        groups.append({
            "ann_frame_idx": int(g["ann_frame_idx"]),
            "original_object_ids": [int(x) for x in g["object_ids"]],
            "points_per_obj": [int(x) for x in g["points_per_object"]],
            "points": np.array(g["all_points"], dtype=np.float32),
        })
    return {
        "video_name": rec["video_name"],
        "groups": groups,
    }


def merge_mask(results, best_logits, out_mask_logits, frame_idx, object_ids, out_dtype):
    logits_np = out_mask_logits.squeeze(0).detach().float().cpu().numpy()

    group_best_logit = logits_np.max(axis=0)
    group_best_idx = logits_np.argmax(axis=0)
    valid = group_best_logit > 0

    if frame_idx not in results:
        h, w = group_best_logit.shape
        results[frame_idx] = np.zeros((h, w), dtype=out_dtype)
        best_logits[frame_idx] = np.full((h, w), -np.inf, dtype=np.float32)

    object_ids = np.asarray(object_ids, dtype=out_dtype)
    update = valid & (group_best_logit > best_logits[frame_idx])

    results[frame_idx][update] = object_ids[group_best_idx[update]]
    best_logits[frame_idx][update] = group_best_logit[update]


def run_single_video(predictor, device, video_dir, record, output_dir):
    video_name = record["video_name"]
    save_dir = os.path.join(output_dir, video_name)
    os.makedirs(save_dir, exist_ok=True)

    all_object_ids = []
    for group in record["groups"]:
        all_object_ids.extend(group["original_object_ids"])
    out_dtype = np.uint8 if max(all_object_ids) <= 255 else np.uint16

    results = {}
    best_logits = {}

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        for group in record["groups"]:
            ann_frame_idx = group["ann_frame_idx"]
            points_per_obj = group["points_per_obj"]
            points = group["points"]
            object_ids = group["original_object_ids"]

            points_tensor = torch.from_numpy(points).float().unsqueeze(0)
            point_labels = torch.ones(points_tensor.shape[0], points_tensor.shape[1])
            
            tmp_dir = make_temp_video_folder(video_dir, video_name, ann_frame_idx)
            inference_state = predictor.init_state(video_path=tmp_dir)
            predictor.reset_state(inference_state)

            _, _, _, out_mask_logits_sparse = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=points_tensor.to(device),
                labels=point_labels.to(device),
                points_per_object=points_per_obj,
                video_name=video_name,
            )
            merge_mask(
                results,
                best_logits,
                out_mask_logits_sparse,
                ann_frame_idx,
                object_ids,
                out_dtype,
            )

            for out_frame_idx, _, _, out_mask_logits_sparse in predictor.propagate_in_video(
                inference_state,
                points_per_object=points_per_obj,
            ):
                if out_frame_idx <= 0:
                    continue
                merge_mask(
                    results,
                    best_logits,
                    out_mask_logits_sparse,
                    ann_frame_idx + out_frame_idx,
                    object_ids,
                    out_dtype,
                )

            shutil.rmtree(tmp_dir)
            torch.cuda.empty_cache()

    frame_files = list_frame_files(video_dir)
    
    if results:
        h, w = next(iter(results.values())).shape
    else:
        w, h = Image.open(os.path.join(video_dir, frame_files[0])).size
    empty_mask = np.zeros((h, w), dtype=out_dtype)
    for fid, frame_name in enumerate(frame_files):
        frame_stem = os.path.splitext(frame_name)[0]
        mask = results.get(fid, empty_mask)
        save_label_mask(mask, frame_stem, save_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark inference")
    p.add_argument("--video_root", default="/data/lvos2_val/JPEGImages/") # change to your path
    p.add_argument("--points_dir", default=os.path.join(ROOT, "demo/points/lvos2"))
    p.add_argument("--output_dir", default=os.path.join(ROOT, "outputs/benchmark/lvos2"))
    p.add_argument("--checkpoint", default=os.path.join(ROOT, "checkpoints/sam-mt.pt"))
    p.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--show_overlay", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    json_paths = sorted(glob.glob(os.path.join(args.points_dir, "*.json")))
    
    predictor = build_sam2_video_predictor(
        config_file=args.config,
        ckpt_path=args.checkpoint,
        apply_postprocessing=False,
        hydra_overrides_extra=["++model.non_overlap_masks=false"],
        vos_optimized=False,
    ).to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Videos to run: {len(json_paths)}", flush=True)

    for i, json_path in enumerate(json_paths, start=1):
        record = load_points_json(json_path)
        video_name = record["video_name"]
        # if video_name != '9HEh93ef': # d83wYdy0
        #     continue
        video_dir = os.path.join(args.video_root, video_name)
        save_dir = os.path.join(args.output_dir, video_name)

        group_msg = ", ".join([f"frame_idx={g['ann_frame_idx']}, ids={g['original_object_ids']}"
            for g in record["groups"]
        ])

        print(f"[{i}/{len(json_paths)}] RUN {video_name}: {group_msg}", flush=True)

        run_single_video(
            predictor=predictor,
            device=device,
            video_dir=video_dir,
            record=record,
            output_dir=args.output_dir,
        )

        if args.show_overlay:
            save_overlay_grid(video_dir, save_dir, os.path.join(args.output_dir, f"{video_name}_grid.jpg"))

    print(f"\nAll finished. Output: {os.path.abspath(args.output_dir)}", flush=True)


if __name__ == "__main__":
    main()