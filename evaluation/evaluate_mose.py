import argparse
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import glob
import json
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
from utils import save_overlay_grid, save_mask

"""
Reference points are provided here for evaluation. Using manual clicks may further improve performance.
"""

def load_points_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        rec = json.load(f)
    oids = [int(x) for x in rec["original_object_ids"]]
    pts_list = [np.array(rec["points_per_object"][str(oid)], dtype=np.float32) for oid in oids]
    points = np.array(rec["all_points"], dtype=np.float32) if "all_points" in rec else np.concatenate(pts_list)
    return {
        "video_name": rec["video_name"],
        "ann_frame_idx": int(rec.get("ann_frame_idx", 0)),
        "original_object_ids": oids,
        "points_per_obj": [p.shape[0] for p in pts_list],
        "points": points,
    }


def run_single_video(predictor, device, video_dir, record, output_dir):
    video_name = record["video_name"]
    ann_frame_idx = record["ann_frame_idx"]
    points_per_obj = record["points_per_obj"]
    points = record["points"]
    points_tensor = torch.from_numpy(points).float().unsqueeze(0)
    point_labels = torch.ones(points_tensor.shape[0], points_tensor.shape[1])

    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)
    
    save_dir = os.path.join(output_dir, video_name); os.makedirs(save_dir, exist_ok=True)
    results = []
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        _, _, _, out_mask_logits_sparse = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=1,
            points=points_tensor.to(device),
            labels=point_labels.to(device),
            points_per_object=points_per_obj,
            video_name=video_name,
        )
        results.append((ann_frame_idx, out_mask_logits_sparse.detach().cpu()))

        for out_frame_idx, _, _, out_mask_logits_sparse in predictor.propagate_in_video(
            inference_state,
            points_per_object=points_per_obj,
        ):
            if out_frame_idx == ann_frame_idx:
                continue    
            results.append((out_frame_idx, out_mask_logits_sparse.detach().cpu()))

    for fid, logits in results:
        save_mask(logits, fid, save_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark inference")
    p.add_argument("--video_root", default="/data/mosev2/valid/JPEGImages/") # change to your path
    p.add_argument("--points_dir", default=os.path.join(ROOT, "demo/points/mosev2"))
    p.add_argument("--output_dir", default=os.path.join(ROOT, "outputs/benchmark/mosev2"))
    p.add_argument("--checkpoint", default=os.path.join(ROOT, "checkpoints/sam-mt.pt"))
    p.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--show_overlay", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    json_paths = sorted(glob.glob(os.path.join(args.points_dir, "*.json")))

    # build predictor
    predictor = build_sam2_video_predictor(
        config_file=args.config,
        ckpt_path=args.checkpoint,
        apply_postprocessing=False,
        hydra_overrides_extra=["++model.non_overlap_masks=false"],
        vos_optimized=False,
    ).to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Videos to run: {len(json_paths)}", flush=True)

    # process each video
    for i, json_path in enumerate(json_paths, start=1):
        record = load_points_json(json_path)
        video_name = record["video_name"]
        video_dir = os.path.join(args.video_root, video_name)
        save_dir = os.path.join(args.output_dir, video_name)

        print(f"[{i}/{len(json_paths)}] RUN {video_name}: objects={record['original_object_ids']}", flush=True)

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