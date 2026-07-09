import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from utils import sample_points

checkpoint = os.path.join(ROOT, "checkpoints/sam-mt.pt")
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
device = "cuda"

predictor = build_sam2_video_predictor(
    config_file=model_cfg,
    ckpt_path=checkpoint,
    apply_postprocessing=False,
    hydra_overrides_extra=["++model.non_overlap_masks=false"],
    vos_optimized=False,
)


def process_single_video(
    video_dir: str,
    ann_frame_idx: int = 0,
    ann_obj_id: int    = 1,
    predictor=None,
    points=None,
    num_objects=None,
    video_name=None,
):
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)
    
    points_tensor = torch.from_numpy(points).float().unsqueeze(0)
    point_labels = torch.ones(points_tensor.shape[0], points_tensor.shape[1])
    points_per_object = [2 for i in range(num_objects)]
    
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points_tensor.to(device),
        labels=point_labels.to(device),
        points_per_object=points_per_object,
        video_name=video_name,
    )
    
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start.record()
    for out_frame_idx, out_obj_ids, out_mask_logits, out_mask_logits_sparse in predictor.propagate_in_video(
        inference_state,
        points_per_object=points_per_object
    ):
        pass
    
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    print("FPS: ", len(frame_names) / (elapsed_ms / 1000.0))
    
    max_reserved = torch.cuda.max_memory_reserved(device=None)
    print(f"max_VRAM: {max_reserved / (1024**3):.4f} GB, {max_reserved / (1024**2):.2f} MB")


if __name__ == "__main__":
    base_folder = os.path.join(ROOT, "demo/synthetic_benchmark")
    video_folder = os.path.join(base_folder, "JPEGImages")
    mask_folder = os.path.join(base_folder, "Annotations")
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for video_name in sorted(os.listdir(video_folder)): 
            video_dir = os.path.join(video_folder, video_name)
            mask_dir = os.path.join(mask_folder, video_name)
            first_mask_file = os.path.join(mask_dir, sorted(os.listdir(mask_dir))[0])
            first_mask = np.array(Image.open(first_mask_file))
            
            # Here we randomly sample 2 points per target for initialization.
            all_pts, original_object_ids = sample_points(first_mask, num_pts_per_object=2, device=device)
            print(f"Processing {video_name}: original_object_ids={original_object_ids}")
                
            process_single_video(
                video_dir=video_dir,
                ann_frame_idx=0,
                ann_obj_id=1,
                predictor=predictor,
                points=all_pts,
                num_objects=len(original_object_ids),
                video_name=video_name,
            )