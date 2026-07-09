import os
import argparse
import numpy as np
import torch
from utils import get_predictor, save_mask, build_overlay_mp4

#-------------------------------------------------------------------------------------------------
# For automatic clicking & annotation, please use "inference_video_gradio.py".
#-------------------------------------------------------------------------------------------------

# TODO: Set your video path and input points
VIDEO_DIR = "demo/video/scene1"
POINTS = [
    [[606, 457]],   # target1
    [[840, 364]],   # target2
    [[1314, 275]],  # target3
    [[1399, 376]],  # target4
]
# Each target supports an arbitrary number of points, e.g., [[606, 457], [627, 494]] for target1.

POINTS_PER_OBJECT = [len(p) for p in POINTS]
POINTS = np.vstack(POINTS).astype(np.float32)
ANN_FRAME_IDX = 0


def run(video_dir, save_dir, checkpoint, config, device, points, points_per_obj, fps):
    os.makedirs(save_dir, exist_ok=True)
    print(f"video={video_dir}", flush=True)
    print(f"points_per_object={points_per_obj}, total={len(points)}", flush=True)

    predictor = get_predictor(checkpoint, config, device)
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    points_t = torch.from_numpy(points).float().unsqueeze(0)
    labels = torch.ones(points_t.shape[0], points_t.shape[1])

    results = []
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        _, _, _, mask_logits = predictor.add_new_points_or_box(
            inference_state,
            frame_idx=ANN_FRAME_IDX,
            obj_id=1,
            points=points_t.to(device),
            labels=labels.to(device),
            points_per_object=points_per_obj,
        )
        results.append((ANN_FRAME_IDX, mask_logits.detach().cpu()))

        for fid, _, _, mask_logits in predictor.propagate_in_video(
            inference_state, points_per_object=points_per_obj
        ):
            if fid == ANN_FRAME_IDX:
                continue
            results.append((fid, mask_logits.detach().cpu()))

    for fid, logits in results:
        save_mask(logits, fid, save_dir)

    mp4_path = os.path.join(save_dir, "result.mp4")
    mp4 = build_overlay_mp4(video_dir, save_dir, mp4_path, fps=fps)
    print(f"Saved mp4 at {mp4}", flush=True)
    return mp4


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir", default=VIDEO_DIR)
    p.add_argument("--save_dir", default="outputs/your_video")
    p.add_argument("--checkpoint", default="checkpoints/sam-mt.pt")
    p.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fps", type=int, default=15)
    args = p.parse_args()

    run(
        video_dir=args.video_dir,
        save_dir=args.save_dir,
        checkpoint=args.checkpoint,
        config=args.config,
        device=args.device,
        points=POINTS,
        points_per_obj=POINTS_PER_OBJECT,
        fps=args.fps,
    )