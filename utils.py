import os
import json
import subprocess
import re
import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2_video_predictor
import torch.nn.functional as F

DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"

PREDICTOR = None
LOADING_END, TRACKING_END = 0.20, 0.85
OBJECT_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 165, 0), (255, 0, 255),
    (0, 255, 255), (255, 255, 0), (128, 0, 255), (255, 128, 0), (0, 255, 128),
]


def list_frame_files(video_dir):
    files = [
        f for f in os.listdir(video_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    if not files:
        raise RuntimeError(f"No frames found in {video_dir}")
    return files


def list_mask_files(save_dir):
    files = [
        f for f in os.listdir(save_dir)
        if f.lower().endswith(".png") and os.path.splitext(f)[0].isdigit()
    ]
    files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    return files


def _natural_key(name):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]


def resolve_video_scenes(videos_root):
    videos_root = os.path.abspath(videos_root)
    if not os.path.isdir(videos_root):
        raise RuntimeError(f"Videos root not found: {videos_root}")
    out = []
    for name in sorted(os.listdir(videos_root), key=_natural_key):
        path = os.path.join(videos_root, name)
        if os.path.isdir(path):
            try:
                list_frame_files(path)
                out.append((name, path))
            except RuntimeError:
                pass
    if not out:
        raise RuntimeError(f"No video scenes found in {videos_root}")
    return out



def load_first_frame(video_dir):
    files = list_frame_files(video_dir)
    return np.array(Image.open(os.path.join(video_dir, files[0])).convert("RGB"))



def save_mask(out_mask_logits, frame_idx, save_dir):
    logits_np = out_mask_logits.squeeze(0).float().cpu().numpy()
    h, w = logits_np.shape[1], logits_np.shape[2]
    union_mask = np.any(logits_np > 0, axis=0)

    full_mask = np.zeros((h, w), dtype=np.uint8)
    max_target_idx = np.argmax(logits_np, axis=0)
    full_mask[union_mask] = max_target_idx[union_mask] + 1

    os.makedirs(save_dir, exist_ok=True)
    img = Image.fromarray(full_mask, mode="P"); img.putpalette(DAVIS_PALETTE)
    img.save(os.path.join(save_dir, f"{frame_idx:05d}.png"))
    

def save_label_mask(mask, frame_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    img = Image.fromarray(mask.astype(np.uint8), mode="P")
    img.putpalette(DAVIS_PALETTE)
    if isinstance(frame_name, int):
        out_name = f"{frame_name:05d}.png"
    else:
        out_name = f"{frame_name}.png"
    img.save(os.path.join(save_dir, out_name))
    
    
def save_overlay_grid(video_dir, mask_dir, out_path, alpha=0.5, ncols=10, thumb_w=320):
    frames, masks = list_frame_files(video_dir), list_mask_files(mask_dir)
    cells = []
    for i, f in enumerate(frames):
        rgb = np.array(Image.open(os.path.join(video_dir, f)).convert("RGB"), dtype=np.float32)
        m = np.array(Image.open(os.path.join(mask_dir, masks[i])).convert("RGB"), dtype=np.float32)
        fg = m.sum(2) > 0
        rgb[fg] = rgb[fg] * (1 - alpha) + m[fg] * alpha
        cell = Image.fromarray(rgb.astype(np.uint8))
        cell = cell.resize((thumb_w, int(cell.height * thumb_w / cell.width)), Image.BILINEAR)
        cells.append(cell)
    tw, th = cells[0].size
    nrows = (len(cells) + ncols - 1) // ncols
    grid = Image.new("RGB", (ncols * tw, nrows * th), (255, 255, 255))
    for i, cell in enumerate(cells):
        r, c = divmod(i, ncols)
        grid.paste(cell, (c * tw, r * th))
    grid.save(out_path)

def build_overlay_mp4(video_dir, save_dir, mp4_path, fps=24, alpha=0.5):
    frame_files = list_frame_files(video_dir)
    mask_files = list_mask_files(save_dir)
    if len(mask_files) < len(frame_files):
        print(
            f"[warn] masks={len(mask_files)} < frames={len(frame_files)}, "
            "missing frames will show without overlay",
            flush=True,
        )
    os.makedirs(save_dir, exist_ok=True)
    first = np.array(Image.open(os.path.join(video_dir, frame_files[0])).convert("RGB"))
    h, w = first.shape[:2]
    raw = mp4_path + ".raw.mp4"
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {raw}")
    for i, name in enumerate(frame_files):
        rgb = np.array(Image.open(os.path.join(video_dir, name)).convert("RGB"))
        if i < len(mask_files):
            mpath = os.path.join(save_dir, mask_files[i])
            m = np.array(Image.open(mpath).convert("RGB"), dtype=np.float32)
            fg = m.sum(axis=2) > 0
            f = rgb.astype(np.float32)
            rgb = f.copy()
            rgb[fg] = f[fg] * (1 - alpha) + m[fg] * alpha
            rgb = rgb.astype(np.uint8)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", raw,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4_path],
            check=True,
        )
        os.remove(raw)
    except (FileNotFoundError, subprocess.CalledProcessError):
        os.replace(raw, mp4_path)
    return os.path.abspath(mp4_path)



def new_click_state(base_image):
    return {"base": base_image, "all_points": [], "points_per_obj": [], "current": [], "obj_idx": 1}


def render_clicks(state):
    img = Image.fromarray(state["base"]).convert("RGB")
    draw = ImageDraw.Draw(img)
    off = 0
    for oi, n in enumerate(state["points_per_obj"]):
        c = OBJECT_COLORS[oi % len(OBJECT_COLORS)]
        for j in range(n):
            x, y = state["all_points"][off + j]
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=c, width=2)
            draw.text((x + 8, y - 8), str(oi + 1), fill=c)
        off += n
    if state["current"]:
        c = OBJECT_COLORS[(state["obj_idx"] - 1) % len(OBJECT_COLORS)]
        for j, (x, y) in enumerate(state["current"]):
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=c)
            draw.text((x + 8, y - 8), f"{state['obj_idx']}.{j + 1}", fill=c)
    return np.array(img)


def status_text(state):
    ppo = state["points_per_obj"]
    return (
        f"Object {state['obj_idx']}: {len(state['current'])} point(s) | "
        f"Finished: {len(ppo)} object(s)"
    )


def _commit_current(state):
    state = dict(state)
    n = len(state["current"])
    state["all_points"] = list(state["all_points"]) + list(state["current"])
    state["points_per_obj"] = list(state["points_per_obj"]) + [n]
    state["current"] = []
    state["obj_idx"] += 1
    return state, n


def finish_current_object(state):
    if not state["current"]:
        return state, "Need at least 1 point for current object."
    state, n = _commit_current(state)
    return state, f"Added object with {n} point(s). Now labeling object {state['obj_idx']}."


def finalize_state_for_tracking(state):
    return _commit_current(dict(state))[0] if state.get("current") else state


def save_user_points_json(state, save_dir, video_dir):
    H, W = state["base"].shape[:2]
    all_pts = list(state["all_points"])
    ppo = list(state["points_per_obj"])
    objects, off, brief = [], 0, []
    for oid, n in enumerate(ppo, start=1):
        pts = all_pts[off: off + n]
        pt_recs = []
        for pid, (x, y) in enumerate(pts, start=1):
            pt_recs.append({
                "point_id": pid, "x": float(x), "y": float(y),
                "x_norm": round(float(x) / W, 6), "y_norm": round(float(y) / H, 6),
            })
        objects.append({"object_id": oid, "num_points": n, "points": pt_recs})
        brief.append(f"  obj{oid}: " + ", ".join(f"({x:.0f},{y:.0f})" for x, y in pts))
        off += n

    record = {
        "video_dir": video_dir,
        "frame_size": {"H": H, "W": W},
        "summary": {"num_objects": len(ppo), "points_per_obj": ppo, "total_points": len(all_pts)},
        "objects": objects,
        "model_input": {
            "frame_idx": 0, "obj_id": 1,
            "points_flat_xy": [[float(x), float(y)] for x, y in all_pts],
            "points_per_object": ppo, "labels": "all_ones",
        },
    }
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "user_points.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    text = f"objects={len(ppo)}, total_points={len(all_pts)}, points_per_obj={ppo}\n" + "\n".join(brief)
    return path, text



def get_predictor(checkpoint, config, device):
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = build_sam2_video_predictor(
            config_file=config, ckpt_path=checkpoint,
            apply_postprocessing=False,
            hydra_overrides_extra=["++model.non_overlap_masks=false"],
            vos_optimized=False,
        )
        PREDICTOR.to(device)
    return PREDICTOR


def run_tracking(video_dir, save_dir, checkpoint, config, device, state, fps=24, progress_fn=None):
    state = finalize_state_for_tracking(state)
    if not state["points_per_obj"]:
        raise gr.Error("Please label at least one object first.")

    n_frames = len(list_frame_files(video_dir))
    json_path, brief = save_user_points_json(state, save_dir, video_dir)
    print(brief, flush=True)
    print(f"[saved] {json_path}", flush=True)

    points = np.array(state["all_points"], dtype=np.float32)
    ppo = state["points_per_obj"]

    def prog(frac, desc=""):
        if progress_fn:
            progress_fn(frac, desc=desc)

    prog(0.0, "Loading frames")
    predictor = get_predictor(checkpoint, config, device)
    prog(LOADING_END * 0.3, "Loading frames")
    inf_state = predictor.init_state(video_path=video_dir)
    prog(LOADING_END * 0.8, "Loading frames")
    predictor.reset_state(inf_state)
    prog(LOADING_END, "Loading frames")

    pts = torch.from_numpy(points).float().unsqueeze(0).to(device)
    lbl = torch.ones(pts.shape[:2], dtype=torch.int32, device=device)
    predictor.add_new_points_or_box(
        inf_state, frame_idx=0, obj_id=1,
        points=pts, labels=lbl, points_per_object=ppo,
    )

    results = []
    span = TRACKING_END - LOADING_END
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        for i, (fid, _, _, logits) in enumerate(
            predictor.propagate_in_video(inf_state, points_per_object=ppo)
        ):
            results.append((fid, logits.detach().cpu()))
            prog(LOADING_END + span * (i + 1) / n_frames, "Propagation")

    print("Propagation done. Start overlay rendering and saving mp4...")
    save_total = len(results) + n_frames
    save_span = 1.0 - TRACKING_END
    prog(TRACKING_END, "Rendering video frames and mp4")
    for i, (fid, logits) in enumerate(results):
        save_mask(logits, fid, save_dir)
        prog(TRACKING_END + save_span * (i + 1) / save_total, "Rendering video frames and mp4")

    mp4 = build_overlay_mp4(video_dir, save_dir, os.path.join(save_dir, "result.mp4"), fps=fps)
    prog(1.0, "Rendering video frames and mp4")

    n_masks = sum(1 for f in os.listdir(save_dir) if f.endswith(".png") and f[:-4].isdigit())
    print(f"\nSaved {n_masks} masks, video: {mp4}", flush=True)
    print("\nSaved! Press Ctrl+C to quit.\n", flush=True)
    return mp4


def _erode_mask(mask: torch.Tensor, k: int):
    x = mask.float().unsqueeze(0).unsqueeze(0)
    w = torch.ones(1, 1, k, k, device=mask.device)
    return (F.conv2d(x, w, padding=k // 2) == k * k).squeeze(0).squeeze(0).bool()

def sample_points_from_mask(mask, num_pts=2, erode_k=10, grid_threshold=100, device="cuda"):
    mask = torch.as_tensor(mask, device=device).bool()
    inner = _erode_mask(mask, k=erode_k)
    if not inner.any(): # very small target
        inner = mask
    allowed_idx = inner.nonzero(as_tuple=False)
    if allowed_idx.size(0) > num_pts:
        if allowed_idx.size(0) > grid_threshold:
            y0, y1 = allowed_idx[:, 0].min(), allowed_idx[:, 0].max()
            x0, x1 = allowed_idx[:, 1].min(), allowed_idx[:, 1].max()
            gy = torch.linspace(y0, y1, num_pts * 2, device=device)
            gx = torch.linspace(x0, x1, num_pts * 2, device=device)
            grid = [torch.tensor([y, x], device=device) for y in gy for x in gx if inner[int(y), int(x)]]
            if len(grid) >= num_pts:
                idx = torch.randperm(len(grid), device=device)[:num_pts]
                sampled = torch.stack([grid[i] for i in idx])
            else:
                sampled = allowed_idx[torch.randperm(allowed_idx.size(0), device=device)[:num_pts]]
        else:
            sampled = allowed_idx[torch.randperm(allowed_idx.size(0), device=device)[:num_pts]]
    else:
        sampled = allowed_idx
    return torch.stack([sampled[:, 1], sampled[:, 0]], dim=1).float()

def sample_points(first_mask, num_pts_per_object=2, device="cuda"):
    object_ids = sorted(v for v in np.unique(first_mask) if v != 0)
    pts = [sample_points_from_mask(first_mask == oid, num_pts_per_object, device=device) for oid in object_ids]
    return torch.cat(pts, dim=0).cpu().numpy(), object_ids