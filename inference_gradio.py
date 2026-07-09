import os
import argparse
import gradio as gr

os.environ["GRADIO_TEMP_DIR"] = "gradio_cache"

from utils import (
    resolve_video_scenes,
    load_first_frame,
    new_click_state,
    render_clicks,
    status_text,
    finish_current_object,
    run_tracking,
)

DEMO_CSS = """
footer {display: none !important}
h1 {text-align: center !important}
#video-radio .wrap {justify-content: center !important}
#video-radio > .block {text-align: center}
"""


def build_demo(args):
    videos_root = os.path.abspath(args.videos_root)
    videos = resolve_video_scenes(videos_root)
    video_map = {name: path for name, path in videos}
    default_name = videos[0][0]
    default_dir = video_map[default_name]

    init_state = new_click_state(load_first_frame(default_dir))

    with gr.Blocks(title="SAM-MT Demo", css=DEMO_CSS) as demo:
        gr.Markdown("# SAM-MT: Real-time Interactive Multi-Target Video Segmentation")

        video_select = gr.Radio(
            choices=[name for name, _ in videos],
            value=default_name,
            label="Video",
            elem_id="video-radio",
        )
        video_dir_state = gr.State(default_dir)
        click_state = gr.State(init_state)

        with gr.Row():
            with gr.Column(scale=4):
                canvas = gr.Image(
                    value=render_clicks(init_state),
                    label="First frame",
                    interactive=False,
                    type="numpy",
                )
            with gr.Column(scale=1, min_width=220, elem_id="side-panel"):
                status = gr.Textbox(value=status_text(init_state), label="Clicks")
                btn_next = gr.Button("Next object")
                btn_reset = gr.Button("Reset")
                btn_run = gr.Button("Run tracking", variant="primary")

        output_video = gr.Video(label="Output video")

        def on_video_change(video_name):
            video_dir = video_map[video_name]
            state = new_click_state(load_first_frame(video_dir))
            return (
                render_clicks(state),
                state,
                status_text(state),
                None,
                video_dir,
            )

        def on_select(evt: gr.SelectData, state):
            state = dict(state)
            x, y = evt.index[0], evt.index[1]
            state["current"] = list(state["current"]) + [[float(x), float(y)]]
            return render_clicks(state), state, status_text(state)

        def on_next(state):
            state, msg = finish_current_object(state)
            return render_clicks(state), state, msg

        def on_reset(video_dir):
            state = new_click_state(load_first_frame(video_dir))
            return render_clicks(state), state, status_text(state), None

        def on_run(state, video_dir, video_name, progress=gr.Progress(track_tqdm=False)):
            def progress_fn(frac, desc=""):
                progress(frac, desc=desc)

            save_dir = os.path.join(args.save_dir, video_name)
            mp4_path = run_tracking(
                video_dir,
                save_dir,
                args.checkpoint,
                args.config,
                args.device,
                state,
                fps=args.fps,
                progress_fn=progress_fn,
            )
            return mp4_path

        video_select.change(
            on_video_change,
            inputs=[video_select],
            outputs=[canvas, click_state, status, output_video, video_dir_state],
        )
        canvas.select(on_select, inputs=[click_state], outputs=[canvas, click_state, status])
        btn_next.click(on_next, inputs=[click_state], outputs=[canvas, click_state, status])
        btn_reset.click(
            on_reset,
            inputs=[video_dir_state],
            outputs=[canvas, click_state, status, output_video],
        )
        btn_run.click(
            on_run,
            inputs=[click_state, video_dir_state, video_select],
            outputs=[output_video],
        )

    return demo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos_root", default="demo/video")
    p.add_argument("--checkpoint", default="checkpoints/sam-mt.pt")
    p.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--save_dir", default="outputs")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_root = os.path.abspath(args.save_dir)
    demo = build_demo(args)
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[save_root],
    )