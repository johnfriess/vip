"""Standalone script to visualize a trained model snapshot against a dataset.

Usage:
    python -m vip.visualize_snapshot \
        --snapshot path/to/snapshot_100000.pt \
        --datapath /data/siddhant/vip/vis_data/kitchen-complete-v0 \
        --dataset kitchen-image \
        --model vip \
        --output vip_trajectory_visualization.png

    # Visualize per-subtask value functions for kitchen:
    python -m vip.visualize_snapshot \
        --snapshot path/to/snapshot.pt \
        --datapath /data/siddhant/vip/vis_data/kitchen-complete-v0 \
        --dataset kitchen-image \
        --model derail \
        --subtasks
"""
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import hydra
from vip.utils import utils
from vip.utils.data_loaders import STATE_DATASETS
from vip.models.model_derail import StateMultilinearDERAIL, StateEuclideanDERAIL

# Subtask frame ranges detected from D4RL/kitchen/complete-v2 episodes.
# (start_frame, goal_frame) — each subtask starts where the previous one ends.
KITCHEN_SUBTASK_FRAMES = {
    'microwave': (0, 40),
    'kettle': (40, 88),
    'light switch': (88, 155),
    'slide cabinet': (155, 200),
}


def compute_values_for_goal_idx(model, model_type, traj, goal_idx, device):
    """Compute V(s_t, g) for all states in traj, where g = traj[goal_idx]."""
    traj_t = traj.to(device)
    with torch.no_grad():
        if isinstance(model.module, StateMultilinearDERAIL):
            goal = traj_t[goal_idx]
            g_rep = goal.unsqueeze(0).expand(traj_t.shape[0], -1)
            h1, h2, h3 = model(traj_t, g_rep)
            return model.module.value(h1, h2, h3).cpu()
        else:
            etraj = model(traj_t)
            eg = etraj[goal_idx]
            return model.module.sim(etraj, eg).cpu()


def load_image_frames(datapath, datasource, traj_idx=0):
    """Load image frames for a trajectory from disk."""
    import glob
    import torchvision
    if datasource == 'kitchen-image':
        traj_dir = sorted(glob.glob(os.path.join(datapath, 'traj*')))[traj_idx]
        num_frames = len(glob.glob(os.path.join(traj_dir, 'img*.png')))
        frames = []
        for i in range(num_frames):
            frames.append(torchvision.io.read_image(os.path.join(traj_dir, 'img%d.png' % i)))
        return torch.stack(frames).float()
    return None


def visualize_subtasks(model, model_type, datapath, datasource, device, output_dir,
                       num_sampled_frames=12):
    """Generate one frames+value plot per kitchen subtask, matching the --frames style."""
    import glob

    # Pick a random trajectory
    traj_dirs = sorted(glob.glob(os.path.join(datapath, 'traj*')))
    traj_idx = np.random.randint(0, len(traj_dirs))

    # Load image frames for display
    image_frames = load_image_frames(datapath, datasource, traj_idx)
    traj_len = image_frames.shape[0] if image_frames is not None else 0

    # If state-based, also load state trajectory for the model
    if datasource in STATE_DATASETS:
        buffer = utils.create_buffer(
            model_type=model_type,
            datasource=datasource,
            datapath=datapath,
            num_workers=0,
            doaug=0,
        )
        state_traj = buffer.get_trajectory().to(device)
        traj_len = state_traj.shape[0]
    else:
        # Image-based: feed images through the model
        state_traj = image_frames

    output_paths = []
    for subtask_name, (start_frame, goal_frame) in KITCHEN_SUBTASK_FRAMES.items():
        start_idx = min(start_frame, traj_len - 1)
        goal_idx = min(goal_frame, traj_len - 1)

        values = compute_values_for_goal_idx(model, model_type, state_traj, goal_idx, device)
        if values is None:
            continue
        values_np = values[start_idx:goal_idx + 1].numpy()
        num_frames_in_range = goal_idx - start_idx + 1

        # Pick evenly-spaced frame indices within the subtask range
        frame_indices = np.linspace(start_idx, goal_idx, num_sampled_frames, dtype=int)

        if image_frames is not None:
            # Frames + value curve layout (matching reference image style)
            fig = plt.figure(figsize=(14, 6))
            gs = fig.add_gridspec(2, num_sampled_frames, height_ratios=[1, 2], hspace=0.3)

            for i, fidx in enumerate(frame_indices):
                ax_img = fig.add_subplot(gs[0, i])
                frame = image_frames[fidx].permute(1, 2, 0).numpy()
                if frame.max() > 2.0:
                    frame = frame / 255.0
                ax_img.imshow(np.clip(frame, 0, 1))
                ax_img.set_title('f%d' % fidx, fontsize=8)
                ax_img.axis('off')

            ax_val = fig.add_subplot(gs[1, :])
        else:
            # State-based: just the value curve
            fig, ax_val = plt.subplots(figsize=(10, 4))

        ax_val.plot(range(start_idx, goal_idx + 1), values_np, linewidth=1.5)

        for fidx in frame_indices:
            ax_val.axvline(x=fidx, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
            ax_val.plot(fidx, values[fidx].item(), 'o', color='red', markersize=5)

        ax_val.axvline(x=goal_idx, color='r', linestyle='--', alpha=0.7,
                       label='goal @ %d' % goal_idx)
        ax_val.set_xlabel("Frame")
        ax_val.set_ylabel("%s Value" % model_type.upper())
        ax_val.set_title("%s Value -> %s goal (frames %d-%d)" % (model_type.upper(), subtask_name, start_idx, goal_idx))
        ax_val.legend()

        safe_name = subtask_name.replace(' ', '_')
        out_path = os.path.join(output_dir, '%s_subtask_%s.png' % (model_type, safe_name))
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        output_paths.append(out_path)
        print('Saved %s' % out_path)

    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Visualize a trained model snapshot")
    parser.add_argument("--snapshot", type=str, required=True, help="Path to snapshot .pt file")
    parser.add_argument("--datapath", type=str, default=None, help="Path to dataset for visualization")
    parser.add_argument("--dataset", type=str, default="kitchen-image", help="Dataset name")
    parser.add_argument("--model", type=str, default="vip", help="Model type (vip, derail, iql)")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--frame-stack", type=int, default=1, help="Frame stack")
    parser.add_argument("--goal-fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0],
                        help="Goal positions as fractions of trajectory length")
    parser.add_argument("--goal-frac", type=float, default=0.25,
                        help="Single goal fraction for --frames mode")
    parser.add_argument("--num-frames", type=int, default=12,
                        help="Number of frames to show in --frames mode")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load snapshot
    print("Loading snapshot from %s" % args.snapshot)
    payload = torch.load(args.snapshot, map_location=device)
    model_cfg = OmegaConf.create(payload["model_cfg"])
    if "device" in model_cfg:
        model_cfg.device = args.device
    OmegaConf.resolve(model_cfg)

    # Reconstruct model
    model = hydra.utils.instantiate(model_cfg)
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    snapshot_dir = os.path.dirname(os.path.abspath(args.snapshot))
    dataset_name = os.path.basename(args.datapath) if args.datapath else args.dataset.replace("/", "_")

    assert args.datapath is not None, "--datapath is required"
    print("Loading data from %s" % args.datapath)
    buffer = utils.create_buffer(
        model_type=args.model,
        datasource=args.dataset,
        datapath=args.datapath,
        num_workers=4,
        doaug=0,
        use_achieved_goal=payload.get("use_achieved_goal", False),
        frame_stack=args.frame_stack,
    )

    # 1. Full trajectory
    output_path = os.path.join(
        snapshot_dir, "%s_trajectory_%s.png" % (args.model, dataset_name))
    utils.visualize_trajectory(
        model=model,
        model_type=args.model,
        datasource=args.dataset,
        buffer=buffer,
        device=device,
        datapath=args.datapath,
        output_path=output_path,
    )
    print("Saved %s" % output_path)

    # 2. Frames + value for each goal fraction
    for frac in args.goal_fractions:
        output_path = os.path.join(
            snapshot_dir, "%s_frames_%dpct_%s.png" % (args.model, int(frac*100), dataset_name))
        utils.visualize_trajectory_with_frames(
            model=model,
            model_type=args.model,
            datasource=args.dataset,
            buffer=buffer,
            device=device,
            goal_fraction=frac,
            num_sampled_frames=args.num_frames,
            datapath=args.datapath,
            output_path=output_path,
        )
        print("Saved %s" % output_path)

    # 3. Per-subtask visualizations
    visualize_subtasks(
        model=model,
        model_type=args.model,
        datapath=args.datapath,
        datasource=args.dataset,
        device=device,
        output_dir=snapshot_dir,
        num_sampled_frames=args.num_frames,
    )


if __name__ == "__main__":
    main()
