"""
This implements a shooting trajectory optimization algorithm.
The closest known algorithm is perhaps MPPI and hence we stick to that terminology.
Uses a filtered action sequence to generate smooth motions.
"""

import numpy as np
from mjrl.utils.logger import DataLog

from trajopt.algos.trajopt_base import Trajectory
from trajopt.utils.utils import gather_paths_parallel

class MPPI(Trajectory):
    def __init__(self, env, H, paths_per_cpu,
                 num_cpu=1,
                 kappa=1.0,
                 gamma=1.0,
                 mean=None,
                 filter_coefs=None,
                 default_act='repeat',
                 warmstart=True,
                 seed=123,
                 env_kwargs=None
                 ):
        self.env, self.seed = env, seed
        self.env_kwargs = env_kwargs
        self.n, self.m = env.observation_dim, env.action_dim
        self.H, self.paths_per_cpu, self.num_cpu = H, paths_per_cpu, num_cpu
        self.warmstart = warmstart

        self.logger = DataLog() 
        
        self.mean, self.filter_coefs, self.kappa, self.gamma = mean, filter_coefs, kappa, gamma
        if mean is None:
            self.mean = np.zeros(self.m)
        if filter_coefs is None:
            self.filter_coefs = [np.ones(self.m), 1.0, 0.0, 0.0]
        self.default_act = default_act

        self.sol_state = []
        self.sol_act = []
        self.sol_reward = []
        self.sol_obs = []
        self.sol_embedding = [] 
        self.sol_info = [] 

        self.env.reset()
        self.env.set_seed(seed)
        self.env.reset(seed=seed)
        self.sol_state.append(self.env.get_env_state().copy())
        self.sol_obs.append(self.env.get_obs())
        self.act_sequence = np.ones((self.H, self.m)) * self.mean
        self.init_act_sequence = self.act_sequence.copy()

        if env_kwargs['embedding_reward']:
            self.sol_embedding.append(self.env.env.get_views(embedding=True))

    def update(self, paths):
        num_traj = len(paths)
        act = np.array([paths[i]["actions"] for i in range(num_traj)])
        R = self.score_trajectory(paths)
        S = np.exp(self.kappa*(R-np.max(R)))

        # blend the action sequence
        weighted_seq = S*act.T
        act_sequence = np.sum(weighted_seq.T, axis=0)/(np.sum(S) + 1e-6)
        self.act_sequence = act_sequence

    def advance_time(self, act_sequence=None):
        act_sequence = self.act_sequence if act_sequence is None else act_sequence
        # accept first action and step
        action = act_sequence[0].copy()
        self.env.real_env_step(True)
        s, r, _, info = self.env.step(action)

        self.sol_act.append(action)
        self.sol_state.append(self.env.get_env_state().copy())
        self.sol_obs.append(self.env.get_obs())
        self.sol_reward.append(r)
        self.sol_info.append(info)

        if 'obs_embedding' in info:
            self.sol_embedding.append(self.env.env.get_views(embedding=True))    
        
        # get updated action sequence
        if self.warmstart:
            self.act_sequence[:-1] = act_sequence[1:]
            if self.default_act == 'repeat':
                self.act_sequence[-1] = self.act_sequence[-2]
            else:
                self.act_sequence[-1] = self.mean.copy()
        else:
            self.act_sequence = self.init_act_sequence.copy()

    def score_trajectory(self, paths):
        scores = np.zeros(len(paths))
        
        for i in range(len(paths)):
            scores[i] = paths[i]["rewards"][-1] # -V(s_T;g)
        return scores

    def do_rollouts(self, seed):
        paths = gather_paths_parallel(self.env,
                                      self.sol_state[-1],
                                      self.act_sequence,
                                      self.filter_coefs,
                                      seed,
                                      self.paths_per_cpu,
                                      self.num_cpu,
                                      env_kwargs=self.env_kwargs
                                      )
        return paths

    def train_step(self, niter=1):
        t = len(self.sol_state) - 1
        for _ in range(niter):
            paths = self.do_rollouts(self.seed+t)
            self.update(paths)
        self.advance_time()
        return paths

    def render_planning_trajectories(self, paths, camera_name=None, n_show=None):
        """
        Render sampled planning trajectories as a grid GIF with scores.
        Uses the same generate_frame() as animate_result_offscreen.
        Sorted best (top-left) to worst (bottom-right).
        """
        from trajopt.algos.trajopt_base import generate_frame
        from PIL import Image, ImageDraw

        scores = self.score_trajectory(paths)
        sorted_indices = np.argsort(scores)[::-1]  # best first

        if n_show is None:
            n_show = len(paths)
        n_show = min(n_show, len(paths))
        selected = sorted_indices[:n_show]

        # Grid layout: arrange n_show frames into rows and columns
        n_cols = int(np.ceil(np.sqrt(n_show)))
        n_rows = int(np.ceil(n_show / n_cols))

        H = len(paths[0]["states"])
        fw, fh = 256, 256

        # Save env state so we can restore after rendering
        saved_state = self.env.get_env_state().copy()
        self.env.real_env_step(False)

        grid_frames = []
        for t in range(H):
            grid = Image.new('RGB', (n_cols * fw, n_rows * fh), color=(0, 0, 0))
            draw = ImageDraw.Draw(grid)

            for k, idx in enumerate(selected):
                self.env.set_env_state(paths[idx]["states"][t])
                frame = generate_frame(self.env, frame_size=(fw, fh),
                                       camera_name=camera_name)

                row, col = k // n_cols, k % n_cols
                grid.paste(Image.fromarray(frame), (col * fw, row * fh))

                # Draw score text with black outline for readability
                rank = k + 1
                text = f"#{rank} V={scores[idx]:.3f}"
                tx, ty = col * fw + 4, row * fh + 4
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    draw.text((tx + dx, ty + dy), text, fill=(0, 0, 0))
                draw.text((tx, ty), text, fill=(255, 255, 0))

            grid_frames.append(np.array(grid))

        # Restore env to where it was
        self.env.set_env_state(saved_state)
        self.env.real_env_step(True)

        return grid_frames, scores
