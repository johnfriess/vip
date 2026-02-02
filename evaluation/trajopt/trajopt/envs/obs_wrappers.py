# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import namedtuple
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import glob
import omegaconf
import torch
import torch.nn as nn
from torch.nn.modules.linear import Identity
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from PIL import Image
from pathlib import Path
import pickle
from torchvision.utils import save_image
import hydra
import os
import sys
import minari
from trajopt.envs.gym_env import GymEnv


def extract_achieved_goal(obs):
    """Extract 11D achieved_goal from 59D observation.

    FrankaKitchen achieved_goal contains:
        - kettle (7D): obs[32:39] - position + quaternion
        - light switch (2D): obs[26:28]
        - microwave (1D): obs[31]
        - slide cabinet (1D): obs[28]

    Returns flattened array in order: [kettle(7), light(2), microwave(1), slide(1)]
    """
    return np.concatenate([
        obs[32:39],  # kettle (7D)
        obs[26:28],  # light switch (2D)
        obs[31:32],  # microwave (1D)
        obs[28:29],  # slide cabinet (1D)
    ])


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def _get_embedding(embedding_name='resnet50', load_path="", *args, **kwargs):
    if load_path == "random":
        prt = False
    else:
        prt = True
    if embedding_name == 'resnet34':
        model = models.resnet34(pretrained=prt, progress=False)
        embedding_dim = 512
    elif embedding_name == 'resnet18':
        model = models.resnet18(pretrained=prt, progress=False)
        embedding_dim = 512
    elif embedding_name == 'resnet50':
        model = models.resnet50(pretrained=prt, progress=False)
        embedding_dim = 2048
    else:
        print("Requested model not available currently")
        raise NotImplementedError
    # make FC layers to be identity
    # NOTE: This works for ResNet backbones but should check if same
    # template applies to other backbone architectures
    model.fc = Identity()
    model = model.eval()
    return model, embedding_dim

def env_constructor(env_name, device='cuda', image_width=256, image_height=256,
                    camera_name=None, embedding_name='resnet50', pixel_based=True,
                    embedding_reward=True,
                    render_gpu_id=0, load_path="", proprio=False, goal_timestep=-1, init_timestep=0,
                    goal_episode=0, use_minari=False):
    # print("Constructing environment with GPU", render_gpu_id)
    if not pixel_based and not embedding_reward and not use_minari:
        env = GymEnv(env_name)
    else:
        if use_minari:
            # Load Minari dataset and recover environment
            minari_dataset = minari.load_dataset(env_name, download=True)
            env = minari_dataset.recover_environment(
                render_mode='rgb_array',
                robot_noise_ratio=0.0,
                object_noise_ratio=0.0
            )
        else:
            env = gym.make(env_name)
        ## Wrap in pixel observation wrapper
        env = MuJoCoPixelObs(env, width=image_width, height=image_height,
                            camera_name=camera_name, device_id=render_gpu_id, use_minari=use_minari)
        ## Wrapper which encodes state in pretrained model (additionally compute reward)
        env = StateEmbedding(env, embedding_name=embedding_name, device=device, load_path=load_path,
                        proprio=proprio, camera_name=camera_name,
                            env_name=env_name, pixel_based=pixel_based,
                            embedding_reward=embedding_reward,
                            goal_timestep=goal_timestep, init_timestep=init_timestep,
                            goal_episode=goal_episode,
                            use_minari=use_minari)
        env = GymEnv(env)
    return env

class ClipEnc(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, im):
        e = self.m.encode_image(im)
        return e


class StateEmbedding(gym.ObservationWrapper):
    """
    This wrapper places a convolution model over the observation.

    From https://pytorch.org/vision/stable/models.html
    All pre-trained models expect input images normalized in the same way,
    i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
    where H and W are expected to be at least 224.

    Args:
        env (Gym environment): the original environment,
        embedding_name (str, 'baseline'): the name of the convolution model,
        device (str, 'cuda'): where to allocate the model.

    """
    def __init__(self, env, embedding_name=None, device='cuda', load_path="", checkpoint="",
    proprio=0,
     camera_name=None, env_name=None, pixel_based=True, embedding_reward=False,
      goal_timestep=-1, init_timestep=0, goal_episode=0, use_minari=False):
        gym.ObservationWrapper.__init__(self, env)

        self.env_name = env_name 
        self.cameras = [camera_name]
        self.camera_name = self.cameras[0]

        self.proprio = proprio
        self.load_path = load_path
        self.start_finetune = False
        self.use_minari = use_minari  # Flag for Minari dataset (59D obs)

        self.state_based_reward = False
        self.multilinear = False
        self.use_achieved_goal = False
        self.is_iql = False
        self.goal_state = None

        if "state_vip:" in load_path or "state_euclidean_derail:" in load_path or "state_multilinear_derail:" in load_path or "state_iql:" in load_path:
            model_name, checkpoint_path = load_path.split(":", 1)
            print(f"Loading {model_name} from {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model_cfg = {k: v for k, v in checkpoint['model_cfg'].items() if not k.startswith('_')}
            self.use_achieved_goal = checkpoint.get('use_achieved_goal', False)

            if model_name == "state_vip":
                from vip.models.model_vip import StateVIP
                embedding = StateVIP(**model_cfg)
            elif model_name == "state_euclidean_derail":
                from vip.models.model_derail import StateEuclideanDERAIL
                embedding = StateEuclideanDERAIL(**model_cfg)
            elif model_name == "state_multilinear_derail":
                from vip.models.model_derail import StateMultilinearDERAIL
                embedding = StateMultilinearDERAIL(**model_cfg)
                self.multilinear = True
            elif model_name == "state_iql":
                from vip.models.model_iql import StateIQL
                embedding = StateIQL(**model_cfg)
                self.is_iql = True

            embedding.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model'].items()})
            embedding.eval()
            embedding = torch.nn.DataParallel(embedding)
            embedding_dim = embedding.module.hidden_dim
            self.transforms = None
            self.state_based_reward = True
        elif "vip" in load_path:
            print(f"Loading pre-trained {load_path} model!")
            from vip import load_vip
            rep = load_vip()
            rep.eval()
            embedding_dim = rep.module.hidden_dim
            embedding = rep
            self.transforms = T.Compose([T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor()]) # ToTensor() divides by 255
        elif "r3m" in load_path:
            print(f"Loading pre-trained {load_path} model!")
            from r3m import load_r3m_reproduce
            rep = load_r3m_reproduce(load_path)
            rep.eval()
            embedding_dim = rep.module.outdim
            embedding = rep
            self.transforms = T.Compose([T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor()]) # ToTensor() divides by 255
        elif load_path == "clip":
            import clip
            model, cliptransforms = clip.load("RN50", device="cuda")
            embedding = ClipEnc(model)
            embedding.eval()
            embedding_dim = 1024
            self.transforms = cliptransforms
        elif (load_path == "random") or (load_path == "resnet"):
                embedding, embedding_dim = _get_embedding(load_path=load_path)
                self.transforms = T.Compose([T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor(), # ToTensor() divides by 255
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            raise NameError("Invalid Model")
        embedding.eval()

        if device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.device = device
        embedding.to(device=device)

        self.embedding, self.embedding_dim = embedding, embedding_dim
        self.pixel_based = pixel_based
        self.embedding_reward = embedding_reward 
        self.init_state = None
        if self.pixel_based:
            self.observation_space = Box(
                        low=-np.inf, high=np.inf, shape=(self.embedding_dim+self.proprio,))
        elif self.state_based_reward:
            # Full 59D state observation (matches VIP training on Minari)
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(59,))
        else:
            self.observation_space = self.env.unwrapped.observation_space

        if self.embedding_reward: 
            self.init_timestep = init_timestep
            self.goal_timestep = goal_timestep

            self.goal_robot_pose = None
            self.goal_object_pose = None
            self.goal_end_effector = None

            if not self.state_based_reward:
                # evaluation information
                from trajopt import DEMO_PATHS
                demopath = DEMO_PATHS[self.env_name] 
                demo_id = demopath[-1] 
                traj_path = demopath[:-1] + f'traj_{demo_id}.pickle'
                demo = pickle.load(open(traj_path, 'rb'))
                self.goal_robot_pose = demo.sol_info[self.goal_timestep]['obs_dict']['robot_jnt']
                self.goal_object_pose = demo.sol_info[self.goal_timestep]['obs_dict']['objs_jnt']
                self.goal_end_effector = demo.sol_info[self.goal_timestep]['obs_dict']['end_effector']
    
            self.goal_embedding = {} 
            for camera in self.cameras:
                if self.state_based_reward:
                    minari_dataset = minari.load_dataset(env_name, download=True)
                    episodes = list(minari_dataset.iterate_episodes())

                    if goal_episode >= len(episodes):
                        goal_episode = 0

                    episode = episodes[goal_episode]
                    obs = episode.observations['observation']  # (T+1, 59)
                    ep_length = len(obs)
                    print(f"Minari Episode {goal_episode}: {ep_length} timesteps")

                    if goal_timestep < 0:
                        actual_goal_idx = ep_length + goal_timestep
                    else:
                        actual_goal_idx = min(goal_timestep, ep_length - 1)
                    print(f"  Goal: timestep {goal_timestep} -> index {actual_goal_idx}")

                    if init_timestep < 0:
                        actual_init_idx = ep_length + init_timestep
                    else:
                        actual_init_idx = min(init_timestep, ep_length - 1)
                    print(f"  Init: timestep {init_timestep} -> index {actual_init_idx}")
                    
                    robot_qpos = obs[actual_init_idx][:9]
                    robot_qvel = obs[actual_init_idx][9:18]
                    obj_qpos   = obs[actual_init_idx][18:39]
                    obj_qvel   = obs[actual_init_idx][39:]
                    self.init_state = {
                        "qpos": np.concatenate([robot_qpos, obj_qpos]),
                        "qvel": np.concatenate([robot_qvel, obj_qvel])
                    }

                    goal_state = obs[actual_goal_idx].astype(np.float32)
                    if self.multilinear or self.is_iql:
                        if self.use_achieved_goal:
                            self.goal_state = extract_achieved_goal(goal_state)
                        else:
                            self.goal_state = goal_state
                    self.goal_embedding[camera] = self.observation(goal_state)
                else:
                    # mj_envs MPPI demo for goal embedding 
                    if init_timestep != 0:
                        self.init_state = {}
                        for key in demo.sol_state[init_timestep]:
                            self.init_state[key] = demo.sol_state[init_timestep][key]
                        self.init_state['env_timestep'] = init_timestep + 1 

                    video_paths = [demopath + f'/{camera}']
                    num_vid = len(video_paths)
                    end_goals = [] 
                    for i in range(num_vid):
                        vid = f"{video_paths[i]}"
                        img = Image.open(f"{vid}/{self.goal_timestep}.png")
                        cur_dir = os.getcwd() 
                        img.save(f"{cur_dir}/goal_image_{camera}.png") # save goal image
                        end_goals.append(img)
                    
                    # hack to get when there is only one goal image working
                    if len(end_goals) == 1:
                        end_goals.append(end_goals[-1])
                    
                    goal_embedding = self.encode_batch(end_goals)
                    self.goal_embedding[camera] = goal_embedding.mean(axis=0) 

    def observation(self, observation):
        if self.state_based_reward:
            if self.use_achieved_goal:
                observation = extract_achieved_goal(observation)
            state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                if self.multilinear:
                    emb = self.embedding.module.encoder1(state_tensor).cpu().numpy().squeeze()
                elif self.is_iql:
                    goal_tensor = torch.tensor(self.goal_state, dtype=torch.float32).unsqueeze(0).to(self.device)
                    sg = torch.cat([state_tensor, goal_tensor], dim=-1)
                    emb = self.embedding.module.v_encoder(sg).cpu().numpy().squeeze()
                else:
                    emb = self.embedding(state_tensor).cpu().numpy().squeeze()
            return emb
        ### INPUT SHOULD BE [0,255]
        if self.embedding is not None and len(observation.shape) > 1:
            if isinstance(observation, np.ndarray):
                o = Image.fromarray(observation.astype(np.uint8))

            inp = self.transforms(o).reshape(-1, 3, 224, 224)
            if  "vip" in self.load_path or "r3m" in self.load_path:
                inp *= 255.0
            inp = inp.to(self.device)

            with torch.no_grad():                
                emb = self.embedding(inp).view(-1, self.embedding_dim).to('cpu').numpy().squeeze()

            ## IF proprioception add it to end of embedding
            if self.proprio:
                try:
                    proprio = self.env.unwrapped.get_obs()[:self.proprio]
                except:
                    proprio = self.env.unwrapped._get_obs()[:self.proprio]
                emb = np.concatenate([emb, proprio])

            return emb
        else:
            return observation

    def encode_batch(self, obs, finetune=False):
        ### INPUT SHOULD BE [0,255]
        inp = []
        for o in obs:
            if isinstance(o, np.ndarray):
                o = Image.fromarray(o.astype(np.uint8))
            o = self.transforms(o).reshape(-1, 3, 224, 224)
            if "vip" in self.load_path or "r3m" in self.load_path:
                o *= 255.0
            inp.append(o)

        inp = torch.cat(inp)
        inp = inp.to(self.device)
        if finetune and self.start_finetune:
            emb = self.embedding(inp).view(-1, self.embedding_dim)
        else:
            with torch.no_grad():
                emb = self.embedding(inp).view(-1, self.embedding_dim).to('cpu').numpy().squeeze()
        return emb

    def get_obs(self):
        if self.embedding is not None and self.pixel_based:
            return self.observation(self.env.observation(None))
        elif self.state_based_reward:
            robot_obs = self.env.unwrapped.robot_env._get_obs()
            return self.env.unwrapped._get_obs(robot_obs)["observation"]
        else:
            return self.env.unwrapped.get_obs()
    def get_views(self, embedding=False):
        views = {}
        embeddings = {}
        for camera in self.cameras:
            view = self.env.get_image(camera_name=camera)
            views[camera] = view
            if embedding:
                if self.state_based_reward:
                    embeddings[camera] = self.observation(self.get_obs())
                else:
                    embeddings[camera] = self.observation(view)
        if embedding:
            return embeddings 
        return views  

    def start_finetuning(self):
        self.start_finetune = True
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        obs_embedding = self.observation(observation)
        info['obs_embedding'] = obs_embedding
        if self.embedding_reward:
            rewards = []
            for camera in self.cameras:
                if self.state_based_reward:
                    if self.multilinear:
                        reward_camera = self._compute_multilinear_value(observation)
                    elif self.is_iql:
                        reward_camera = self._compute_iql_value(observation)
                    else:
                        # VIP/Euclidean DERAIL: embedding distance
                        reward_camera = -np.linalg.norm(obs_embedding-self.goal_embedding[camera])
                else:
                    img_camera = self.env.get_image(camera_name=camera)
                    obs_embedding_camera = self.observation(img_camera)
                    obs_embedding_camera = obs_embedding_camera if self.proprio == 0 else obs_embedding_camera[:-self.proprio]
                    reward_camera = -np.linalg.norm(obs_embedding_camera-self.goal_embedding[camera])
                rewards.append(reward_camera) 

            if 'obs_dict' in info and self.goal_end_effector is not None:
                if 'end_effector' in info['obs_dict']:
                    info['obs_dict']['ee_error'] = np.linalg.norm(self.goal_end_effector-info['obs_dict']['end_effector'])
                if 'hand_jnt' in info['obs_dict']:
                    info['obs_dict']['robot_error'] = np.linalg.norm(self.goal_robot_pose-info['obs_dict']['hand_jnt'])
                elif 'robot_jnt' in info['obs_dict']:
                    info['obs_dict']['robot_error'] = np.linalg.norm(self.goal_robot_pose-info['obs_dict']['robot_jnt'])
                if 'objs_jnt' in info['obs_dict']:
                    info['obs_dict']['objs_error'] = np.linalg.norm(self.goal_object_pose-info['obs_dict']['objs_jnt'])
            
            reward = min(rewards)
        if not self.pixel_based:
            state = self.get_obs()
        else: 
            state = obs_embedding 
        return state, reward, terminated, truncated, info

    def _compute_multilinear_value(self, observation):
        """Compute multilinear value: V(s, g) = φ1(s) @ M(g) @ φ3(g)"""
        if self.use_achieved_goal:
            observation = extract_achieved_goal(observation)
        state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        goal_tensor = torch.tensor(self.goal_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            h1, h2, h3 = self.embedding.module(state_tensor, goal_tensor)
            value = self.embedding.module.value(h1, h2, h3).cpu().numpy().squeeze()
        return float(value)

    def _compute_iql_value(self, observation):
        """Compute IQL value: V(s || g) using v_value"""
        if self.use_achieved_goal:
            observation = extract_achieved_goal(observation)
        state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        goal_tensor = torch.tensor(self.goal_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        sg = torch.cat([state_tensor, goal_tensor], dim=-1)
        with torch.no_grad():
            value = self.embedding.module.v_value(sg).cpu().numpy().squeeze()
        return float(value)

    def reset(self, seed):
        observation, _ = self.env.reset(seed=seed)
        try:
            if self.init_state is not None:
                self.set_env_state(self.init_state)
        except Exception as e:
            print("Resetting Initial State Error")
            print("Unexpected error:", sys.exc_info()[0])
            print(e)
        if not self.pixel_based:
            observation = self.get_obs()
        else:
            observation = self.observation(observation) # This is needed for IL, but would it break other evaluations?
        return observation

    def get_env_state(self):
        """Get environment state for MPPI rollouts.

        Returns qpos/qvel from MuJoCo data. Note: the 59D observation
        is NOT the same as qpos/qvel (observation includes noise and
        has different structure).
        """
        if not self.state_based_reward:
            return self.env.get_env_state()
            
        return {
            'qpos': self.env.unwrapped.data.qpos.copy(),
            'qvel': self.env.unwrapped.data.qvel.copy(),
        }

    def set_env_state(self, state_dict):
        """Set environment state for MPPI rollouts.

        Directly sets qpos/qvel on MuJoCo data and forwards simulation.
        """
        if not self.state_based_reward:
            self.env.set_env_state(state_dict)
            return

        qpos = state_dict["qpos"]
        qvel = state_dict["qvel"]
        self.env.unwrapped.robot_env.set_state(qpos, qvel)
        
        # import mujoco
        # self.env.unwrapped.data.qpos[:] = state_dict['qpos']
        # self.env.unwrapped.data.qvel[:] = state_dict['qvel']
        # mujoco.mj_forward(self.env.unwrapped.model, self.env.unwrapped.data)

class MuJoCoPixelObs(gym.ObservationWrapper):
    def __init__(self, env, width, height, camera_name, device_id=-1, depth=False, use_minari=False, *args, **kwargs):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(low=0., high=255., shape=(3, width, height))
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.depth = depth
        self.device_id = device_id
        self.use_minari = use_minari

    def get_image(self, camera_name=None):
        if camera_name is None:
            camera_name = self.camera_name

        if self.use_minari:
            # Gymnasium kitchen render
            img = self.env.render()
            return img

        # mj_envs uses mujoco_py rendering
        if camera_name == "default" or camera_name == "all":
            img = self.sim.render(width=self.width, height=self.height, depth=self.depth,
             device_id=self.device_id)
        else:
            img = self.sim.render(width=self.width, height=self.height, depth=self.depth,
                            camera_name=camera_name, device_id=self.device_id)
        img = img[::-1,:,:]
        return img

    def observation(self, observation=None):
        # This function creates observations based on the current state of the environment.
        # Argument `observation` is ignored, but `gym.ObservationWrapper` requires it.
        return self.get_image() if not self.use_minari else observation["observation"]
        