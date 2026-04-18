import numpy as np
from torch.utils.data import IterableDataset
import d4rl
import pickle

def get_frame_stack_idx(idx, frame_stack=4):
    # If idx - i is less than 0, repeat the first frame

    return [idx - i if idx - i >= 0 else 0 for i in range(frame_stack)]

def get_stacked_state(traj_idx, idx, transform, dataset_paths, frame_stack=4):
    # Get the indices of the frames to stack
    frame_stack_idx = get_frame_stack_idx(idx, frame_stack)

    # Load the frames
    if type(dataset_paths) == str:
        frames = [Image.open(os.path.join(dataset_paths, f'img{frame_idx}.png')) for frame_idx in frame_stack_idx]
    else:
        frames = [Image.open(os.path.join(dataset_paths[traj_idx], f'img{frame_idx}.png')) for frame_idx in frame_stack_idx]

    # Apply the transform to each frame
    stacked_state = torch.stack([transform(frame) for frame in frames])

    return stacked_state

class D4RLBuffer(IterableDataset):
    def __init__(self, env, expert_env, num_workers=10, env_name='', algo = ''):
        # import pdb; pdb.set_trace()
        self._num_workers = max(1, num_workers)

        self.env = env
        self.env_name = env_name
        self.algo_name = algo
        # self.goal_indices = goal_indices

        if 'kitchen' in env_name:
            self.obs_dim = 30
        else:
            self.obs_dim = env.observation_space.shape[0]
        # self.obs_dim = 30
        print("Observation Dimension", self.obs_dim)

        # self.dataset = env.get_dataset()
        # self.expert_dataset = expert_env.get_dataset()
        
        if 'antmaze' in env_name:
            self.dataset = d4rl.qlearning_dataset(env)
        else:
            self.dataset = env.get_dataset()
        
        self.expert_dataset = d4rl.qlearning_dataset(expert_env)
        # import ipdb;ipdb.set_trace()
        self.expert_trajectories = self.expert_dataset['observations']
        self.expert_actions = self.expert_dataset['actions']
        # print(self.expert_trajectories)
        self.expert_terminals = np.where(self.expert_dataset['terminals'])[0]
        # print(self.expert_terminals)
        print("Expert Trajectories")
        # print(self.expert_trajectories.shape)
        # self.expert_terminals = np.where(expert_env.get_dataset()['terminals'])[0]
        print("Expert Terminals")
        
        # import ipdb;ipdb.set_trace()
        if 'kitchen' in env_name:
            self.eval_trajectory = self.expert_trajectories[0:self.expert_terminals[0], :30]
            self.random_trajectory = self.dataset['observations'][0:1000,:30]
        elif 'antmaze-large-diverse' in env_name:
            with open('/data/harshit_sikchi/work/vip_d4rl/antmaze-large-diverse.pickle', 'rb') as f:
                data = pickle.load(f)
            self.eval_trajectory = np.array(data['observations'][0:999])
            self.random_trajectory = self.dataset['observations'][0:999]
        else:
            self.eval_trajectory = self.expert_trajectories[0:999]
            self.random_trajectory = self.dataset['observations'][0:999]
        # self.eval_trajectory = self.expert_dataset['observations'][:700]
        print("Expert Trajectory")
        # print(self.eval_trajectory.shape)

        self.traj_end_idx = None
        #self._find_trajectory_idx()
        self.find_trajectory_indices()

    def _find_trajectory_idx(self):
        # Timeouts - episode ended due to max time limit
        # Terminals - episode ended due to goal reached
        print("Finding Trajectory Indices")
        self.traj_start_idx=[0]
        if 'antmaze' in self.env_name:
            dones_float = np.zeros_like(self.dataset['rewards'])
            for i in range(len(dones_float) - 1):
                if np.linalg.norm(self.dataset['observations'][i + 1] -
                                self.dataset['next_observations'][i]
                                ) > 1e-6 or self.dataset['terminals'][i] == 1.0:
                    dones_float[i] = 1
                else:
                    dones_float[i] = 0
                if dones_float[i]==1:
                    # if (i+1) - self.traj_start_idx[-1]==1000:
                    self.traj_start_idx.append(i+1)

            self.corrected_traj_start_idx = []
            for i in range(len(self.traj_start_idx)-1):
                if self.traj_start_idx[i+1]-self.traj_start_idx[i]==1000:
                    self.corrected_traj_start_idx.append(self.traj_start_idx[i])
            self.traj_start_idx = np.array(self.corrected_traj_start_idx)
            self.traj_end_idx = self.traj_start_idx+1000
            self.num_traj = len(self.traj_start_idx) - 1
            # import ipdb;ipdb.set_trace()
        else:
            timeouts = np.where(self.dataset['timeouts'] == True)[0]
            terminals = np.where(self.dataset['terminals'] == True)[0]

            # # Combine the two arrays
            self.traj_start_idx = np.concatenate([np.array([0]),np.sort(np.unique(np.concatenate((timeouts, terminals))))+1])
            # self.traj_start_idx = np.concatenate((self.traj_start_idx, [len(self.dataset['terminals'])]))
            self.num_traj = len(self.traj_start_idx) - 1
            # import ipdb;ipdb.set_trace()
            # print("Found Trajectory Indices")
            # print(self.traj_start_idx)

            # # Debug
            # modified_traj_start_idx = []
            # for i in range(self.num_traj):
            #     traj_start = self.traj_start_idx[i]
            #     traj_end = self.traj_start_idx[i+1]
            #     if traj_end - 2 > traj_start:
            #         # print(f'Traj {i} from {traj_start} to {traj_end}')
            #         # print('Observations x, y : ', self.dataset['observations'][traj_start:traj_end, 0:2])
            #         # print('Goals x, y : ', self.dataset['infos/goal'][traj_start:traj_end, 0:2])
            #         # print('Terminals : ', self.dataset['terminals'][traj_start:traj_end])
            #         # print('Timeouts : ', self.dataset['timeouts'][traj_start:traj_end])
            #         # print('Rewards : ', self.dataset['rewards'][traj_start:traj_end])
            #         modified_traj_start_idx.append(traj_start)
            # self.traj_start_idx = np.array(modified_traj_start_idx)
            # self.traj_start_idx = np.arange(0, len(self.dataset['terminals']), 1000)
            # import ipdb;ipdb.set_trace()
            # self.num_traj = len(self.traj_start_idx) - 1

    def find_trajectory_indices(self):
        """
        Finds the starting indices of trajectories in the D4RL dataset for antmaze environments,
        filtering out trajectories shorter than 985 steps.
        
        Args:
            dataset (dict): D4RL dataset containing 'observations', 'next_observations', and 'terminals'.
            env_name (str): Name of the environment (e.g., "antmaze-umaze-diverse-v0").
        
        Returns:
            num_traj (int): Total number of trajectories longer than 985 steps.
            traj_start_idx (list): List of indices where each valid trajectory starts.
            traj_end_idx (list): List of indices where each valid trajectory ends.
        """
        print("Finding Trajectory Indices")

        # Initialize the list of trajectory start indices
        self.traj_start_idx = [0]  # First trajectory starts at index 0

        # AntMaze environments: identify trajectory boundaries
        if 'antmaze' in self.env_name:
            dones_float = np.zeros_like(self.dataset['rewards'])  # Array to mark trajectory boundaries

            # Identify transitions where trajectories end
            for i in range(len(dones_float) - 1):
                is_discontinuous = (
                    np.linalg.norm(self.dataset['observations'][i + 1] - self.dataset['next_observations'][i]) > 1e-6
                )
                is_terminal = self.dataset['terminals'][i] == 1.0

                # Mark as trajectory end if discontinuity or terminal state
                if is_discontinuous or is_terminal:
                    dones_float[i] = 1

                # Add the index of the next trajectory start
                if dones_float[i] == 1:
                    self.traj_start_idx.append(i + 1)

            # Calculate trajectory end indices
            self.traj_start_idx = np.array(self.traj_start_idx)
            self.traj_end_idx = np.append(self.traj_start_idx[1:], len(self.dataset['observations']))  # Ends at dataset end

            # Filter out trajectories shorter than 985 steps
            valid_indices = [i for i in range(len(self.traj_start_idx))
                            if self.traj_end_idx[i] - self.traj_start_idx[i] > 985]
            self.traj_start_idx = self.traj_start_idx[valid_indices]
            self.traj_end_idx = self.traj_end_idx[valid_indices]

            # Number of valid trajectories
            self.num_traj = len(self.traj_start_idx)

            #return num_traj, traj_start_idx, traj_end_idx

        else:
            raise ValueError("This function is designed for AntMaze environments.")



    def _sample(self):
        # Sample a trajectory from data
        if self.algo_name=='icvf':
            p_samegoal = 0.5
            p_trajgoal = 0.5
            p_randomgoal = 0.3
            p_currgoal = 0.2
            traj_idx = np.random.randint(0, self.num_traj)
            

            traj_start = self.traj_start_idx[traj_idx]
            if self.traj_end_idx is None:
                traj_end = self.traj_start_idx[traj_idx+1]
            else:
                traj_end = self.traj_end_idx[traj_idx]
            s0 = self.dataset['observations'][traj_start]

            s0_ind_vip = np.random.randint(traj_start, traj_end-1)
            s1_ind_vip = s0_ind_vip+1
            s0_vip = self.dataset['observations'][s0_ind_vip, :self.obs_dim]
            s1_vip = self.dataset['observations'][s1_ind_vip, :self.obs_dim]
            prob_sample = np.random.uniform()
            if prob_sample<p_currgoal:
                sT = s1_vip
            elif prob_sample>=p_currgoal and prob_sample<p_currgoal+p_trajgoal:
                sT = self.dataset['observations'][np.random.randint(s0_ind_vip+1, traj_end),:self.obs_dim]
            else:
                other_traj_idx = np.random.randint(0, self.num_traj)
                other_traj_start = self.traj_start_idx[other_traj_idx]
                if self.traj_end_idx is None:
                    other_traj_end = self.traj_start_idx[other_traj_idx+1]
                else:
                    other_traj_end = self.traj_end_idx[other_traj_idx]
                sT = self.dataset['observations'][np.random.randint(other_traj_start, other_traj_end),:self.obs_dim]
        
            # Following VIP
            prob_sample_sampe_goal = np.random.uniform()
            if prob_sample_sampe_goal<p_samegoal:
                s_future = sT
            else:
                prob_sample = np.random.uniform()
                if prob_sample<p_currgoal:
                    s_future = s1_vip
                elif prob_sample>=p_currgoal and prob_sample<p_currgoal+p_trajgoal:
                    s_future= self.dataset['observations'][np.random.randint(s0_ind_vip+1, traj_end),:self.obs_dim]
                else:
                    other_traj_idx = np.random.randint(0, self.num_traj)
                    other_traj_start = self.traj_start_idx[other_traj_idx]
                    if self.traj_end_idx is None:
                        other_traj_end = self.traj_start_idx[other_traj_idx+1]
                    else:
                        other_traj_end = self.traj_end_idx[other_traj_idx]
                    s_future = self.dataset['observations'][np.random.randint(other_traj_start, other_traj_end),:self.obs_dim]
            
            # # Self-supervised reward (this is always -1)
            reward = 0
            terminal = float((np.abs(s1_vip-sT)).mean()<1e-3)

            return (s0, sT, s_future, s0_vip, s1_vip, reward, terminal)
        elif 'smore' in self.algo_name or 'smore_vodice' in self.algo_name:

            # import pdb; pdb.set_trace()

            p_trajgoal = 0.5
            p_currgoal = 0.0
            traj_idx = np.random.randint(0, self.num_traj)

            # print(self.num_traj, "NUM TRAJECTORIES", "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # print(traj_idx, "TRAJECTORY INDEX", "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # print(self.traj_start_idx, "START INDICES", "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


            traj_start = self.traj_start_idx[traj_idx]
            if self.traj_end_idx is None:
                traj_end = self.traj_start_idx[traj_idx+1]
            else:
                traj_end = self.traj_end_idx[traj_idx]
            
            s0 = self.dataset['observations'][traj_start]

            s0_ind_vip = np.random.randint(traj_start, traj_end-1)
            s1_ind_vip = s0_ind_vip+1
            s0_vip = self.dataset['observations'][s0_ind_vip, :self.obs_dim]
            s1_vip = self.dataset['observations'][s1_ind_vip, :self.obs_dim]
            prob_sample = np.random.uniform()
            if prob_sample<p_currgoal:
                sT = s1_vip
            elif prob_sample>=p_currgoal and prob_sample<p_currgoal+p_trajgoal:
                sT = self.dataset['observations'][np.random.randint(s0_ind_vip+1, traj_end),:self.obs_dim]
            else:
                other_traj_idx = np.random.randint(0, self.num_traj)
                other_traj_start = self.traj_start_idx[other_traj_idx]
                if self.traj_end_idx is None:
                    other_traj_end = self.traj_start_idx[other_traj_idx+1]
                else:
                    other_traj_end = self.traj_end_idx[other_traj_idx]
                sT = self.dataset['observations'][np.random.randint(other_traj_start, other_traj_end),:self.obs_dim]

            # # Self-supervised reward (this is always -1)
            reward = 0
            terminal = float((np.abs(s1_vip-sT)).mean()<1e-3)

            # Sample goal transition distribution uniformly
            gt_traj_idx = np.random.randint(0, self.num_traj)
            gt_traj_start = self.traj_start_idx[gt_traj_idx]
            if self.traj_end_idx is None:
                gt_traj_end = self.traj_start_idx[gt_traj_idx+1]
            else:
                gt_traj_end = self.traj_end_idx[gt_traj_idx]
            gt_s0_ind_vip = np.random.randint(gt_traj_start, gt_traj_end-1)
            gt_s1_ind_vip = gt_s0_ind_vip+1
            gt_s0 = self.dataset['observations'][gt_s0_ind_vip, :self.obs_dim]
            gt_s1 = self.dataset['observations'][gt_s1_ind_vip, :self.obs_dim]

            return (s0, sT, s0_vip, s1_vip, gt_s0, gt_s1, reward, terminal)
        elif 'dvl_odice' in self.algo_name:

            p_trajgoal = 0.3
            p_currgoal = 0.2
            traj_idx = np.random.randint(0, self.num_traj)
            

            traj_start = self.traj_start_idx[traj_idx]
            if self.traj_end_idx is None:
                traj_end = self.traj_start_idx[traj_idx+1]
            else:
                traj_end = self.traj_end_idx[traj_idx]
            
            s0 = self.dataset['observations'][traj_start]

            s0_ind_vip = np.random.randint(traj_start, traj_end-1)
            s1_ind_vip = s0_ind_vip+1
            s0_vip = self.dataset['observations'][s0_ind_vip, :self.obs_dim]
            s1_vip = self.dataset['observations'][s1_ind_vip, :self.obs_dim]
            prob_sample = np.random.uniform()
            if prob_sample<p_currgoal:
                sT = s1_vip
            elif prob_sample>=p_currgoal and prob_sample<p_currgoal+p_trajgoal:
                sT = self.dataset['observations'][np.random.randint(s0_ind_vip+1, traj_end),:self.obs_dim]
            else:
                other_traj_idx = np.random.randint(0, self.num_traj)
                other_traj_start = self.traj_start_idx[other_traj_idx]
                if self.traj_end_idx is None:
                    other_traj_end = self.traj_start_idx[other_traj_idx+1]
                else:
                    other_traj_end = self.traj_end_idx[other_traj_idx]
                sT = self.dataset['observations'][np.random.randint(other_traj_start, other_traj_end),:self.obs_dim]

            # # Self-supervised reward (this is always -1)
            reward = float((np.abs(s1_vip-sT)).mean()<1e-3)
            terminal = float((np.abs(s1_vip-sT)).mean()<1e-3)

            # Sample goal transition distribution uniformly
            gt_traj_idx = np.random.randint(0, self.num_traj)
            gt_traj_start = self.traj_start_idx[gt_traj_idx]
            if self.traj_end_idx is None:
                gt_traj_end = self.traj_start_idx[gt_traj_idx+1]
            else:
                gt_traj_end = self.traj_end_idx[gt_traj_idx]
            gt_s0_ind_vip = np.random.randint(gt_traj_start, gt_traj_end-1)
            gt_s1_ind_vip = gt_s0_ind_vip+1
            gt_s0 = self.dataset['observations'][gt_s0_ind_vip, :self.obs_dim]
            gt_s1 = self.dataset['observations'][gt_s1_ind_vip, :self.obs_dim]

            return (s0, sT, s0_vip, s1_vip, gt_s0, gt_s1, reward, terminal)
        else:

            traj_idx = np.random.randint(0, self.num_traj)
            traj_start = self.traj_start_idx[traj_idx]
            if self.traj_end_idx is None:
                traj_end = self.traj_start_idx[traj_idx+1]
            else:
                traj_end = self.traj_end_idx[traj_idx]
            
        
            # Following VIP
            # if traj_end - 2 <= traj_start:
            #     print(f'Traj {traj_idx} from {traj_start} to {traj_end}')
            start_ind = np.random.randint(traj_start, traj_end-1)
            end_ind = np.random.randint(start_ind+1, traj_end)

            s0_ind_vip = np.random.randint(start_ind, end_ind)
            s1_ind_vip = min(s0_ind_vip+1, end_ind)

            # Self-supervised reward (this is always -1)
            reward = float(s1_ind_vip == end_ind) - 1
            terminal = float(s1_ind_vip == end_ind)
            random_goal = self.dataset['observations'][np.random.randint(0,len(self.dataset['observations'])), :self.obs_dim]
            
            # Get o_t, o_k, o_k+1, o_T
            s0 = self.dataset['observations'][start_ind, :self.obs_dim]
            sT = self.dataset['observations'][end_ind, :self.obs_dim]
            if self.algo_name!='vip':
                if np.random.uniform()<0.2:
                    sT = random_goal
            s0_vip = self.dataset['observations'][s0_ind_vip, :self.obs_dim]
            s1_vip = self.dataset['observations'][s1_ind_vip, :self.obs_dim]

            return (s0, sT, s0_vip, s1_vip, reward, terminal)
    
    def __iter__(self):
        while True:
            yield self._sample()
import os
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import torch
from torchvision.transforms import ToTensor
from concurrent.futures import ThreadPoolExecutor
class VisualD4RL(IterableDataset):
    def __init__(self, env, env_name, expert_env_name, path='', batch_size=32, num_workers=32, algo = '', frame_stack=4):
        self._num_workers = max(1, num_workers)

        self.env = env
        self.env_name = env_name
        self.algo_name = algo
        self.batch_size = batch_size
        self.frame_stack = frame_stack
        # self.goal_indices = goal_indices

        print("ENV NAME", env_name)
        print("EXPERT ENV NAME", expert_env_name)

        if 'kitchen' in env_name:
            self.obs_dim = 30
        else:
            self.obs_dim = env.observation_space.shape[0]
        # self.obs_dim = 30
        
        # self.dataset_path = os.path.join(path, env_name)
        self.dataset_path = [os.path.join(path, 'kitchen-complete-v0'), os.path.join(path, 'kitchen-mixed-v0')]
        self.expert_dataset_path = os.path.join(path, expert_env_name)

        print("DATASET PATH", self.dataset_path)

        #Load an expert trajectory
        self.expert_trajectory_path = os.path.join(self.expert_dataset_path, 'traj0')
        # self.eval_trajectory = [Image.open(os.path.join(self.expert_trajectory_path, f)) for f in os.listdir(self.expert_trajectory_path)]
        eval_ids = np.arange(len(os.listdir(self.expert_trajectory_path)))
        # print("path: ", self.expert_dataset_path)
        self.eval_trajectory = [get_stacked_state(0, idx, ToTensor(), self.expert_trajectory_path, self.frame_stack) for idx in eval_ids]

        #Load a random trajectory
        self.random_trajectory_path = os.path.join(self.dataset_path[1], 'traj0')
        # self.random_trajectory = [Image.open(os.path.join(self.random_trajectory_path, f)) for f in os.listdir(self.random_trajectory_path)]
        random_ids = np.arange(len(os.listdir(self.random_trajectory_path)))
        self.random_trajectory = [get_stacked_state(0, idx, ToTensor(), self.random_trajectory_path, self.frame_stack) for idx in random_ids]

        #paths of all trajectories in dataset
        # self.dataset_paths = [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path)]
        self.dataset_paths = [os.path.join(self.dataset_path[0], f) for f in os.listdir(self.dataset_path[0])] + [os.path.join(self.dataset_path[1], f) for f in os.listdir(self.dataset_path[1])]
        #Store the length of each trajectory
        self.traj_lengths = [len(os.listdir(f)) for f in self.dataset_paths]
        self.num_traj = len(self.dataset_paths)


    def _sample(self):
        # Sample a trajectory from data
        if self.algo_name=='icvf':
            p_samegoal = 0.5
            p_trajgoal = 0.5
            p_randomgoal = 0.3
            p_currgoal = 0.2
            traj_idx = np.random.randint(0, self.num_traj)
            

            traj_start = 0
            traj_end = self.traj_lengths[traj_idx]-1
            s0 = Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{traj_start}.png'))

            s0_ind_vip = np.random.randint(traj_start, traj_end)
            s1_ind_vip = s0_ind_vip+1
            s0_vip = Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{s0_ind_vip}.png'))
            s1_vip = Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{s1_ind_vip}.png'))
            prob_sample = np.random.uniform()
            if prob_sample<p_currgoal:
                sT = s1_vip
            elif prob_sample>=p_currgoal and prob_sample<p_currgoal+p_trajgoal:
                sT = Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{np.random.randint(s0_ind_vip+1, traj_end+1)}.png'))
            else:
                other_traj_idx = np.random.randint(0, self.num_traj)
                other_traj_start = 0
                other_traj_end = self.traj_lengths[other_traj_idx]-1
                sT = Image.open(os.path.join(self.dataset_paths[other_traj_idx], f'img{np.random.randint(other_traj_start, other_traj_end+1)}.png'))
        
            # Following VIP
            prob_sample_sampe_goal = np.random.uniform()
            if prob_sample_sampe_goal<p_samegoal:
                s_future = sT
            else:
                prob_sample = np.random.uniform()
                if prob_sample<p_currgoal:
                    s_future = s1_vip
                elif prob_sample>=p_currgoal and prob_sample<p_currgoal+p_trajgoal:
                    s_future= Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{np.random.randint(s0_ind_vip+1, traj_end+1)}.png'))
                else:
                    other_traj_idx = np.random.randint(0, self.num_traj)
                    other_traj_start = 0
                    other_traj_end = self.traj_lengths[other_traj_idx]-1
                    s_future = Image.open(os.path.join(self.dataset_paths[other_traj_idx], f'img{np.random.randint(other_traj_start, other_traj_end+1)}.png'))
            
            # # Self-supervised reward (this is always -1)
            reward = 0
            terminal = float((np.abs(s1_vip-sT)).mean()<1e-3)

            return (s0, sT, s_future, s0_vip, s1_vip, reward, terminal)
        elif self.algo_name=='smore':

        #     p_trajgoal = 0.5
        #     p_currgoal = 0.0
        #     traj_idx = np.random.randint(0, self.num_traj)
            

        #     traj_start = 0
        #     traj_end = self.traj_lengths[traj_idx]-1
        #     s0 = Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{traj_start}.png'))

        #     s0_ind_vip = np.random.randint(traj_start, traj_end)
        #     s1_ind_vip = s0_ind_vip+1
        #     s0_vip = Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{s0_ind_vip}.png'))
        #     s1_vip = Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{s1_ind_vip}.png'))
        #     prob_sample = np.random.uniform()
        #     if prob_sample<p_currgoal:
        #         sT = s1_vip
        #     elif prob_sample>=p_currgoal and prob_sample<p_currgoal+p_trajgoal:
        #         sT = Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{np.random.randint(s0_ind_vip+1, traj_end+1)}.png'))
        #     else:
        #         other_traj_idx = np.random.randint(0, self.num_traj)
        #         other_traj_start = 0
        #         other_traj_end = self.traj_lengths[other_traj_idx]-1
        #         sT = Image.open(os.path.join(self.dataset_paths[other_traj_idx], f'img{np.random.randint(other_traj_start, other_traj_end+1)}.png'))

        #     # # Self-supervised reward (this is always -1)
        #     reward = 0
        #     terminal = float((np.abs(s1_vip-sT)).mean()<1e-3)

        #     # Sample goal transition distribution uniformly
        #     gt_traj_idx = np.random.randint(0, self.num_traj)
        #     gt_traj_start = 0
        #     gt_traj_end = self.traj_lengths[gt_traj_idx]-1

        #     gt_s0_ind_vip = np.random.randint(gt_traj_start, gt_traj_end)
        #     gt_s1_ind_vip = gt_s0_ind_vip+1
        #     gt_s0 = Image.open(os.path.join(self.dataset_paths[gt_traj_idx], f'img{gt_s0_ind_vip}.png'))
        #     gt_s1 = Image.open(os.path.join(self.dataset_paths[gt_traj_idx], f'img{gt_s1_ind_vip}.png'))

        #     return (s0, sT, s0_vip, s1_vip, gt_s0, gt_s1, reward, terminal)
            p_trajgoal = 0.5
            p_currgoal = 0.0
            # traj_idx = np.random.randint(0, self.num_traj)
            traj_idx = np.random.randint(1, self.num_traj)

            traj_start = 0
            traj_end = self.traj_lengths[traj_idx] - 1

            # Define the transform to convert images to PyTorch tensors
            transform = ToTensor()

            # Convert images to PyTorch tensors
            s0 = transform(Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{traj_start}.png')))
            s0_ind_vip = np.random.randint(traj_start, traj_end)
            s1_ind_vip = s0_ind_vip + 1
            s0_vip = transform(Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{s0_ind_vip}.png')))
            s1_vip = transform(Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{s1_ind_vip}.png')))

            prob_sample = np.random.uniform()
            if prob_sample < p_currgoal:
                sT = s1_vip
            elif prob_sample >= p_currgoal and prob_sample < p_currgoal + p_trajgoal:
                sT = transform(Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{np.random.randint(s0_ind_vip + 1, traj_end + 1)}.png')))
            else:
                other_traj_idx = np.random.randint(0, self.num_traj)
                other_traj_start = 0
                other_traj_end = self.traj_lengths[other_traj_idx] - 1
                sT = transform(Image.open(os.path.join(self.dataset_paths[other_traj_idx], f'img{np.random.randint(other_traj_start, other_traj_end + 1)}.png')))

            # Calculate terminal using PyTorch tensors
            terminal = float((torch.abs(s1_vip - sT)).mean().item() < 1e-3)

            # Sample goal transition distribution uniformly
            gt_traj_idx = np.random.randint(1, self.num_traj)
            gt_traj_start = 0
            gt_traj_end = self.traj_lengths[gt_traj_idx] - 1

            gt_s0_ind_vip = np.random.randint(gt_traj_start, max(1, gt_traj_end))
            gt_s1_ind_vip = gt_s0_ind_vip + 1
            gt_s0 = transform(Image.open(os.path.join(self.dataset_paths[gt_traj_idx], f'img{gt_s0_ind_vip}.png')))
            gt_s1 = transform(Image.open(os.path.join(self.dataset_paths[gt_traj_idx], f'img{gt_s1_ind_vip}.png')))

            # Return tensors for all outputs
            return s0, sT, s0_vip, s1_vip, gt_s0, gt_s1, torch.tensor(0.0), torch.tensor(terminal)
        
        elif self.algo_name == "dvl_odice":
            p_trajgoal = 0.5
            p_currgoal = 0.0
            traj_idx = np.random.randint(1, self.num_traj)

            traj_start = 0
            traj_end = self.traj_lengths[traj_idx] - 1

            # Define the transform to convert images to PyTorch tensors
            transform = ToTensor()

            # Convert images to PyTorch tensors
            # s0 = transform(Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{traj_start}.png')))
            s0 = get_stacked_state(traj_idx, traj_start, transform, self.dataset_paths, self.frame_stack)
            s0_ind_vip = np.random.randint(traj_start, traj_end)
            s1_ind_vip = s0_ind_vip + 1
            # s0_vip = transform(Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{s0_ind_vip}.png')))
            # s1_vip = transform(Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{s1_ind_vip}.png')))
            s0_vip = get_stacked_state(traj_idx, s0_ind_vip, transform, self.dataset_paths, self.frame_stack)
            s1_vip = get_stacked_state(traj_idx, s1_ind_vip, transform, self.dataset_paths, self.frame_stack)

            prob_sample = np.random.uniform()
            if prob_sample < p_currgoal:
                sT = s1_vip
            elif prob_sample >= p_currgoal and prob_sample < p_currgoal + p_trajgoal:
                # sT = transform(Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{np.random.randint(s0_ind_vip + 1, traj_end + 1)}.png')))
                sT = get_stacked_state(traj_idx, np.random.randint(s0_ind_vip + 1, traj_end+1), transform, self.dataset_paths, self.frame_stack)
            else:
                other_traj_idx = np.random.randint(0, self.num_traj)
                other_traj_start = 0
                other_traj_end = self.traj_lengths[other_traj_idx] - 1
                # sT = transform(Image.open(os.path.join(self.dataset_paths[other_traj_idx], f'img{np.random.randint(other_traj_start, other_traj_end + 1)}.png')))
                sT = get_stacked_state(other_traj_idx, np.random.randint(other_traj_start, other_traj_end+1), transform, self.dataset_paths, self.frame_stack)

            # Calculate reward and terminal using PyTorch tensors
            reward = float((torch.abs(s1_vip - sT)).mean().item() < 5e-3)
            terminal = float((torch.abs(s1_vip - sT)).mean().item() < 5e-3)

            # Sample goal transition distribution uniformly
            gt_traj_idx = np.random.randint(1, self.num_traj)
            gt_traj_start = 0
            gt_traj_end = self.traj_lengths[gt_traj_idx] - 1

            gt_s0_ind_vip = np.random.randint(gt_traj_start, max(1, gt_traj_end))
            gt_s1_ind_vip = gt_s0_ind_vip + 1
            # gt_s0 = transform(Image.open(os.path.join(self.dataset_paths[gt_traj_idx], f'img{gt_s0_ind_vip}.png')))
            # gt_s1 = transform(Image.open(os.path.join(self.dataset_paths[gt_traj_idx], f'img{gt_s1_ind_vip}.png')))
            gt_s0 = get_stacked_state(gt_traj_idx, gt_s0_ind_vip, transform, self.dataset_paths, self.frame_stack)
            gt_s1 = get_stacked_state(gt_traj_idx, gt_s1_ind_vip, transform, self.dataset_paths, self.frame_stack)

            # Return tensors for all outputs
            return s0, sT, s0_vip, s1_vip, gt_s0, gt_s1, torch.tensor(reward), torch.tensor(terminal)

        else:

            traj_idx = np.random.randint(0, self.num_traj)
            traj_start = 0
            traj_end = self.traj_lengths[traj_idx]-1
            
        
            # Following VIP
            # if traj_end - 2 <= traj_start:
            #     print(f'Traj {traj_idx} from {traj_start} to {traj_end}')
            start_ind = np.random.randint(traj_start, traj_end-1)
            end_ind = np.random.randint(start_ind+1, traj_end)

            s0_ind_vip = np.random.randint(start_ind, end_ind)
            s1_ind_vip = min(s0_ind_vip+1, end_ind+1)

            # Self-supervised reward (this is always -1)
            reward = float(s1_ind_vip == end_ind) - 1
            terminal = float(s1_ind_vip == end_ind)
            random_traj = np.random.randint(0,len(self.dataset_paths))
            # random_goal = Image.open(os.path.join(self.dataset_paths[random_traj], f'img{np.random.randint(0, self.traj_lengths[random_traj])}.png'))
            random_goal = get_stacked_state(random_traj, np.random.randint(0, self.traj_lengths[random_traj]), ToTensor(), self.dataset_paths, self.frame_stack)
            
            # Get o_t, o_k, o_k+1, o_T
            # s0 = Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{start_ind}.png'))
            # sT = Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{end_ind}.png'))
            s0 = get_stacked_state(traj_idx, start_ind, ToTensor(), self.dataset_paths, self.frame_stack)
            sT = get_stacked_state(traj_idx, end_ind, ToTensor(), self.dataset_paths, self.frame_stack)
            if self.algo_name!='vip':
                if np.random.uniform()<0.2:
                    sT = random_goal
            # s0_vip = Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{s0_ind_vip}.png'))
            # s1_vip = Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{s1_ind_vip}.png'))
            s0_vip = get_stacked_state(traj_idx, s0_ind_vip, ToTensor(), self.dataset_paths, self.frame_stack)
            s1_vip = get_stacked_state(traj_idx, s1_ind_vip, ToTensor(), self.dataset_paths, self.frame_stack)

            # return (pil_to_tensor(s0), pil_to_tensor(sT), pil_to_tensor(s0_vip), pil_to_tensor(s1_vip), reward, terminal)
            return (s0, sT, s0_vip, s1_vip, reward, terminal)
    

    def _batch_sample(self, batch_size):
        transform = ToTensor()
        batch = []

        for _ in range(batch_size):
            try:
                # Set probabilities
                p_trajgoal = 0.5
                p_currgoal = 0.0

                # Sample a trajectory index and its start/end points
                traj_idx = np.random.randint(1, self.num_traj)
                traj_start = 0
                traj_end = self.traj_lengths[traj_idx] - 1

                while traj_start >= traj_end:
                    traj_idx = np.random.randint(0, self.num_traj)
                    traj_start = 0
                    traj_end = self.traj_lengths[traj_idx] - 1

                # Sample states
                s0 = transform(Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{traj_start}.png')))
                s0_ind_vip = np.random.randint(traj_start, traj_end)
                s1_ind_vip = s0_ind_vip + 1
                s0_vip = transform(Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{s0_ind_vip}.png')))
                s1_vip = transform(Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{s1_ind_vip}.png')))

                prob_sample = np.random.uniform()
                if prob_sample < p_currgoal:
                    sT = s1_vip
                elif prob_sample >= p_currgoal and prob_sample < p_currgoal + p_trajgoal:
                    sT = transform(Image.open(os.path.join(self.dataset_paths[traj_idx], f'img{np.random.randint(s0_ind_vip + 1, traj_end + 1)}.png')))
                else:
                    other_traj_idx = np.random.randint(0, self.num_traj)
                    other_traj_start = 0
                    other_traj_end = self.traj_lengths[other_traj_idx] - 1
                    while other_traj_start >= other_traj_end:
                        other_traj_idx = np.random.randint(0, self.num_traj)
                        other_traj_start = 0
                        other_traj_end = self.traj_lengths[other_traj_idx] - 1
                    sT = transform(Image.open(os.path.join(self.dataset_paths[other_traj_idx], f'img{np.random.randint(other_traj_start, other_traj_end + 1)}.png')))

                # Compute terminal
                terminal = float((torch.abs(s1_vip - sT)).mean().item() < 1e-3)

                # Sample goal transitions
                gt_traj_idx = np.random.randint(0, self.num_traj)
                gt_traj_start = 0
                gt_traj_end = self.traj_lengths[gt_traj_idx] - 1

                while gt_traj_start >= gt_traj_end:
                    gt_traj_idx = np.random.randint(0, self.num_traj)
                    gt_traj_start = 0
                    gt_traj_end = self.traj_lengths[gt_traj_idx] - 1

                gt_s0_ind_vip = np.random.randint(gt_traj_start, max(1, gt_traj_end))
                gt_s1_ind_vip = gt_s0_ind_vip + 1
                gt_s0 = transform(Image.open(os.path.join(self.dataset_paths[gt_traj_idx], f'img{gt_s0_ind_vip}.png')))
                gt_s1 = transform(Image.open(os.path.join(self.dataset_paths[gt_traj_idx], f'img{gt_s1_ind_vip}.png')))

                # Add to batch
                batch.append((s0, sT, s0_vip, s1_vip, gt_s0, gt_s1, torch.tensor(0.0), torch.tensor(terminal)))

            except Exception as e:
                print(f"Error sampling batch: {e}")
                continue

        # Validate and stack batch tensors
        batch = [sample for sample in batch if all(isinstance(x, torch.Tensor) for x in sample)]
        batch_tensors = [torch.stack([sample[i] for sample in batch]) for i in range(len(batch[0]))]
        return batch_tensors


        
    def __iter__(self):
        while True:
            yield self._sample()
        # while True:
        #     yield self._batch_sample(batch_size=self.batch_size)
