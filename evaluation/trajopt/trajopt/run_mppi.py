"""
This is a launcher script for launching mjrl training using hydra
"""
import os
# MUST set before any mujoco/dm_control imports
if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import time as timer
import glob
import hydra
from omegaconf import DictConfig, OmegaConf
import pickle 
from tqdm import tqdm 
import multiprocessing as mp
from matplotlib import pyplot as plt
from moviepy.editor import ImageSequenceClip
import skvideo.io
import os 
import wandb
from PIL import Image 
 
from mj_envs.envs.env_variants import register_env_variant
from trajopt.envs.obs_wrappers import env_constructor
from trajopt.algos.mppi import MPPI
from trajopt import DEMO_PATHS


@hydra.main(config_name="mppi_config", config_path="config")
def configure_jobs(job_data):
    OUT_DIR = '.'
    PICKLE_FILE = OUT_DIR + '/trajectories.pickle'

    assert job_data.embedding_reward ==  True
    assert job_data.env_kwargs.embedding_reward == True 

    if "state_" in job_data.embedding:
        job_data.env_kwargs.load_path = f"{job_data.embedding}:{job_data.checkpoint_path}"
    else:
        job_data.env_kwargs.load_path = job_data.embedding
    job_data.job_name = job_data.embedding
    reward_type = f"{job_data.embedding}" if job_data.embedding_reward else "true"
    job_data.job_name = f"{job_data.env}-{reward_type}-{job_data.camera}-{job_data.env_kwargs.init_timestep}-{job_data.env_kwargs.goal_timestep}-seed{job_data.seed}"
    
    with open('job_config.yaml', 'w') as fp:
        OmegaConf.save(config=job_data, f=fp.name)

    # Construct environment 
    env_kwargs = job_data['env_kwargs']
    env = env_constructor(**env_kwargs)

    mean = np.zeros(env.action_dim)
    sigma = 1.0*np.ones(env.action_dim)
    filter_coefs = [sigma, job_data['filter']['beta_0'], job_data['filter']['beta_1'], job_data['filter']['beta_2']]
    trajectories = []  # TrajOpt format (list of trajectory classes)

    # Generate trajectories and plot embedding distances
    for i in range(job_data['num_traj']):
        os.makedirs(f"./{i}", exist_ok=True)
        os.mkdir(f"./{i}/logs")
        start_time = timer.time()
        print("Currently optimizing trajectory : %i" % i)
        seed = job_data['seed'] + i*12345
        env.reset(seed=seed)
        
        agent = MPPI(env,
                    H=job_data['plan_horizon'],
                    paths_per_cpu=job_data['paths_per_cpu'],
                    num_cpu=job_data['num_cpu'],
                    kappa=job_data['kappa'],
                    gamma=job_data['gamma'],
                    mean=mean,
                    filter_coefs=filter_coefs,
                    default_act=job_data['default_act'],
                    seed=seed,
                    env_kwargs=env_kwargs)

        # trajectory optimization
        distances = {}
        for camera in agent.env.env.cameras:
            distances[camera] = []
            goal_embedding = agent.env.env.goal_embedding[camera]
            distance = np.linalg.norm(agent.sol_embedding[-1][camera]-goal_embedding)
            distances[camera].append(distance)

        for t_step in tqdm(range(job_data['H_total'])):
            # take one-step with trajectory optimization
            paths = agent.train_step(job_data['num_iter'])

            # At the first planning step (init_timestep), save GIFs of sampled trajectories
            if t_step == 0:
                os.makedirs(f"./{i}/planning", exist_ok=True)
                for camera in agent.env.env.cameras:
                    grid_frames, scores = agent.render_planning_trajectories(
                        paths, camera_name=camera)
                    plan_gif = f"./{i}/planning/sampled_trajectories_{camera}.gif"
                    cl = ImageSequenceClip(grid_frames, fps=5)
                    cl.write_gif(plan_gif, fps=5)
                    print(f"Saved planning GIF: {plan_gif}")
                    print(f"  Scores: min={scores.min():.3f} max={scores.max():.3f} "
                          f"mean={scores.mean():.3f} std={scores.std():.3f}")
            step_info = agent.sol_info[-1]
            obs_dict = step_info.get('obs_dict', {})
            step_log = {
                't': obs_dict.get('t', None),
                'rwd_sparse': step_info.get('rwd_sparse', None),
                'rwd_dense': step_info.get('rwd_dense', None),
                'solved': step_info.get('solved', 0) * 1.0,
                'ee_error': obs_dict.get('ee_error', None),
                'robot_error': obs_dict.get('robot_error', None),
                'objs_error': obs_dict.get('objs_error', None)
            }
            
            # Save embedding distance curve
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,6))
            for camera_id, camera in enumerate(agent.env.env.cameras):
                goal_embedding = agent.env.env.goal_embedding[camera]
                goal_distance = np.linalg.norm(agent.sol_embedding[-1][camera]-goal_embedding)
                distances[camera].append(goal_distance)
                ax[camera_id].plot(np.arange(len(distances[camera])), distances[camera])
                ax[camera_id].set_title(camera)
                step_log[camera] = goal_distance
            
            for key in step_log:
                agent.logger.log_kv(key, step_log[key])

            agent.logger.save_log(f'./{i}/logs')
            plt.suptitle(f"{job_data.env} Video MPPI {job_data.embedding} Distance")
            plt.savefig(f"{i}_{job_data.embedding}_embedding_distance.png")
            plt.close() 

            # Save trajectory video
            for camera in agent.env.env.cameras:
                os.makedirs(f"./{i}/{camera}", exist_ok=True)
                frames = agent.animate_result_offscreen(camera_name=camera)
                VID_FILE = OUT_DIR + f'/{i}/{i}_{job_data.embedding}_{camera}' + '.gif'
                cl = ImageSequenceClip(frames, fps=20)
                cl.write_gif(VID_FILE, fps=20)
                frames = np.array(frames)
                for t2 in range(frames.shape[0]):
                    img = frames[t2]
                    result = Image.fromarray((img).astype(np.uint8))
                    result.save(f"./{i}/{camera}/{t2}.png")
            
        # Save trajectory
        SAVE_FILE = OUT_DIR + '/traj_%i.pickle' % i
        pickle.dump(agent, open(SAVE_FILE, 'wb'))
        
        end_time = timer.time()
        print("Trajectory reward = %f" % np.sum(agent.sol_reward))
        print("Optimization time for this trajectory = %f" % (end_time - start_time))
        trajectories.append(agent)
        pickle.dump(trajectories, open(PICKLE_FILE, 'wb'))
    
if __name__ == "__main__":
    mp.set_start_method('spawn')
    configure_jobs()