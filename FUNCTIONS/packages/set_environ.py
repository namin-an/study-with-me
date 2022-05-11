import os
import math
import re
from collections import deque
import cv2 as cv
import numpy as np
import torch
from torch import save
import atari_py as ap
from gym import make, ObservationWrapper, Wrapper
from gym.spaces import Box



def cal_temp_diff_loss(main_dql_model, tgt_dqn_model, batch, gm, device):
    """
    Calculating time/temporal difference loss
    """
    
    s, a, r, next_s, fin = batch
    
    s = torch.from_numpy(np.float32(s)).to(device)
    next_s = torch.from_numpy(np.float32(next_s)).to(device)
    a = torch.from_numpy(np.float32(a)).to(device)
    r = torch.from_numpy(np.float32(r)).to(device)
    fin = torch.from_numpy(np.float32(fin)).to(device)
    
    q_vals = main_dql_model(s)
    next_q_vals = tgt_dqn_model(next_s)
    
    q_val = q_vals.gather(1, a.type(torch.int64).unsqueeze(-1)).squeeze(-1) # take out Q-values that only correspond to the action indices
    next_q_val = next_q_vals.max(1)[0] # the mazimum value along all columns
    exp_q_val = r + gm*next_q_val*(1-fin) # multiply the next Q-values with gamma (gm) to determine how much future value should be taken into the consideration when calculating the current expected Q-values
    
    loss = (q_val - exp_q_val.data.to(device)).pow(2).mean()
    loss.backward()
    
   ##################################################################################################
        
def gym_to_atari_format(gym_env):
    """
    Initializing the format of gym to Atari
    """
    
    return re.sub(r"(?<!^)(?=[A-Z])", "_", gym_env).lower()


def upd_grph(main_dqn_model, tgt_dqn_model, opt, rep_buffer, device, log, INIT_LEARN=10000, TGT_UPD_FRQL=1000, BATCH_SIZE=64, G=0.99):
    """
    Updating graph by sampling batch of data from replay buffer, calculating time/temporal loss within the batch, and occasionally updating weights of target DQN model. 
    """
    
    if len(rep_buffer) > INIT_LEARN: # as long as initial learning length does not exceed replay buffer length
        if not log.idx % TGT_UPD_FRQL: 
            tgt_dqn_model.load_state_dict(main_dqn_model.state_dict()) # update target dqn model with  weights of main dqn model for every TGT_UPD_FRQPL=1000
        batch = rep_buffer.sample(BATCH_SIZE) # sample from replay buffer
        cal_temp_diff_loss(main_dqn_model, tgt_dqn_model, batch, G, device)
        opt.step()
          
            
def upd_eps(epd, EPS_FINAL=0.005, EPS_START=1.0, EPS_DECAY=100000):
    """
    Updating epsilon function after each episode.
    """
    
    last_eps = EPS_FINAL # the minimum epsilon allowed
    first_eps = EPS_START
    eps_decay = EPS_DECAY
    
    eps = last_eps + (first_eps - last_eps) * math.exp(-1*((epd+1)/eps_decay))
    return eps
        ##################################################################################################
        
def check_atari_env(env):
    """
    Initializing gaming environment by downsampling image frames from video games and putting frames into replay buffer.
    """
    
    for f in ["Deterministic", "ramDeterministic", "ram", "NoFrameskip", "ramNoFrameSkip"]:
        env = env.replace(f, "")
    env = re.sub(r"-v\d+", "", env)
    env = gym_to_atari_format(env) 
    return True if env in ap.list_games() else False
    
class CCtrl(Wrapper):
    def __init__(self, env, is_atari):
        super(CCtrl, self).__init__(env)
        self._is_atari = is_atari

    def reset(self):
        if self._is_atari:
            return self.env.reset()
        else:
            self.env.reset()
            return self.env.render(mode="rgb_array")
        
        
class FrmDwSmpl(ObservationWrapper):
    def __init__(self, env):
        super(FrmDwSmpl, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self._width = 84
        self._height = 84

    def observation(self, observation):
        frame = cv.cvtColor(observation, cv.COLOR_RGB2GRAY)
        frame = cv.resize(frame, (self._width, self._height), interpolation=cv.INTER_AREA)
        return frame[:, :, None]

    
class MaxNSkpEnv(Wrapper):
    def __init__(self, env, atari, skip=4):
        super(MaxNSkpEnv, self).__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip
        self._atari = atari

    def step(self, act):
        total_rwd = 0.0
        fin = None
        for _ in range(self._skip):
            obs, rwd, fin, log = self.env.step(act)
            if not self._atari:
                obs = self.env.render(mode="rgb_array")
            self._obs_buffer.append(obs)
            total_rwd += rwd
            if fin:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_rwd, fin, log

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class FrRstEnv(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env)
        if len(env.unwrapped.get_action_meanings()) < 3:
            raise ValueError("min required action space of 3!")

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, fin, _ = self.env.step(1)
        if fin:
            self.env.reset(**kwargs)
        obs, _, fin, _ = self.env.step(2)
        if fin:
            self.env.reset(**kwargs)
        return obs

    def step(self, act):
        return self.env.step(act)


class FrmBfr(ObservationWrapper):
    def __init__(self, env, num_steps, dtype=np.float32):
        super(FrmBfr, self).__init__(env)
        obs_space = env.observation_space
        self._dtype = dtype
        self.observation_space = Box(obs_space.low.repeat(num_steps, axis=0),
                                     obs_space.high.repeat(num_steps, axis=0), dtype=self._dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self._dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class Img2Trch(ObservationWrapper):
    def __init__(self, env):
        super(Img2Trch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(low=0.0, high=1.0, shape=(obs_shape[::-1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class NormFlts(ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0
    
class TrMetadata:
    """
    Saving different types of metrics to evaluate built models.
    """
    
    def __init__(self):
        self._avg = 0.0
        self._bst_rwd = -float("inf")
        self._bst_avg = -float("inf")
        self._rwds = []
        self._avg_rng = 100
        self._idx = 0

    @property
    def bst_rwd(self):
        return self._bst_rwd

    @property
    def bst_avg(self):
        return self._bst_avg

    @property
    def avg(self):
        """
        Returning the reward averge for the past few episodes.
        """

        avg_rng = self._avg_rng * -1
        return sum(self._rwds[avg_rng:]) / len(self._rwds[avg_rng:])

    @property
    def idx(self):
        """
        Calculating indices to determine when to update weights of target DQN model.
        """
        return self._idx
    
    def _upd_bst_rwd(self, epd_rwd):
        if epd_rwd > self.bst_rwd:
            self._bst_rwd = epd_rwd

    def _upd_bst_avg(self):
        """
        Returning false until the best average comes out and the episode gets terminated
        """
        
        if self.avg > self.bst_avg:
            self._bst_avg = self.avg
            return True
        return False

    def upd_rwds(self, epd_rwd):
        self._rwds.append(epd_rwd)
        self._upd_bst_rwd(epd_rwd)
        return self._upd_bst_avg()

    def upd_idx(self):
        self._idx += 1
        
def fin_epsd(DQN_model, env, log, eps_rwd, epd, eps):
    """
    Defining each episode by saving the best weights obtained so far for CNN printing out the rewards.
    """
    
    bst_so_fat = log.upd_rwds(eps_rwd)
    if bst_so_fat:
        print(f"checkpointing current model weights. the highest running_avg_rew of {round(log.bst_avg, 3)} achieved!")
        save(DQN_model.state_dict(), f"{env}.dat")
    print(f"episode_num {epd}, curr_reward: {eps_rwd}, best_reward: {log.bst_rwd}, \
            running_avg_rew: {round(log.avg, 3)}, curr_epsilon: {round(eps, 4)}")    

    
def run_episode(env, main_dqn_model, tgt_dqn_model, opt, rep_buffer, device, log, episode, ENV="Pong-v4"):
    """
    Running episodes for DQN loops until the agent reaches the end state.
    """
    
    epd_r = 0.0 # initialize rewards
    s = env.reset() # initialize states
    
    while True: # infinity loop until the agent reaches the final state
        eps = upd_eps(log.idx) # update the value of epsilon
        a = main_dqn_model.act_perf(s, eps, device) # return the action depending on the value of the current epsilon
        if True:
            env.render()
            pass
        next_s, r, fin, _ = env.step(a) # automatically get states and rewards according to the action that the agent has taken
        rep_buffer.push(s, a, r, next_s, fin) # saves states and actions into the buffer
        s = next_s # the next state becomes the current one
        epd_r += r # accumulate reward
        log.upd_idx() # update the index
        upd_grph(main_dqn_model, tgt_dqn_model, opt, rep_buffer, device, log) # update the graph by sampling batch from replay buffer by using the updated state value
        if fin:
            fin_epsd(main_dqn_model, ENV, log, epd_r, episode, eps) # print out the current best reward so far for the end episode
            break
 