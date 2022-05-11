# Environment
# - OS: Ubuntu 20.04.4 LTS (64-bit)
# - Processor: Intel Core i5-9400 CPU @ 2.90GHz x 6
# - Graphics: NVDIA Corporation GP102 [GeForce GTX 1080 TI]
# - Dependencies: python (3.8.10), atari_py (0.2.6), gym (0.17.2), nes_py, gym-super-mario-bros (7.3.0)

import argparse
import os
import datetime

import torch
from torch.optim import Adam
from pathlib import Path

# from FUNCTIONS.triplet_loss import *
from FUNCTIONS.reinforcement_learning import *
from FUNCTIONS.double_deep_q_learning import *



def main():
    """
    1. Compiling arguments.
    2. Preparing for the next step.
    3. Testing functions.
        - 3.1. Triplet loss
        - 3.2. Reinforcement learning
                - 3.2.1 Q-learning
                - 3.2.2 Deep Q-learning
                - 3.2.3 Double deep Q-learning
    """

    # 1
    parser = argparse.ArgumentParser(description='Testing functions')
    parser.add_argument('--functions', default='triplet_loss', help='type of functions to test')
    parser.add_argument('--evaluate', default=False, required=False, metavar='TF', help='whether to train a new model')                   
    global args
    args = parser.parse_args()
    print("args", args)

    # 2
    os.makedirs('./checkpoints', exist_ok=True)
    today = datetime.datetime.now()
    today = today.strftime('%H%M%S_%m_%d_%Y')

    # 3
    ## 3.1
    if args.functions == 'triplet_loss':
        tri_model = TripletLossModel()
        train_generator = tri_model.generate_triplets()
        test_generator = tri_model.generate_triplets(test=True)
        batch = next(train_generator)
        base_model = tri_model.embed_model()
        model = tri_model.complete_model(base_model)
        model.summary()
        if args.evaluate == True:
            history = model.fit_generator(train_generator, 
                            validation_data=test_generator, 
                            epochs=20, 
                            verbose=2,steps_per_epoch=20, 
                            validation_steps=30)
            model.save_weights(f'./checkpoints/model_{today}.hdf5')
            plot_model(history)
        elif args.evaluate == False:
            pass
    ## 3.2
    elif args.functions == 'reinforcement_learning':
        
        ### 3.2.1
        ql_model = QLearnModel()
        qvals, acttraj = ql_model.qlearn()
        print("final Q-values: ", qvals)
        print("final actions: ", acttraj[-4:])
        
        ### 3.2.2
#         env = wrap_env(env_ip="Pong-v4")
#         device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#         print("device: ", device)
#         main_dql_model, tgt_dql_model = init_models(env, device)
#         opt = Adam(main_dql_model.parameters(), lr=1e-4)
#         print(f"optimizer: {opt}")
#         replay_buffer = ReplayBuffer(max_cap=20000)
#         print(f"replay buffer: {replay_buffer}")
#         train_model(env, main_dql_model, tgt_dql_model, opt, replay_buffer, device)
        
        ## 3.3
        # Initializing environment
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0") # initialize Super Mario environment
        env = JoypadSpace(env, [['right'], ['right', 'A']]) # limit the action-space to 0. walk right and 1. jump right
        env.reset()

        next_state, reward, done, info = env.step(action=0)
        print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

        # Preprocessing environment
        env = SkipFrame(env, skip=4)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=84)
        env = FrameStack(env, num_stack=4) # four gray-scaled (84, 84) consecutive frames stacked states

        # Training agent for at least 40,000 episodes
        use_cuda = torch.cuda.is_available()
        print(f"Using CUDA: {use_cuda}\n")
        save_dir = Path(f'../checkpoints') # currrent dir: ./scripts
        # save_dir.mkdir(parents=True) #, exist_ok=True)
        # os.mkdir(save_dir) #, exist_ok=True)
        
        evaluation = True # either train or replay
        if evaluation:
            checkpoint_file = Path(f'../checkpoints/trained_mario.chkpt')
        else:
            checkpoint_file = None
        mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint_file)
        if evaluation:
            mario.exploration_rate = mario.exploration_rate_min
        logger = MetricLogger(save_dir)
        episodes = 10 #4000
        for e in range(episodes):
            state = env.reset()

            # Play the game!
            while True:
                # if evaluation:
                  # show environment
                  # env.render()
                # Run agent on the state
                action = mario.act(state)
                # Return experiences based on agent's action
                next_state, reward, done, info = env.step(action)
                # remember
            mario.cache(state, next_state, action, reward, done)
            if evaluation:
                logger.log_step(reward, None, None)
            else:
                # learn Q-values for agent
                q, loss = mario.learn()
                # log saved Q-values
                logger.log_step(reward, loss, q)
            # update state
            state = next_state
            # check if the game ended
            if done or info['flag_get']:
                break

        logger.log_episode()

        if e % 20 == 0:
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

        

if __name__ == '__main__':
    main()