import argparse
import pickle
import time
import numpy as np
import re

from numpy.core.fromnumeric import ptp

from env import TradingEnv
from agent import Agent
from utils import get_dataset, get_scaler, load_dataset, maybe_make_dir


if __name__ == '__main__':
    # command line interface
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--getdata', nargs='+', type=str, 
        help='retrieve data set using IEX cloud API. Takes 3 args: str symbol, str timeframe (see IEX Cloud docs for values), str version (using production API or not)')
    parser.add_argument('-e', '--episode', type=int, default=1,
        help='number of episode to run')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
        help='batch size for experience replay')
    parser.add_argument('-i', '--initial_invest', type=int, default=20000,
        help='initial investment amount')
    parser.add_argument('-m', '--mode', type=str, required=True,
        help='either "train" or "test"')
    parser.add_argument('-w', '--weights', type=str, help='a trained model weights')
    args = parser.parse_args()

    maybe_make_dir('weights')
    maybe_make_dir('portfolio_val')

    timestamp = time.strftime('%Y%m%d%H%M')

    if args.getdata is not None:
        # * operator used to expand iterable into function call
        get_dataset(*args.getdata)
    
    data = np.around(load_dataset('IBM'))
    splt = int(data.shape[1]*.7) # some index to split the dataset on (70/30 train test split)
 
    train_data = data[:,:splt]
    test_data = data[:,splt:]

    env = TradingEnv(train_data, args.initial_invest)
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = Agent(state_size, action_size)
    scaler = get_scaler(env)

    portfolio_value = []

    for i in range(args.episode):
        state = env.reset()
        state = scaler.transform([state])
        for t in range(env.n_step):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            # Print out value at every 100 steps in the episode
            if t % 100 == 0:
                print('episode: {}/{}, step: {}/{}, current portfolio value: {}'.format(
                    i + 1, args.episode, t, env.n_step, info['curr_val']
                ))

            next_state = scaler.transform([next_state])
            if args.mode == 'train':
                agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, episode end value: {}".format(
                    i + 1, args.episode, info['curr_val']))
                portfolio_value.append(info['curr_val']) # append episode end portfolio value
                break
            if args.mode == 'train' and len(agent.memory) > args.batch_size:
                agent.replay(args.batch_size)
        if args.mode == 'train' and (i + 1) % 10 == 0:  # checkpoint weights
            agent.save('weights/{}-dqn.h5'.format(timestamp))

    # save portfolio value history to disk
    with open('portfolio_val/{}-{}.p'.format(timestamp, args.mode), 'wb') as fp:
        pickle.dump(portfolio_value, fp)

    