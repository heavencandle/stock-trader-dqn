import json
import logging
import collections
from collections import deque
import datetime
import os
from keras.initializers import random_uniform
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, BatchNormalization
from keras.optimizers import Adam
import pandas as pd
from math import log10
import Agent as Agent
import Environment as Environment
import numpy as np

# 날짜, 시간 관련 문자열 형식
FORMAT_DATE = "%Y%m%d"
FORMAT_DATETIME = "%Y%m%d%H%M%S"


class DQN():
    # Replay Memory 설정
    REPLAY_MEMORY_SIZE = 1_000

    # DNN hyperparameter 설정
    LEARNING_RATE = 0.001

    # DQN settings
    EPSILON_START = 0.5  # not a constant, going to be decayed
    EPSILON_DECAY = 0.99
    EPSILON_MIN = 0.0001
    EPISODES = 30
    BATCHSIZE = 128
    ITERATION = 0
    DISCOUNT_FACTOR = 0.9
    LEARNING_RATE_DQN = 0.0001
    UPDATE_TARGET_EVERY = 5

    def __init__(self, start_date, end_date, stock_code, initial_balance, chart_data, train_data, params_path, models_path, network_type = 'DNN'):
        self.start_date = start_date
        self.end_date = end_date
        self.stock_code = stock_code
        #agent, environment
        self.env = Environment.Environment(chart_data, train_data)
        self.agent = Agent.Agent(self.env, initial_balance)
        # chart visualize용 데이터
        self.chart_data = chart_data

        # initialize memory
        self.memory_curr_state = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.memory_curr_qs = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.memory_action = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.memory_short_reward = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.memory_long_reward = deque(maxlen=self.REPLAY_MEMORY_SIZE)

        # main model - gets trained every step
        # target model - predict every step

        self.model = self.create_model(network_type) #default : DNN
        self.target_model = self.create_model(network_type) #default : DNN
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0
        self.max_episode_reward = 0 # early stopping

        # model 경로 설정
        self.models_path = models_path
        self.model_nm = None
        self.target_model_nm = None

        # DQN 클래스 속성
        self.epsilon = self.EPSILON_START
        self.ITERATION = (len(train_data)-1//self.BATCHSIZE) + 1
        self.loss = 0.
        self.reward = 0.

        # 파라미터 기록
        hyperparams = {
            "learning_rate": self.LEARNING_RATE,
            "replay_memory_size": self.REPLAY_MEMORY_SIZE,
            "epsilon_start": self.EPSILON_START, "epsilon_decay": self.EPSILON_DECAY, "epsilon_min": self.EPSILON_MIN,
            "episodes": self.EPISODES, "batchsize": self.BATCHSIZE, "iteration": self.ITERATION,
            "discount_factor": self.DISCOUNT_FACTOR, "learning_rate_dqn": self.LEARNING_RATE_DQN, "update_target_every": self.UPDATE_TARGET_EVERY
        }
        with open(params_path, 'r+') as f:
            params = json.load(f)
            params = dict(params, **hyperparams)
            f.seek(0)  # rewind
            json.dump(params, f)

    def create_model(self, network_type):
        model = Sequential()

        if network_type=='DNN':
            initializer = random_uniform(minval=-0.05, maxval=0.05, seed=None)
            model.add(Dense(64, input_shape=(self.env.FEATURE_NUM+self.agent.STATE_DIM,), kernel_initializer=initializer))
            model.add(BatchNormalization())
            model.add(Dense(64,kernel_initializer=initializer))
            model.add(Dense(self.agent.NUM_ACTIONS,kernel_initializer=initializer, activation='linear'))
            model.compile(optimizer = Adam(lr = self.LEARNING_RATE), loss = 'mse', metrics = ['mae'])
        return model

    def reset(self):
        # Agent 초기화
        self.agent.reset()
        # 환경 초기화
        self.env.reset()
        # 메모리 초기화
        self.memory_curr_state.clear()
        self.memory_curr_qs.clear()
        self.memory_action.clear()
        self.memory_short_reward.clear()
        self.memory_long_reward.clear()

        # Network 초기화
        self.loss = 0.
        self.reward = 0.
        self.epsilon = self.EPSILON_START

    def update_memory(self, update_size):
        # state, q, action, reward 업데이트
        done = False
        for _ in range(update_size):
            curr_state = self.env.observe()
            if curr_state is None: # 마지막 배치 종료 조건
                done = True
                return done

            # S_t 메모리 추가
            curr_state.extend(self.agent.get_states())
            curr_state = np.array(curr_state)
            # curr_state_reshape = np.array(curr_state).reshape((1,-1))
            self.memory_curr_state.append(np.array(curr_state))
            # self.memory_curr_state.append(curr_state_reshape)
            # Q_t 메모리 추가
            curr_state_reshape = np.array(curr_state).reshape((1,-1))
            curr_qs = self.model.predict(curr_state_reshape)
            curr_qs = list(curr_qs[0])
            self.memory_curr_qs.append(curr_qs)
            # a_t, r_t 메모리 추가
            action, confidence, exploration = self.agent.decide_action(curr_qs, self.epsilon)
            decided_action, short_reward, long_reward = self.agent.act(action, confidence)  # 결정한 행동을 수행하고 보상 획득
            self.memory_action.append(action)
            self.memory_short_reward.append(short_reward)
            self.memory_long_reward.append(long_reward)
        return done

    def create_batch(self, batch_size):
        shuffled_idx = np.random.random_integers(low = 0, high = len(self.memory_curr_state)-1, size = (self.BATCHSIZE,))
        data = zip(np.array(self.memory_curr_state)[shuffled_idx],
                   np.array(self.memory_curr_qs)[shuffled_idx],
                   np.array(self.memory_action)[shuffled_idx],
                   np.array(self.memory_short_reward)[shuffled_idx],
                   np.array(self.memory_long_reward)[shuffled_idx])

        # data = zip(reversed(list(self.memory_curr_state)[-batch_size:]),
        #            reversed(list(self.memory_curr_qs)[-batch_size:]),
        #            reversed(list(self.memory_action)[-batch_size:]),
        #            reversed(list(self.memory_short_reward)[-batch_size:]),
        #            reversed(list(self.memory_long_reward)[-batch_size:]))

        X = np.zeros((batch_size, self.agent.STATE_DIM + self.env.FEATURE_NUM))
        y = np.zeros((batch_size, self.agent.NUM_ACTIONS))

        for i, (curr_state, curr_qs, action, short_reward, long_reward) in enumerate(data):
            reward = long_reward + short_reward
            self.reward+=reward
            X[i] = curr_state
            try:
                # Q(s,a)=Q(s,a)+α(R+γ*maxQ(s′,a′)−Q(s,a))
                y[i][action] = curr_qs[action] + self.LEARNING_RATE_DQN * (reward + future_max_q * self.DISCOUNT_FACTOR - curr_qs[action])
            except UnboundLocalError: y[i][action] = reward

            future_max_q = np.max(self.target_model.predict(curr_state.reshape(1,-1)))
        return (X, y)

    def fit(self, model, batch, discount_factor):
        self.loss += abs(model.train_on_batch(batch[0], batch[1])[0]) #model.metrics_names: ['loss', 'mae']
        return self.loss

    def train(self):
        max_portfolio_value = 0
        for episode in range(self.EPISODES):
            # 에포크 마다 agent, environment, memory, epsilon 재설정
            self.reset()

            # 학습을 진행할 수록 탐험 비율 감소
            epsilon_decayed = self.EPSILON_START * (1. - float(episode) / (self.EPISODES - 1))
            if episode!=0:
                epsilon_decayed *= self.EPSILON_DECAY
            else:
                epsilon_decayed = self.EPSILON_START * self.EPSILON_DECAY
            self.epsilon = max(epsilon_decayed, self.EPSILON_MIN)

            # iterate batch
            for _ in range(self.ITERATION):
                done = self.update_memory(self.BATCHSIZE)
                _batch = self.create_batch(self.BATCHSIZE)
                self.fit(self.model, _batch, self.DISCOUNT_FACTOR)

                # target_model - 주기적으로 weight 전이
                self.target_update_counter += 1
                if self.target_update_counter > self.UPDATE_TARGET_EVERY:
                    self.target_model.set_weights(self.model.get_weights())
                    self.target_update_counter = 0

            # max portfolio value 갱신
            max_portfolio_value = max(max_portfolio_value, self.agent.portfolio_value)
            logging.info("[{}][Episode {}/{}] Epsilon:{:.4f} "
                         "#Buy:{} #Sell:{} #Hold:{} "
                         "#Stocks:{} PV:{:,.0f} "
                         "Reward:{:.3f} Loss:{:.3f} ".format(
                self.stock_code, episode, self.EPISODES, self.epsilon,
                self.agent.num_buy, self.agent.num_sell, self.agent.num_hold,
                self.agent.num_stocks, self.agent.portfolio_value,
                self.reward, self.loss))

        logging.info("[{code}] Max PV:{max_pv:,.0f}".format(
            code=self.stock_code, max_pv=max_portfolio_value))
        self.saveModel()

    def saveModel(self):
        if not os.path.isdir(self.models_path):
            os.makedirs(self.models_path)

        self.model_nm = f'{self.stock_code}_{self.start_date}_{self.end_date}.h5'
        self.target_model_nm = f'{self.stock_code}_{self.start_date}_{self.end_date}_target.h5'

        self.model.save(os.path.join(self.models_path, self.model_nm))
        self.target_model.save(os.path.join(self.models_path, self.target_model_nm))
        logging.info("EPISODES Finished. Model saved")

    def earlyStopping(self, episode_num, curr_ep_reward, best_n = 1):
        if not os.path.isdir(self.models_path):
            os.makedirs(self.models_path)

        self.model_nm = '{}.h5'.format(episode_num)
        self.target_model_nm = '{}_target.h5'.format(episode_num)

        if curr_ep_reward > self.max_episode_reward:
            # save best only
            befores = os.listdir(self.models_path)[:-2]
            for before in befores: os.remove(os.path.join(self.models_path, before))

            self.max_episode_reward = curr_ep_reward
            self.model.save(os.path.join(self.models_path, self.model_nm))
            self.target_model.save(os.path.join(self.models_path, self.target_model_nm))

    def test(self, model_name):
        # 시각화를 위한 변수 선언
        unit_price = []
        stock_value = []
        profit_stockwise = []
        portfolio_value = []
        profit_pvwise = []
        reward = []
        action = []

        # 모델 불러오기
        model = load_model(os.path.join(self.models_path, model_name))
        X = np.zeros((1, self.agent.STATE_DIM + self.env.FEATURE_NUM))

        # 상태 초기 설정
        observation = self.env.observe()

        while observation is not None:
            state = [o for o in observation]
            state.extend(self.agent.get_states())
            X[0] = state

            q_vals = model.predict(X)[0]
            _action, _confidence, _exploration = self.agent.decide_action(q_vals, epsilon=-1)
            decided_action, _short_reward,_long_reward = self.agent.act(_action, _confidence)

            # 시각화 데이터 업데이트
            unit_price.append(self.agent.unit_price)
            stock_value.append(self.agent.stock_value)
            profit_stockwise.append(self.agent.profit_stockwise)
            portfolio_value.append(self.agent.portfolio_value)
            profit_pvwise.append(self.agent.profit_pvwise)
            reward.append(_short_reward + _long_reward)
            action.append(decided_action)

            # 다음 상태 정의
            observation = self.env.observe()

        # save results
        self.chart_data['unit_price'] = unit_price
        self.chart_data['stock_value'] = stock_value
        self.chart_data['profit_stockwise'] = profit_stockwise
        self.chart_data['portfolio_value'] = portfolio_value
        self.chart_data['profit_pvwise'] = profit_pvwise
        self.chart_data['reward'] = reward
        self.chart_data['action'] = action

        self.chart_data.to_csv(f"./test_report/{model_name}_{self.start_date}_{self.end_date}.csv", index=False)





