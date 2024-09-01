import numpy as np
import pandas as pd

class Agent():
    STATE_DIM = 2  # 주식 보유 비율, initial balance 대비 포트폴리오 가치 변화율

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0.00015  # 거래 수수료 0.015%
    TRADING_TAX = 0.0025  # 거래세 0.25%

    # 행동 정의
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 홀딩
    ACTIONS = [ACTION_BUY, ACTION_SELL]
    NUM_ACTIONS = len(ACTIONS)

    def __init__(self, environment, initial_balance):
        # 현재 주식 가격을 가져오기 위해 환경 참조
        self.env = environment

        # 최소 매매 단위, 최대 매매 단위
        # self.min_trading_unit = max(int(10_000 / environment.train_data.iloc[-1]['Close']), 1)  # 최소 : 1만원 / 거래
        self.min_trading_unit = 1
        max_trading_price = int(initial_balance//10)
        self.max_trading_unit = max((max_trading_price / environment.train_data.iloc[-1]['close']), 1)  # 최대: 10만원 / 거래

        # Agent 클래스의 속성
        self.initial_balance = initial_balance  # 초기 자본금
        self.balance = initial_balance # 현재 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        self.unit_price = 0 # 주식 단가
        self.stock_value = 0 # 보유 주식 수 * 주식 단가
        self.portfolio_value = self.initial_balance # 포트폴리오 가치: balance + num_stocks * {현재 주가}
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 홀딩 횟수
        self.exploration_threshold = 0

        # Agent 클래스의 상태
        self.ratio_hold = 0  # 주식 보유 비율
        self.profit_pvwise = 0  # 첫 portfolio 가치 대비 손익
        self.profit_stockwise = 0  # 주식 가치 변화율

        # Reward
        self.profit_pvwise_before = 0
        self.profit_stockwise_before = 0
        self.short_reward = 0.  # 단기적 보상
        self.long_reward = 0.  # 장기적 보상


    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.unit_price = 0  # 주식 단가
        self.stock_value = 0  # 보유 주식 수 * 주식 단가
        self.portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.short_reward = 0.
        self.long_reward = 0.

        self.profit_pvwise = 0  # 첫 portfolio 가치 대비 손익
        self.profit_stockwise = 0
        self.profit_stockwise_before = 0
        self.profit_pvwise = 0

        self.ratio_hold = 0  # 주식 보유 비율

        self.exploration_threshold = 0.5 + np.random.rand() / 2  # 탐험시 행동 양상 결정 기준 (높을수록 탐험시 매수할 가능성이 높음)

    def get_states(self):
        available = int(self.portfolio_value / self.env.get_price())
        if self.num_stocks >0 and available>0:
            self.ratio_hold = self.num_stocks / int(self.portfolio_value / self.env.get_price())
        else:
            self.ratio_hold = 0 # underflow, division by zero 방지 위함

        return [self.ratio_hold, self.profit_pvwise]

    def decide_action(self, q_vals, epsilon):
        confidence = 0.
        # q values가 모두 같은 경우 탐험
        max_q = np.max(q_vals)
        if (q_vals == max_q).all(): epsilon = 1

        # 탐험 결정
        if np.random.rand() < epsilon:
            exploration = True
            if np.random.rand() < self.exploration_threshold:
                action = self.ACTION_BUY
            else:
                action = np.random.randint(self.NUM_ACTIONS - 1) + 1
        else:
            exploration = False
            action = np.argmax(q_vals)

        confidence = softmax(q_vals)[action]

        return action, confidence, exploration


    def validate_action(self, action):
        if action == Agent.ACTION_BUY:
            # min_trading_unit만큼 살 수 있는지 확인
            if self.balance < self.env.get_price() * (1 + self.TRADING_CHARGE) * self.min_trading_unit:
                return False
        elif action == Agent.ACTION_SELL:
            # 적어도 1주를 팔 수 있는지 확인
            if self.num_stocks <= 0:
                return False
        return True

    def decide_trading_unit(self, confidence):
        # 0 <= additional <= max_trading_unit - min_trading_unit
        additional = max(min(
            int(confidence * (self.max_trading_unit - self.min_trading_unit)), self.max_trading_unit - self.min_trading_unit
        ), 0)
        return self.min_trading_unit + additional

    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 환경에서 현재 가격 얻기
        curr_price = self.env.get_price()

        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)

            # balance가 부족할 경우 trading_unit 재설정, 0< trading_unit <=
            if self.balance < curr_price * (1 + self.TRADING_CHARGE) * trading_unit:
                trading_unit = self.balance // (curr_price * (1 + self.TRADING_CHARGE))
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            if invest_amount > 0:
                self.balance -= invest_amount
                self.num_stocks += trading_unit
                self.stock_value += curr_price * trading_unit
                self.num_buy += 1

        # 매도
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도
            invest_amount = curr_price * (1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            if invest_amount > 0:
                self.num_stocks -= trading_unit
                self.balance += invest_amount
                self.stock_value -= curr_price * trading_unit
                self.num_sell += 1
        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 홀딩 횟수 증가

        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        self.profit_pvwise_before = self.profit_pvwise
        self.profit_pvwise = ((self.portfolio_value - self.initial_balance) / self.initial_balance)

        # 주식 가치 갱신
        if self.num_stocks>0:
            self.unit_price = self.stock_value / self.num_stocks
            self.profit_stockwise_before = self.profit_stockwise
            self.profit_stockwise = (curr_price - self.unit_price) / self.unit_price

        # 보상 = 누적 수익율 + 직전 상태 대비 변화량. 누적 수익율만 반영할 경우, 다음 state에 이득이 되는 행동을 하더라도 반영되지 않음
        # 1. stockwise reward
        # self.short_reward = self.profit_stockwise-self.profit_stockwise_before
        # self.long_reward = self.profit_stockwise
        # 2. pvwise reward
        self.short_reward = self.profit_pvwise - self.profit_pvwise_before
        self.long_reward = self.profit_pvwise

        return action, self.short_reward,self.long_reward

def softmax(val_list):
    max_v = np.max(val_list)
    val_list -= max_v
    return np.exp(val_list)/np.sum(np.exp(val_list), axis=0)


