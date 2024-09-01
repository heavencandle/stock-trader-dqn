import OpenApi from OpenApi
import os
import sqlite3 as sql

'''
[timeline]
0. 오전 8시 30분 - 전일 state 추가(100일보다 적으면 100일 먼저 적재, 이동평균 위함), 전일 state 기준으로 action 선정
1. 오전 9시 - action 시행  
'''

class Kiwoom():
    # 행동 정의
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 홀딩

    def __init__(self):
        # 키움증권 연동
        self.api = OpenApi.Openapi()
        # DB 연동
        conn = self.db_connect()
        # DB 연동 종료
        conn.cursor().close()
        conn.close()

    def db_connect(self, db_name = 'STOCK_TRADING.db', table_name = None):
        db_path = os.path.dirname(os.path.abspath('__file__'))
        db_path = os.path.join(f'run_data/{db_name}')

        # DB 연동, 없을 경우 생성
        conn = sql.connect(db_path)
        cur = conn.cursor()
        
        # 종목코드 이름으로 된 테이블이 없을 경우, 생성
        # [일자, 시가, 종가, 고가, 저가, 거래량, 보유수량, 수익률, 다음날 행동, 행동 확신도]
        cur.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (date text, open int, close int, high int, low int, volume int, stock_cnt int, profit float, action int, confidence float, unit int)")
        # commit the changes to db
        conn.commit()

        return conn
    def update_state(self, when):
        pass
    def act(self, action, confidence):
        # 매매 / 매도 불가한 상황에서는 관망
        if not self.validate_action(action):
            action = self.ACTION_HOLD

        # 환경에서 현재 가격 얻기
        curr_price = self.env.get_price()

        # 매수
        if action == self.ACTION_BUY:
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
                self.num_sell += 1
        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 홀딩 횟수 증가

        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        self.profitloss = ((self.portfolio_value - self.initial_balance) / self.initial_balance)

        # 보상 = 누적 수익율 + 직전 상태 대비 변화량. 누적 수익율만 반영할 경우, 다음 state에 이득이 되는 행동을 하더라도 반영되지 않음
        self.short_reward = self.profitloss-self.profitloss_before
        self.long_reward = self.profitloss
        self.profitloss_before = self.profitloss

        return self.short_reward,self.long_reward
    def validate_action(self, action):
        if action == self.ACTION_BUY:
            # min_trading_unit만큼 살 수 있는지 확인
            if self.balance < self.env.get_price() * (1 + self.TRADING_CHARGE) * self.min_trading_unit:
                return False
        elif action == self.ACTION_SELL:
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



