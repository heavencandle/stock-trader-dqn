import pandas as pd
import os
from keras.models import load_model
import numpy as np
import sqlite3 as sql

class Trader():
    def __init__(self, mode = 'simulation'):
        ### DB 연결
        conn = sql.connect("testPython.db")
        cur = conn.cursor()

    # 전일 상태를 통해 다음날의 행동 결정
    def decide_action(self, stock_codes, today = '000000' ):
        for stock_code in stock_codes:
            # 전일 상태 불러오기

            reader = csv.DictReader(f)
            today_state = reader[-1]
            _t_state = today_state.copy()
            today_state = [today_state['close'], today_state['open_ratio'], today_state['close_ratio'], today_state['diff_ratio'], today_state['volume_ratio'], today_state['ratio_hold'], today_state['ratio_portfolio_value']]
            # [일자, 종가(절대값), 시가, 종가, 고가-저가, 거래량, 최대보유가능량, 현재 주식가치, 보유수량, 포트폴리오가치, action, confidence]
            # 모델 불러오기
            model_path = os.path.join("./models", f'{stock_code}.h5')
            model = load_model(model_path)

            # 행동 예측
            q_vals = model.predict(np.array(today_state))
            action = np.argmax(q_vals)
            confidence = np.max(q_vals)

            # 확신도, 행동, 거래 unit 저장
            # oreder_reservation table를 따로 만들까? [stock_code, action, trade_unit]
            # order_log 실제 체결 결과 table
            data.loc[idx,'action'] = action
            data.loc[idx, 'confidence'] = confidence
            pd.to_csv(data_path, index = False)




