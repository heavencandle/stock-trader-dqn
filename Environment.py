import numpy as np
import pandas as pd
import Agent as agent

class Environment():
    # COLUMNS_CHART = ['Date', 'Open', 'Close', 'Low', 'High', 'Volume']
    # COLUMNS_TRAIN = ['Open_ratio', 'Close_ratio', 'Diff_ratio', 'Volume_ratio']

    CLOSE_IDX = 0  # 종가의 위치
    OPEN_RATIO_IDX = 1 # 시가의 위치
    CLOSE_RATIO_IDX = 2  # 종가의 위치
    DIFF_RATIO_IDX = 3  # diff의 위치
    VOLUME_RATIO_IDX = 4  # 거래량의 위치

    def __init__(self, chart_data, train_data):
        self.chart_data = chart_data
        self.train_data = train_data
        self.FEATURE_NUM = len(train_data.columns)
        self.observation = None
        self.idx = -1

    def reset(self):
        self.observation = None
        self.idx = -1

    def observe(self):
        if len(self.train_data) > self.idx+1:
            self.idx += 1
            self.observation = self.train_data.iloc[self.idx, :]
            return list(self.observation)
        return None

    def get_price(self):
        if self.observation is not None:
            return self.observation[self.CLOSE_IDX]
        return None






