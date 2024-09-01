import pandas as pd
import numpy as np

COLUMNS_CHART = ['date', 'open', 'close', 'low', 'high', 'volume']
COLUMNS_TRAIN = ['close', 'open_ratio', 'close_ratio', 'diff_ratio', 'volume_ratio']

def load_data(fpath, date_from, date_to):
    data = pd.read_csv(fpath, thousands=',', converters={'date': lambda x: str(x).replace("-","")})

    # 고가 - 저가 Column 추가
    data['diff'] = data['high'] - data['low']

    # 100일 이동평균 칼럼 추가
    window = 100
    data['open_ma'] = data['open'].rolling(window=window).mean()
    data['close_ma'] = data['close'].rolling(window=window).mean()
    data['diff_ma'] = data['diff'].rolling(window=window).mean()
    data['volume_ma'] = data['volume'].rolling(window=window).mean()
    data = data.dropna()
    
    # 기간 필터링
    data = data[(data['date'] >= date_from) & (data['date'] <= date_to)]

    # 100일 이동평균 대비 값으로 상대치화
    data['open_ratio'] = data.loc[:,'open'] / data.loc[:,'open_ma']
    data['close_ratio'] = data.loc[:,'close'] / data.loc[:,'close_ma']
    data['diff_ratio'] = data.loc[:,'diff'] / data.loc[:,'diff_ma']
    data['volume_ratio'] = data.loc[:,'volume'] / data.loc[:,'volume_ma']
    data.drop(columns=['open_ma', 'close_ma', 'diff_ma', 'volume_ma'])

    # 차트 데이터, 학습 데이터 분리
    chart_data = data[COLUMNS_CHART]
    training_data = data[COLUMNS_TRAIN]

    return chart_data, training_data
