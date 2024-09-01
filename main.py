import os
import sys
import logging
import argparse
import json
import datetime

import Network
import data_helper

# backend 설정
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"]='0'

# 날짜, 시간 관련 문자열 형식
FORMAT_DATE = "%Y%m%d"
FORMAT_DATETIME = "%Y%m%d%H%M%S"

# 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
model_nm = 'model'
target_model_nm = 'target_model_name'
output_time = datetime.datetime.now().strftime(FORMAT_DATETIME)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('train', 'test', 'simulate'), default='train')
    parser.add_argument('--stock_code', nargs='+', default=['005930']) # 삼성전자 : 005930
    parser.add_argument('--balance', type=int, default=50_000_000)
    parser.add_argument('--start_date', default='20100101')
    parser.add_argument('--end_date', default='20180430') # 삼성전자 180504 액면분할
    parser.add_argument('--model_name', default='005930_20201203114508_20100101_20180430.h5')
    args = parser.parse_args()

    # 출력 경로 설정
    output_path = os.path.join(base_dir,'output/{}'.format(output_time))
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # 파라미터 기록
    params_path = os.path.join(output_path, 'params.json')
    with open(params_path, 'w') as f:
        f.write(json.dumps(vars(args)))

    # 모델 경로 준비
    models_path = os.path.join(output_path, 'models')
    archive_path = os.path.join(base_dir, 'archive')

    # 로그 기록 설정
    file_handler = logging.FileHandler(filename=os.path.join(output_path, "{}.log".format(output_time)), encoding='utf-8')
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
                        handlers=[file_handler, stream_handler], level=logging.DEBUG)

    params = {}
    list_stock_code = []
    list_chart_data = []
    list_train_data = []

    if args.mode=='train':
        for stock_code in args.stock_code:
            # 차트 데이터, 학습 데이터 준비
            chart_data, train_data = data_helper.load_data(
                os.path.join(base_dir,'train_data/{}.csv'.format(stock_code)), args.start_date, args.end_date)
            # 파라미터 설정
            params = {'start_date':args.start_date, 'end_date': args.end_date, 'stock_code': stock_code, 'initial_balance' : args.balance,
                      'chart_data': chart_data, 'train_data': train_data,
                      'params_path': params_path, 'models_path': models_path}
            # 강화학습
            DQN = Network.DQN(**params)
            DQN.train()
    elif args.mode=='test':
        for stock_code in args.stock_code:
            # 차트 데이터, 학습 데이터 준비
            chart_data, train_data = data_helper.load_data(
                os.path.join(base_dir, 'train_data/{}.csv'.format(stock_code)), args.start_date, args.end_date)
            # 파라미터 설정
            params = {'start_date': args.start_date, 'end_date': args.end_date, 'stock_code': stock_code, 'initial_balance': args.balance,
                      'chart_data': chart_data, 'train_data': train_data,
                      'params_path': params_path, 'models_path': archive_path}
            # 강화학습 테스트
            DQN = Network.DQN(**params)
            DQN.test(args.model_name)
    elif args.mode=='simulate':
        # 1. 상태 불러오기
        # [stock_code, [Environment observation, Agent state]] = [시가, 종가, 고가-저가, 거래량, 최대보유가능량, 현재 주식가치]
        os.system("conda activate py36_32") # 키움 api 사용을 위해 32비트 환경 활성화

        # 2. 상태에 따른 action, confidence 예측
        os.system("conda activate DLCourse") # keras 사용을 위해 32비트 환경 활성화
          # keras 사용을 위해 32비트 환경 활성화

        # 3. 행동
        os.system("conda activate py36_32") # 키움 api 사용을 위해 32비트 환경 활성화
    # mode == 'run'
    else:
        pass






