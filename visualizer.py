import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
from matplotlib.ticker import LinearLocator, FuncFormatter
import matplotlib.dates as mdates
from Agent import Agent


class Visualizer:


    def __init__(self, vnet=False):
        self.canvas = None
        # 캔버스 같은 역할을 하는 Matplotlib의 Figure 클래스 객체
        self.fig = None
        # 차트를 그리기 위한 Matplotlib의 Axes 클래스 객체
        self.axes = None
        self.title = ''  # 그림 제목

    def prepare(self, title, chart_data):
        x = np.arange(len(chart_data))  # 모든 차트가 공유하는 x축

        # 캔버스를 초기화하고 5개의 차트를 그릴 준비
        self.fig, self.axes = plt.subplots(nrows=3, ncols=1, facecolor='w', sharex=True,
                                           gridspec_kw={
                                               'width_ratios': [1],
                                               'height_ratios': [5, 2, 3]}
                                           )
        self.fig.suptitle(title)
        for ax in self.axes:
            # 과학적 표기 비활성화
            ax.get_xaxis().get_major_formatter().set_scientific(False)
            ax.get_yaxis().get_major_formatter().set_scientific(False)
            # y axis 위치 오른쪽으로 변경
            ax.yaxis.tick_right()
            # tick 단위 조절
            ax.get_yaxis().set_major_locator(LinearLocator(numticks=5))

        # 차트 1. 일봉 차트
        self.axes[0].set_ylabel('Action & Price')  # y 축 레이블 표시
        # date, open, high, low, close 순서로된 2차원 배열
        ohlc = np.array(chart_data[['open', 'high', 'low', 'close']])
        dohlc = np.hstack((np.reshape(x, (-1, 1)), ohlc))
        # 양봉은 빨간색으로 음봉은 파란색으로 표시
        candlestick_ohlc(self.axes[0], dohlc, colorup='r', colordown='b')
        self.axes[0].get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        # 거래량 가시화
        self.axes[1].get_yaxis().set_ticks([])
        self.axes[1].set_ylabel('Volume')  # y 축 레이블 표시
        volume = list(chart_data.loc[:,'volume'])
        self.axes[1].bar(x, volume, color='b', alpha=0.3)

    def plot_train(self):
        pass

    def plot_result(self, chart_data):
        x = np.arange(len(chart_data))  # 모든 차트가 공유하는 x축
        action_list = [0, 1, 2]
        action_name = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
        actions = np.array(chart_data.loc[:, ['action']]).reshape((-1,)) #0 : buy, 1 : sell 2: hold
        unit_price = np.array(chart_data.loc[:, ['unit_price']]).reshape((-1,))
        profit_stockwise = np.array(chart_data.loc[:, ['profit_stockwise']]).reshape((-1,))
        profit_pvwise = np.array(chart_data.loc[:, ['profit_pvwise']]).reshape((-1,))
        portfolio_value = np.array(chart_data.loc[:, ['portfolio_value']]).reshape((-1,))
        kospi_ratio = np.array(chart_data.loc[:, ['kospi_ratio']]).reshape((-1,))

        # 차트 2. 에이전트 상태 (행동, 주식단가, 이익률, pv)
        COLORS = ['lightcoral', 'cornflowerblue', 'lightgreen']
        for action, color in zip(action_list, COLORS):
            for i, v in enumerate(actions == action):
                if v==True:
                    # 배경 색으로 행동 표시
                    self.axes[0].axvline(i, color=color, linewidth = 4, alpha=0.3)
        self.axes[0].plot(x, unit_price, '-k', alpha = 0.5, label = 'unit_price')  # 주식단가 그리기
        # 차트 3. 포트폴리오 가치, 코스피 비교
        self.axes[2].set_ylabel("Profit")
        self.axes[2].get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: '{:.2f}%'.format(x*100)))
        self.axes[2].plot(x, profit_pvwise, '-k', label = 'profit_pv')
        self.axes[2].fill_between(x, profit_pvwise, 0, where=profit_pvwise >= 0, facecolor='r', alpha=0.1)
        self.axes[2].fill_between(x, profit_pvwise, 0, where=profit_pvwise < 0, facecolor='b', alpha=0.1)
        self.axes[2].plot(x, kospi_ratio, '-y', label = 'kospi_ratio')
        # x축 날짜 label
        _xticks = []
        _xlabels = []
        for _x, d in zip(x, chart_data.date.values):
            weekday = datetime.datetime.strptime(str(d), '%Y%m%d').weekday()
            if weekday <= 0: #월요일인 경우만 추가
                _xticks.append(_x)
                _xlabels.append(datetime.datetime.strptime(str(d), '%Y%m%d').strftime('%Y/%m/%d'))
        self.axes[-1].set_xticks(_xticks)
        self.axes[-1].set_xticklabels(_xlabels, rotation=45, minor=False)

        # 범례 설정
        lines = []
        labels = []
        for ax in self.axes:
            axLine, axLabel = ax.get_legend_handles_labels()
            lines.extend(axLine)
            labels.extend(axLabel)

        self.fig.legend(lines, labels, loc='upper right')
        plt.show()

    def plot(self, epoch_str=None, num_epoches=None, epsilon=None,
             action_list=None, actions=None, num_stocks=None,
             outvals_value=[], outvals_policy=[], exps=None,
             learning_idxes=None, initial_balance=None, pvs=None):

        x = np.arange(len(actions))  # 모든 차트가 공유할 x축 데이터
        actions = np.array(actions)  # 에이전트의 행동 배열
        # 초기 자본금 배열
        pvs_base = np.zeros(len(actions)) + initial_balance

        # 차트 2. 에이전트 상태 (행동, 보유 주식 수)
        for action, color in zip(action_list, self.COLORS):
            for i in x[actions == action]:
                # 배경 색으로 행동 표시
                self.axes[1].axvline(i, color=color, alpha=0.1)
        self.axes[1].plot(x, num_stocks, '-k')  # 보유 주식 수 그리기

        # 차트 3. 가치 신경망
        if len(outvals_value) > 0:
            max_actions = np.argmax(outvals_value, axis=1)
            for action, color in zip(action_list, self.COLORS):
                # 배경 그리기
                for idx in x:
                    if max_actions[idx] == action:
                        self.axes[2].axvline(idx,
                                             color=color, alpha=0.1)
                # 가치 신경망 출력의 tanh 그리기
                self.axes[2].plot(x, outvals_value[:, action],
                                  color=color, linestyle='-')

        # 차트 4. 정책 신경망
        # 탐험을 노란색 배경으로 그리기
        for exp_idx in exps:
            self.axes[3].axvline(exp_idx, color='y')
        # 행동을 배경으로 그리기
        _outvals = outvals_policy if len(outvals_policy) > 0 \
            else outvals_value
        for idx, outval in zip(x, _outvals):
            color = 'white'
            if np.isnan(outval.max()):
                continue
            if outval.argmax() == Agent.ACTION_BUY:
                color = 'r'  # 매수 빨간색
            elif outval.argmax() == Agent.ACTION_SELL:
                color = 'b'  # 매도 파란색
            self.axes[3].axvline(idx, color=color, alpha=0.1)
        # 정책 신경망의 출력 그리기
        if len(outvals_policy) > 0:
            for action, color in zip(action_list, self.COLORS):
                self.axes[3].plot(
                    x, outvals_policy[:, action],
                    color=color, linestyle='-')

        # 차트 5. 포트폴리오 가치
        self.axes[4].axhline(
            initial_balance, linestyle='-', color='gray')
        self.axes[4].fill_between(x, pvs, pvs_base,
                                  where=pvs > pvs_base, facecolor='r', alpha=0.1)
        self.axes[4].fill_between(x, pvs, pvs_base,
                                  where=pvs < pvs_base, facecolor='b', alpha=0.1)
        self.axes[4].plot(x, pvs, '-k')
        # 학습 위치 표시
        for learning_idx in learning_idxes:
            self.axes[4].axvline(learning_idx, color='y')

        # 에포크 및 탐험 비율
        self.fig.suptitle('{} \nEpoch:{}/{} e={:.2f}'.format(
            self.title, epoch_str, num_epoches, epsilon))
        # 캔버스 레이아웃 조정
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.85)

    def clear(self, xlim):
        _axes = self.axes.tolist()
        for ax in _axes[1:]:
            ax.cla()  # 그린 차트 지우기
            ax.relim()  # limit를 초기화
            ax.autoscale()  # 스케일 재설정
        # y축 레이블 재설정
        self.axes[1].set_ylabel('Agent')
        self.axes[2].set_ylabel('V')
        self.axes[3].set_ylabel('P')
        self.axes[4].set_ylabel('PV')
        for ax in _axes:
            ax.set_xlim(xlim)  # x축 limit 재설정
            ax.get_xaxis().get_major_formatter().set_scientific(False)  # 과학적 표기 비활성화
            ax.get_yaxis().get_major_formatter().set_scientific(False)  # 과학적 표기 비활성화
            # x축 간격을 일정하게 설정
            ax.ticklabel_format(useOffset=False)

    def save(self, path):
       self.fig.savefig(path)


if __name__ == '__main__':
    fname = '005930_20201129232202_20100101_20180430.h5_20190701_20190830.csv'
    chart_data = pd.read_csv(f"./test_report/{fname}")
    visualizer = Visualizer()
    visualizer.prepare(title = '005930(train: 10/01/01~18/04/30)', chart_data = chart_data)
    visualizer.plot_result(chart_data)