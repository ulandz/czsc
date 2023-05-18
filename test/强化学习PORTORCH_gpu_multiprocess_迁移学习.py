import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import math
import statsmodels.api as sm
from numpy import *
from math import sqrt
from pandas import *
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pickle import dump
from sklearn.model_selection import train_test_split
from hurst import compute_Hc, random_walk
from pykalman import KalmanFilter
import os
import gym
# from vnpy.trader.constant import Direction, Offset
# from vnpy.trader.object import OrderRequest, CancelRequest
# from vnpy.trader.utility import load_json, save_json

# from vnpy_ctastrategy.backtesting import BacktestingEngine,OptimizationSetting
# from vnpy_ctastrategy.strategies import test_strategy
## import data
from PyQt5 import QtWidgets
def parse(indata,ptype):
    # datas = open(input_file_name,"r").readlines()
    datas=list(indata)

    datas = [i.strip() for i in datas]

    HEADER = datas[0]

    datas = datas[1:]

    datas = [i.split(",") for i in datas]


    current_ptr = {}
    current_ptr['date'] = datas[0][0]
    current_ptr['open'] = []
    current_ptr['high'] = []
    current_ptr['low'] = []
    current_ptr['close'] = []
    current_ptr['volume'] = []
    current_ptr['open_oi']=[]

    results = []
    for i in datas:
        if i[0] != current_ptr['date']:
            #submit
            line = current_ptr['date'] + "," + str(current_ptr['open'][0]) + "," + str(max(current_ptr["high"])) + "," \
                + str(min(current_ptr['low'])) + "," + str(current_ptr['close'][-1]) + "," + str(sum(current_ptr["volume"]))+ \
                    "," + str(sum(current_ptr["open_oi"]))
            results.append(line)
            #new ptr
            current_ptr = {}
            current_ptr['date'] = i[0]
            current_ptr['open'] = []
            current_ptr['high'] = []
            current_ptr['low'] = []
            current_ptr['close'] = []
            current_ptr['volume'] = []
        current_ptr['open'].append(int(i[1]))
        current_ptr['high'].append(int(i[2]))
        current_ptr['low'].append(int(i[3]))
        current_ptr['close'].append(int(i[4]))
        current_ptr['volume'].append(int(i[5]))

    # with open(output_file_name,"w") as file:
    #     file.write(HEADER)
    #     file.write("\n")
    #     for item in results:
    #         file.write(item)
    #         file.write("\n")

def pre_data_process(origin_data):
        # print(origin_data.columns)
        origin_data.rename(columns={'amount': 'turnover'}, inplace=True)
        # origin_data.rename(columns={'ts_code': 'code'}, inplace=True)
        
        origin_data.drop_duplicates(['trade_time'], keep='first', inplace=True)
        origin_data.rename(columns={'trade_time': 'date'}, inplace=True)

        return origin_data 
def clean_dataframe(df):
    """
    清洗DataFrame：
    - 删除包含NaN和infinity的行；
    - 将非数值（如空白或字符串）转换为NaN
    """
    
    df=df.drop('date', axis=1)
    df=df.drop('ts_code', axis=1)
    # df=df.set_index('stamp')
    
    
    # 将所有非数字转换成 NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # 删除包含 NaN 和 infinity 的行
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    
    
    return df
def parse_date(data):
    data['year']=data['date'].dt.year
    data['month']=data['date'].dt.month
    data['day']=data['date'].dt.day
    data['week']=data['date'].dt.isocalendar().week
    data['weekday']=data['date'].dt.weekday
    data['hour']=data['date'].dt.hour
    data['minute']=data['date'].dt.minute
    data['second']=data['date'].dt.second
def make_bigdata(indata,ptype):
    # datas = open(input_file_name,"r").readlines()
    
    datas=list(indata)

    datas = [i.strip() for i in datas]

    HEADER = datas[0]

    datas = datas[1:]

    datas = [i.split(",") for i in datas]


    current_ptr = {}
    current_ptr['date'] = datas[0][0]
    current_ptr['open'] = []
    current_ptr['high'] = []
    current_ptr['low'] = []
    current_ptr['close'] = []
    current_ptr['volume'] = []
    current_ptr['open_oi']=[]

    results = []
    for i in datas:
        if i[0] != current_ptr['date']:
            #submit
            line = current_ptr['date'] + "," + str(current_ptr['open'][0]) + "," + str(max(current_ptr["high"])) + "," \
                + str(min(current_ptr['low'])) + "," + str(current_ptr['close'][-1]) + "," + str(sum(current_ptr["volume"]))+ \
                    "," + str(sum(current_ptr["open_oi"]))
            results.append(line)
            #new ptr
            current_ptr = {}
            current_ptr['date'] = i[0]
            current_ptr['open'] = []
            current_ptr['high'] = []
            current_ptr['low'] = []
            current_ptr['close'] = []
            current_ptr['volume'] = []
        current_ptr['open'].append(int(i[1]))
        current_ptr['high'].append(int(i[2]))
        current_ptr['low'].append(int(i[3]))
        current_ptr['close'].append(int(i[4]))
        current_ptr['volume'].append(int(i[5]))

    # with open(output_file_name,"w") as file:
    #     file.write(HEADER)
    #     file.write("\n")
    #     for item in results:
    #         file.write(item)
    #         file.write("\n")
# parse('DATA.csv')




def zdc_calu_hurst(data):
    Hlist=[]
    close=data['close']
    length = len(data)
    for i in range(0,121,1):
        Hlist.append(0)    
    for i in range(121, length, 1):
        
        H, c, datax =compute_Hc(close[i-120:i], kind='price',max_window=120, simplified=True)
        Hlist.append(H)   
 
 
    return Hlist
def calu_hurst(ts=None, lags=None):
    """
    Returns the Hurst Exponent of the time series
    hurst < 0.5: mean revert
    hurst = 0.5: random
    hurst > 0.5: trend
    :param:
        ts[,]   a time-series, with 100+ elements
    :return:
        float - a Hurst Exponent approximation
    """
    if ts is None:
        ts = [None, ]
    if lags is None:
        lags = [2, 80]

    if isinstance(ts, pd.Series):
        ts = ts.dropna().to_list()

    too_short_list = lags[1] + 1 - len(ts)
    if 0 < too_short_list:  # IF NOT:
        # 序列長度不足則以第一筆補滿
        ts = too_short_list * ts[:1] + ts  # PRE-PEND SUFFICIENT NUMBER of [ts[0],]-as-list REPLICAS TO THE LIST-HEAD
    # Create the range of lag values
    lags = range(lags[0], lags[1])
    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    # Return the Hurst exponent from the polyfit output ( a linear fit to estimate the Hurst Exponent
    return 2.0 * np.polyfit(np.log(lags), np.log(tau), 1)[0]    
# Calculate technical indicators

def vol_feature(d):
    """
    add 2 volume feature to df
    vol_deg_change: degree change in minmax scale, origin value belong to [-90, 90]
    vol_percentile: volume percentile
    :param d: df
    """
    df=d.copy()
    df['vol_deg_change'] = d['volume'].diff(1).apply(lambda x: (np.arctan2(x, 100) / np.pi) + 0.5).round(4)
    df['vol_percentile'] = d['volume'].rolling(len(d), min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).values[-1], raw=False).round(3)
    return df

def oi_feature(d):
    """
    same as vol_feature, for futures
    :param d: df
    """
    df=d.copy()
    df['oi_deg_change'] = d['open_oi'].diff(1).apply(lambda x: (np.arctan2(x, 100) / np.pi) + 0.5).round(4)
    df['oi_percentile'] = d['open_oi'].rolling(len(d), min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).values[-1], raw=False).round(3)
    return df

def kalman(ts=None):
    if ts is None:
        ts = [None, ]
    if ts[0] is None:
        return
    kf = KalmanFilter(initial_state_mean=0,
                      initial_state_covariance=1,
                      transition_matrices=[1],
                      observation_matrices=[1],
                      observation_covariance=1,
                      transition_covariance=.01)
    state_means, _ = kf.filter(ts)
    state_means = pd.Series(state_means.flatten(), index=ts.index)
    return state_means
def candle(d):
    """
    add candlestick feature to df
    body: abs(o - c)
    upper_shadow: h - max(o, c)
    lower_shadow: min(o, c) - l
    divide by range in order to make all candlestick in same scale
    :param d: df
    """
    df=d.copy()
    df['range'] = d['high'] - d['low']
    df['body'] = (np.abs(d['open'] - d['close']) / d['range']).fillna(1).round(2)
    df['upper_shadow'] = ((d['high'] - d[['open', 'close']].max(axis=1)) / d['range']).fillna(0).round(2)
    df['lower_shadow'] = ((d[['open', 'close']].min(axis=1) - d['low']) / d['range']).fillna(0).round(2)
    return df.drop(columns=['range'])
def norm_ohlc(d):
    """
    normalize ohlc
    :param d: df
    """
    df=d.copy()
    tmp_o = np.log(d['open'])
    df['norm_o'] = (tmp_o - np.log(d['close'].shift(1))).round(4)
    df['norm_h'] = (np.log(d['high']) - tmp_o).round(4)
    df['norm_l'] = (np.log(d['low']) - tmp_o).round(4)
    df['norm_c'] = (np.log(d['close']) - tmp_o).round(4)

    return df
def calc_moving_avg(df, window):
    return df['close'].rolling(window=window).mean()

def calc_bollinger_bands(df, window, n_std=2):
    rolling_mean = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    upper_band = rolling_mean + n_std * rolling_std
    lower_band = rolling_mean - n_std * rolling_std
    return upper_band, lower_band

def calc_macd(df, s=12, l=26, m=9):
    ema_s = df['close'].ewm(span=s, adjust=False).mean()
    ema_l = df['close'].ewm(span=l, adjust=False).mean()
    macd = ema_s - ema_l
    signal_line = macd.ewm(span=m, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram



def get_technical_indicators(df):
    datao=df.copy()
    #别人的方式
    datao['ma7'] = calc_moving_avg(df, 7)
    datao['ma21'] = calc_moving_avg(df, 21)
    datao['ma30'] = calc_moving_avg(df, 30)
    datao['ma60'] = calc_moving_avg(df, 60)
    datao['ma120'] = calc_moving_avg(df, 120)
    datao['ma240'] = calc_moving_avg(df, 240)
    datao['ma480'] = calc_moving_avg(df, 480)
    df['ma21']=datao['ma21']
    datao['upper_band'], datao['lower_band'] = calc_bollinger_bands(df, window=20)
    datao['20sd'] = (df['close'] - df['ma21']).rolling(window=20).std()
    datao['macd'], df['signal'], df['histogram'] = calc_macd(df)
    #我的方式


    #Create Exponential moving average
    datao['ema'] = df['close'].ewm(com=0.5).mean()
    datao['weekday']=df['date'].dt.weekday
    datao['hour']=df['date'].dt.hour
    datao['minute']=df['date'].dt.minute
    # Create LogMomentum
    # data['logmomentum'] = np.log(data['close'] - 1)
    #compute hurst 
    # data['hurst']=data.iloc[:, 4].rolling(120).apply(lambda x: calu_hurst(x)).round(1)
    # data['kalman_log_rtn1'] = np.log(kalman(data.iloc[:, 4])).diff(1).round(5)
    # data=vol_feature(data)
    # data=oi_feature(data)
    # data=candle(data)
    #data=norm_ohlc(data)

    return datao


#Drop the first 21 rows
#For doing the fourier
# dataset = T_df.iloc[20:,:].reset_index(drop=True)
#去掉前479个数据,因为取了480
# dataset = T_df.iloc[479:,:].reset_index(drop=True)


#Getting the Fourier transform features
def get_fourier_transfer(dataset):
    # Get the columns for doing fourier
    data_FT = dataset[['date', 'close']]

    close_fft = np.fft.fft(np.asarray(data_FT['close'].tolist()))
    fft_df = pd.DataFrame({'fft': close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

    fft_list = np.asarray(fft_df['fft'].tolist())
    fft_com_df = pd.DataFrame()
    for num_ in [3, 6, 9]:
        fft_list_m10 = np.copy(fft_list)
        fft_list_m10[num_:-num_] = 0
        fft_ = np.fft.ifft(fft_list_m10)
        fft_com = pd.DataFrame({'fft': fft_})
        fft_com['absolute of ' + str(num_) + ' comp'] = fft_com['fft'].apply(lambda x: np.abs(x))
        fft_com['angle of ' + str(num_) + ' comp'] = fft_com['fft'].apply(lambda x: np.angle(x))
        fft_com = fft_com.drop(columns='fft')
        fft_com_df = pd.concat([fft_com_df, fft_com], axis=1)

    return fft_com_df

#Get Fourier features
def get_final_data(origin_data):
        df=pre_data_process(origin_data)
        dt_temp=df.iloc[0:2000,:]
        
        # parse_date(dt_temp)
        # print(dt_temp)
        
        data_indicate=get_technical_indicators(dt_temp)
        #为了数据的完整性,去掉空白的480个MACD480
        dataset = data_indicate.iloc[479:,:].reset_index(drop=True)
        
        #不要傅里叶变换了
        # dataset_f = get_fourier_transfer(dataset)
        # Final_data = pd.concat([dataset, dataset_f], axis=1)
        
        return  clean_dataframe(dataset)
#Get Fourier features
def get_test_data(origin_data):
        df=pre_data_process(origin_data)
        dt_temp=df.iloc[5000:5500,:]
        
        # parse_date(dt_temp)
        # print(dt_temp)
        
        data_indicate=get_technical_indicators(dt_temp)
        #为了数据的完整性,去掉空白的480个MACD480
        dataset = data_indicate.iloc[479:,:].reset_index(drop=True)
        
        #不要傅里叶变换了
        # dataset_f = get_fourier_transfer(dataset)
        # Final_data = pd.concat([dataset, dataset_f], axis=1)
        
        return  clean_dataframe(dataset)

# Final_data = clean_dataframe(get_final_data(df))

import numpy as np
import pandas as pd

class SimpleStockTradingEnv:
    def __init__(self, data, initial_balance=10000, window_size=10):
        self.data = data
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = self.window_size
        self.done = False
        return self._get_state()

    def step(self, action):
        if self.done:
            raise RuntimeError("Cannot call step() in a finished episode.")

        prev_balance = self.balance

        if action == 0:  # 买入
            self.balance -= self.data['close'][self.current_step]
        elif action == 1:  # 卖出
            self.balance += self.data['close'][self.current_step]

        self.current_step += 1
        if self.current_step >= len(self.data):
            self.done = True

        reward = self.balance - prev_balance
        state = self._get_state()
        return state, reward, self.done, {}

    def _get_state(self):
        return self.data.iloc[self.current_step - self.window_size:self.current_step].values.flatten()



import torch
import torch.nn as nn
import torch.optim as optim
import concurrent.futures

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, input_size, output_size, device):
        self.device = device
        self.model = DQN(input_size, output_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = nn.MSELoss()

    def train(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        q_values = self.model(state)
        next_q_values = self.model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + (1 - done) * next_q_value

        loss = self.loss_fn(q_value, expected_q_value.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        q_values = self.model(state)
        return q_values.argmax().item()

def train_agent(agent, env, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.train(state, action, reward, next_state, done)
            state = next_state

def evaluate_agent(agent, env, num_episodes):
    total_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)
model_path= os.path.join(os.getcwd(), 'model.pth')  # 模型保存路径 （不要带后缀） 或 不写保存路
def check_model(model_path):
    
    file_path = model_path
    if os.path.exists(file_path):
        return 1
    else:
        return 0

def trainit(data):

    # 创建环境
    env = SimpleStockTradingEnv(data)

    # 创建智能体
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = env.window_size * data.shape[1]
    output_size = 3  # 买入、卖出、持有
    agent = DQNAgent(input_size, output_size, device)
    

    # 加载模型
    if check_model==1:
        agent.model.load_state_dict(torch.load(model_path))
    # 多线程训练
    num_threads = 4
    num_episodes_per_thread = 250
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(train_agent, agent, env, num_episodes_per_thread) for _ in range(num_threads)]
        concurrent.futures.wait(futures)

    # 评估收益
    num_evaluation_episodes = 100
    average_reward = evaluate_agent(agent, env, num_evaluation_episodes)
    print(f"平均收益：{average_reward}")
    # 保存模型
    torch.save(agent.model.state_dict(), model_path)


def predict(model_path, data):
    # 创建环境
    env = SimpleStockTradingEnv(data)

    # 创建智能体
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = env.window_size * data.shape[1]
    output_size = 3  # 买入、卖出、持有
    agent = DQNAgent(input_size, output_size, device)

    # 加载模型
    agent.model.load_state_dict(torch.load(model_path))

    # 进行预测
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, _, done, _ = env.step(action)
        state = next_state

    # 返回最终收益
    return env.balance


import backtrader as bt
class btDQNStrategy(bt.Strategy):
    params = (
        ('window_size', 10),
        ('model_path', 'model.pth'),
    )

    def __init__(self):
        self.data_close = self.datas[0].close
        self.window_size = self.params.window_size
        self.model_path = self.params.model_path
        self.agent = DQNAgent(self.window_size, 2, torch.device('cpu'))
        self.agent.model.load_state_dict(torch.load(self.model_path))

    def next(self):
        if len(self) < self.window_size:
            return

        state = self._get_state()
        action = self.agent.get_action(state)
        if action == 0:  # 买入
            self.buy()
        elif action == 1:  # 卖出
            self.sell()

    def _get_state(self):
        return self.data_close.get(size=self.window_size)

class PandasData(bt.feeds.PandasData):
    """
    Define pandas DataFrame structure
    """
    lines = ('datetime','open', 'high', 'low', 'close', 'volume', 'turnover', 'open_oi', 'ma7', 'ma21', 'ma30', 'ma60', 'ma120', 'ma240', 'ma480', 'macd', '20sd', 'upper_band', 'lower_band', 'ema', 'weekday', 'hour', 'minute','stamp')

    params = (
        ('datetime', 'date'),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('turnover', 'turnover'),
        ('open_oi', 'open_oi'),
        ('ma7', 'ma7'),
        ('ma21', 'ma21'),
        ('ma30', 'ma30'),
        ('ma60', 'ma60'),
        ('ma120', 'ma120'),
        ('ma240', 'ma240'),
        ('ma480', 'ma480'),
        ('macd', 'macd'),
        ('20sd', '20sd'),
        ('upper_band', 'upper_band'),
        ('lower_band', 'lower_band'),
        ('ema', 'ema'),
        ('weekday', 'weekday'),
        ('hour', 'hour'),
        ('minute', 'minute'),
        ('stamp', 'stamp'),
    )


def run_backtest(indata):
    indata['datetime'] = pd.to_datetime(indata['stamp'])
    cerebro = bt.Cerebro()
    data = PandasData(dataname=indata)
    cerebro.adddata(data)
    cerebro.addstrategy(btDQNStrategy)
    cerebro.broker.set_cash(100000)
    cerebro.broker.setcommission(commission=0.001)    
    cerebro.run()
    cerebro.plot()




def trade(model_path, data):
    # 创建环境
    env = SimpleStockTradingEnv(data)

    # 创建智能体
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = env.window_size * data.shape[1]
    output_size = 3  # 买入、卖出、持有
    agent = DQNAgent(input_size, output_size, device)

    # 加载模型
    agent.model.load_state_dict(torch.load(model_path))

    # 进行交易
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, _, done, _ = env.step(action)
        state = next_state

    # 返回交易指示
    return env.actions  # 可以是：hold, buy, short, long.  或者：None(未指示交易) 或者：空


def prepare_train_data(file_path):
    df = pd.read_hdf(file_path, parse_dates=['trade_time'])
    # print(df)
    # print(df.columns)
    outdata=get_final_data(df)
    
    print('traindate')
    print(outdata.columns)
    print(outdata)
    return outdata
def prepare_test_data(file_path):
    df = pd.read_hdf(file_path, parse_dates=['trade_time'])

    outdata=get_test_data(df)
    
    print('testdat')
    print(outdata.columns)
    print(outdata)
    return outdata

def traverse_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            traindata = prepare_train_data(os.path.join(root, file))
            testdata = prepare_test_data(os.path.join(root, file))
            #训练数据时
            trainit(traindata)
            #发出交易指示
            # trade(data)
            #vnpy实盘对接
            # vnpytrade(data)
            #baktrader回测
            run_backtest(testdata)


fileplace = "D:\\Weisoft Stock(x64)\\qihuo\\min1\\"

def main():
    
    traverse_directory(fileplace)

from multiprocessing import  set_start_method


# 添加 __name__ == '__main__' 判断和 freeze_support() 函数调用
if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    from multiprocessing import freeze_support
    main()