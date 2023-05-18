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

import torch

if torch.cuda.is_available():
    print('CUDA is available')
    print('Device:', torch.cuda.get_device_name(0))
    print('CUDA version:', torch.version.cuda)
    print('cuDNN version:', torch.backends.cudnn.version())
else:
    print('CUDA not available')
# from vnpy.trader.constant import Direction, Offset
# from vnpy.trader.object import OrderRequest, CancelRequest
# from vnpy.trader.utility import load_json, save_json

# from vnpy_ctastrategy.backtesting import BacktestingEngine,OptimizationSetting
# from vnpy_ctastrategy.strategies import test_strategy
## import data
# from PyQt5 import QtWidgets
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
    
    # df=df.drop('date', axis=1)
    df=df.drop('ts_code', axis=1)
    # df=df.set_index('stamp')
    
    
    # 将所有非数字转换成 NaN
    # df = df.apply(pd.to_numeric, errors='coerce')

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





fileplace = "D:\\Weisoft Stock(x64)\\qihuo\\min1\\"
onefile="D:\\Weisoft Stock(x64)\\qihuo\\min1\\SQAG00min1.H5"
#**********************
import backtrader as bt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 将数据加载到 Pandas DataFrame 中
data = prepare_train_data(onefile)
data=data.set_index('date')
# 构建训练集和测试集
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# 定义 PyTorch 数据集
class MyDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.features = data[['ma7', 'ma21', 'ma30', 'ma60', 'ma120', 'ma240', 'ma480',
                              'upper_band', 'lower_band', '20sd', 'macd', 'ema', 'weekday', 'hour', 'minute']]
        self.targets = data['close']

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = torch.tensor(self.features.values[index], dtype=torch.float32)
        target = torch.tensor(self.targets.values[index], dtype=torch.float32)
        return feature, target

# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(15, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义策略
class MyStrategy(bt.Strategy):
    params = (('printlog', False),
              ('ma_7', 7),
              ('ma_21', 21),
              ('ma_30', 30),
              ('ma_60', 60),
              ('ma_120', 120),
              ('ma_240', 240),
              ('ma_480', 480))

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.model = Net()
        # 将模型移到GPU上
        self.model = self.model.to('cuda')
        # 定义损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def next(self):
        # 获取当前特征因子
        features = [self.data.ma_7[0], self.data.ma_21[0], self.data.ma_30[0],
                    self.data.ma_60[0], self.data.ma_120[0], self.data.ma_240[0],
                    self.data.ma_480[0], self.data.upper_band[0], self.data.lower_band[0],
                    self.data['20sd'][0], self.data.macd[0], self.data.ema[0],
                    int(self.data.weekday[0]), int(self.data.hour[0]), int(self.data.minute[0])]
        # 在GPU上计算预测值
        with torch.no_grad():
            input_tensor = torch.tensor(features, dtype=torch.float32).reshape(1, -1)
            input_tensor = input_tensor.to('cuda')
            predicted_price = self.model(input_tensor).item()

        if self.dataclose[0] < predicted_price:
            # 如果当前价格小于预测价格，买入
            self.buy()
        elif self.dataclose[0] > predicted_price:
            # 如果当前价格大于预测价格，卖出
            self.sell()

        if self.params.printlog:
            print('Close:', self.dataclose[0])
            print('Predicted Price:', predicted_price)

    def stop(self):
        torch.cuda.empty_cache()  # 释放 GPU 缓存

# 初始化 Cerebro 引擎
cerebro = bt.Cerebro(stdstats=False, maxcpus=4)

# 添加数据到引擎中
data_feed = bt.feeds.PandasData(dataname=data)
cerebro.adddata(data_feed)

# 设置策略
strat = MyStrategy
cerebro.addstrategy(strat)

# 运行回测
cerebro.run()
cerebro.plot()
# 在测试数据集上评估模型
test_dataset = MyDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=64)

with torch.no_grad():
    total_loss = 0
    for inputs, targets in test_loader:
        # 在GPU上计算预测值
        inputs = inputs.to('cuda')
        targets = targets.to('cuda')
        predicted_prices = strat.model(inputs).squeeze()
        loss = strat.criterion(predicted_prices, targets)
        total_loss += loss.item()*inputs.size()[0]

    print('MSE on Test Data:', total_loss/len(test_data))

# 在实盘上运行策略
# 将代码上传到云服务器或部署在本地计算机上，并以适当的方式自动化交易。 





#***********************

def traverse_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            traindata = prepare_train_data(os.path.join(root, file))
            testdata = prepare_test_data(os.path.join(root, file))
            #训练数据时
            # trainit(traindata)
            #发出交易指示
            # trade(data)
            #vnpy实盘对接
            # vnpytrade(data)
            #baktrader回测
            # run_backtest(testdata)
def main():
    
    traverse_directory(fileplace)

# from multiprocessing import  set_start_method


# # 添加 __name__ == '__main__' 判断和 freeze_support() 函数调用
# if __name__ == '__main__':
#     try:
#         set_start_method('spawn')
#     except RuntimeError:
#         pass
#     from multiprocessing import freeze_support
#     main()
