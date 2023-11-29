# import libraries
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# read the csv file
stock_data = pd.read_csv('C:/Users/doriro/Coding/Code/Algorithm_trading/005930.KS.csv')
stock_data.drop(['Adj Close'], axis=1, inplace=True) # adjusted close 삭제
stock_data['increase'] = stock_data['Close'] - stock_data['Open']
# 이후에 사용하기 위해 원래 'Open' 가격 저장
original_open = stock_data['Open'].values
original_increase = stock_data['increase'].values
# 날짜 분리하여 추후 그래프에 사용
dates = pd.to_datetime(stock_data['Date'])

# 학습에 사용할 변수
cols = list(stock_data)[1:7]

# 학습 데이터만 있는 새로운 데이터프레임 생성 - 5개 열
stock_data = stock_data[cols].astype(float)

# 데이터 정규화
scaler = StandardScaler()
scaler = scaler.fit(stock_data)
stock_data_scaled = scaler.transform(stock_data)
stock_data_scaled.shape[0]

# 학습 데이터와 테스트 데이터로 분리
n_train = int(0.9*stock_data_scaled.shape[0])
train_data_scaled = stock_data_scaled[0: n_train]
train_dates = dates[0: n_train]

test_data_scaled = stock_data_scaled[n_train:]
test_dates = dates[n_train:]

# LSTM에 입력하기 위한 데이터 형식 변환
pred_days = 1  # 예측 기간
seq_len = 14   # 시퀀스 길이 = 과거 일 수
input_dim = 6  # 입력 차원 = ['Open', 'High', 'Low', 'Close', 'Volume', 'increase']

trainX = []
trainY = []
testX = []
testY = []

for i in range(seq_len, n_train-pred_days +1):
    trainX.append(train_data_scaled[i - seq_len:i, 0:train_data_scaled.shape[1]])
    trainY.append(train_data_scaled[i + pred_days - 1:i + pred_days, 5])

for i in range(seq_len, len(test_data_scaled)-pred_days +1):
    testX.append(test_data_scaled[i - seq_len:i, 0:test_data_scaled.shape[1]])
    testY.append(test_data_scaled[i + pred_days - 1:i + pred_days, 5])

trainX, trainY = np.array(trainX), np.array(trainY)
testX, testY = np.array(testX), np.array(testY)

# LSTM 모델 생성
model = Sequential()
model.add(LSTM(64, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True)) # (시퀀스 길이, 입력 차원)
model.add(LSTM(32, return_sequences=False))
model.add(Dense(trainY.shape[1]))

model.summary()

# 학습률 설정
learning_rate = 0.01
# 지정된 학습률로 Adam 옵티마이저 생성
optimizer = Adam(learning_rate=learning_rate)
# 컴파일 시 옵티마이저와 손실 함수 설정
model.compile(optimizer=optimizer, loss='mse')

# 모델의 가중치 로드 시도
try:
    model.load_weights('./save_weights/lstm_weights.h5')
    print("Loaded model weights from disk")
except:
    print("No weights found, training model from scratch")


stock_data = pd.read_csv('C:/Users/doriro/Coding/Code/Algorithm_trading/test.csv')
stock_data.drop(['Adj Close'], axis=1, inplace=True) # adjusted close 삭제
stock_data['increase'] = stock_data['Close'] - stock_data['Open']
# 날짜 분리하여 추후 그래프에 사용
dates = pd.to_datetime(stock_data['Date'])


# 학습에 사용할 변수
cols = list(stock_data)[1:7]

# 학습 데이터만 있는 새로운 데이터프레임 생성 - 5개 열
stock_data = stock_data[cols].astype(float)

# 데이터 정규화
scaler = StandardScaler()
scaler = scaler.fit(stock_data)
stock_data_scaled = scaler.transform(stock_data)
stock_data_scaled.shape[0]

# 학습 데이터와 테스트 데이터로 분리

test_data_scaled = stock_data_scaled
test_dates = dates

# LSTM에 입력하기 위한 데이터 형식 변환
pred_days = 1  # 예측 기간
seq_len = 14   # 시퀀스 길이 = 과거 일 수

testX = []
testY = []
for i in range(seq_len, len(test_data_scaled)-pred_days +1):
    testX.append(test_data_scaled[i - seq_len:i, 0:test_data_scaled.shape[1]])
    testY.append(test_data_scaled[i + pred_days - 1:i + pred_days, 5])

testX, testY = np.array(testX), np.squeeze(np.array(testY))

prediction = model.predict(testX)
prediction = np.squeeze(prediction)
print(prediction)
print("--------------------------------")
print(testY)


# 예측 정확도 계산
accurate_predictions = np.sum(np.sign(prediction) == np.sign(testY))
total_predictions = len(prediction)
accuracy = (accurate_predictions / total_predictions) * 100

print(f"예측 정확도: {accuracy:.2f}%")