# import libraries
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
from datetime import datetime, timedelta

# LSTM 모델 생성
model = Sequential()
model.add(LSTM(64, input_shape=(60, 6), return_sequences=True)) # (시퀀스 길이, 입력 차원)
model.add(LSTM(32, return_sequences=False))
model.add(Dense(1))

model.summary()

# 모델의 가중치 로드 시도
try:
    model.load_weights('./save_weights/lstm_weights.h5')
    print("Loaded model weights from disk")
except:
    print("No weights found, training model from scratch")

today = datetime.today()
start_date = (today - timedelta(days=100)).date()
end_date = today.date()
stock_data = fdr.DataReader('005930', start=start_date, end=end_date)


# 학습에 사용할 변수
cols = list(stock_data)

# 학습 데이터만 있는 새로운 데이터프레임 생성 - 5개 열
stock_data = stock_data[cols].astype(float)

# 데이터 정규화
scaler = StandardScaler()
scaler = scaler.fit(stock_data)
stock_data_scaled = scaler.transform(stock_data)
stock_data_scaled.shape[0]

# 학습 데이터와 테스트 데이터로 분리
# 데이터는 3달치가 입력되면 될듯 
test_data_scaled = stock_data_scaled

# LSTM에 입력하기 위한 데이터 형식 변환
pred_days = 1  # 예측 기간
seq_len = 60   # 시퀀스 길이 = 과거 일 수

testX = []
testY = []
for i in range(seq_len, len(test_data_scaled)-pred_days +1):
    testX.append(test_data_scaled[i - seq_len:i, 0:test_data_scaled.shape[1]])
    testY.append(test_data_scaled[i + pred_days - 1:i + pred_days, 5])

testX, testY = np.array(testX), np.squeeze(np.array(testY))
testY = testY[1:]

prediction = model.predict(testX)
prediction = np.squeeze(prediction)

opt = prediction[-1]

if (opt > 0):
    print("BUY")
else:
    print("NO")