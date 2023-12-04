# import libraries
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# read the csv file
stock_data = pd.read_csv('./005930.KS.csv')
stock_data.drop(['Adj Close'], axis=1, inplace=True) # adjusted close 삭제
stock_data['Change'] = stock_data['Close'] - stock_data['Open']
# 이후에 사용하기 위해 원래 'Open' 가격 저장
original_open = stock_data['Open'].values
original_increase = stock_data['Change'].values
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
seq_len = 60   # 시퀀스 길이 = 과거 일 수
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
learning_rate = 0.004
# 지정된 학습률로 Adam 옵티마이저 생성
optimizer = Adam(learning_rate=learning_rate)
# 컴파일 시 옵티마이저와 손실 함수 설정
model.compile(optimizer=optimizer, loss='mse')

# 모델의 가중치 로드 시도
try:
    print("No weights found, training model from scratch")
    # 모델 학습
    history = model.fit(trainX, trainY, epochs=70, batch_size=32, validation_split=0.1, verbose=1)
    # 학습 후 모델 가중치 저장
    model.save_weights('./save_weights/lstm_weights.h5')

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()
except:
    print("none")

# 예측
prediction = model.predict(testX)
prediction = np.squeeze(prediction)
print(prediction.shape, testY.shape)

# 예측 결과를 평균값으로 채운 배열 생성
mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0)

# 첫 번째 열에 예측 값을 대입
mean_values_pred[:, 0] = np.squeeze(prediction)

# 역변환
y_pred = scaler.inverse_transform(mean_values_pred)[:,0]
print(y_pred.shape)

# 테스트 데이터의 평균값으로 채운 배열 생성
mean_values_testY = np.repeat(scaler.mean_[np.newaxis, :], testY.shape[0], axis=0)

# 첫 번째 열에 testY 대입
mean_values_testY[:, 0] = np.squeeze(testY)

# 역변환
testY_original = scaler.inverse_transform(mean_values_testY)[:,0]
print(testY_original.shape)

# 그래프 그리기
plt.figure(figsize=(14, 5))

# 원래 'increase' 가격 그래프
plt.plot(dates, original_increase, color='green', label='Original Increase Price')

testY = np.squeeze(testY)
# 실제와 예측 그래프
plt.plot(test_dates[seq_len:], testY, color='blue', label='Actual Increase Price')
plt.plot(test_dates[seq_len:], prediction, color='red', linestyle='--', label='Predicted Increase Price')
plt.xlabel('Date')
plt.ylabel('Increase Price')
plt.title('Original, Actual and Predicted Increase Price')
plt.legend()
plt.show()

# 확대해서 그래프 그리기
zoom_start = len(test_dates) - 50

zoom_end = len(test_dates)
plt.figure(figsize=(14, 5))

adjusted_start = zoom_start - seq_len

plt.plot(test_dates[zoom_start:zoom_end],
         testY[adjusted_start:zoom_end - zoom_start + adjusted_start],
         color='blue',
         label='Actual Increase Price')

plt.plot(test_dates[zoom_start:zoom_end],
         prediction[adjusted_start:zoom_end - zoom_start + adjusted_start ],
         color='red',
         linestyle='--',
         label='Predicted Increase Price')

plt.xlabel('Date')
plt.ylabel('Increase Price')
plt.title('Zoomed In Actual vs Predicted Increase Price')
plt.legend()
plt.show()

