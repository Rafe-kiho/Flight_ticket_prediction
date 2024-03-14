import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt

# 데이터 로드
file_path = 'ICN_to_KIX4.csv'  # 데이터 파일 경로
data = pd.read_csv(file_path)

# 데이터 전처리
data['날짜'] = pd.to_datetime(data['날짜'], format='%Y.%m.%d %a')
data['연'] = data['날짜'].dt.year
data['월'] = data['날짜'].dt.month
data['일'] = data['날짜'].dt.day
data['요일'] = data['날짜'].dt.weekday
data['출발시간'] = pd.to_datetime(data['출발시간'], format='%H:%M').dt.hour
data['가격'] = data['가격'].str.replace('원', '').str.replace(',', '').astype(int)
data = data.drop(['날짜', '출발지', '도착시간', '도착지', '소요시간', '최저가 결제수단'], axis=1)
encoder = OneHotEncoder()
airline_encoded = encoder.fit_transform(data[['항공사']]).toarray()
airline_encoded_df = pd.DataFrame(airline_encoded, columns=encoder.get_feature_names_out(['항공사']))
processed_data = pd.concat([data.drop(['항공사'], axis=1), airline_encoded_df], axis=1)

# 데이터 분리
X = processed_data.drop(['가격'], axis=1)
y = processed_data['가격']
test_size = int(len(X) * 0.2)
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

# 데이터 스케일링
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

# 시퀀스 데이터 생성
sequence_length = 7
train_generator = TimeseriesGenerator(X_train_scaled, y_train_scaled, length=sequence_length, batch_size=1)

# LSTM 모델 설계
model = Sequential([
    LSTM(50, activation='relu', input_shape=(sequence_length, X_train_scaled.shape[1])),
    Dense(1)
])

# 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='mse')
history = model.fit(train_generator, epochs=20, verbose=1)

# 학습 과정 시각화
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)


plot_loss(history)

# 예측 데이터 뽑아내기
# 1. 특정 날짜에 대한 입력 데이터 준비
# 이 부분은 실제 모델에 맞게 조정되어야 합니다. 예시를 위해 간단한 데이터를 사용합니다. 2024년 4월 17일 오후 3시 ICN to KIX 항공편 최저가
example_features = np.array([[15, 2024, 4, 17, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# 특성 데이터 스케일링
example_features_scaled = scaler.transform(example_features)

# LSTM 모델에 필요한 시퀀스 형태로 데이터 변환
sequence_length = 7  # 모델 학습 시 사용된 시퀀스 길이
example_sequence = np.array([example_features_scaled] * sequence_length).reshape(1, sequence_length, -1)

# 예측 수행
predicted_price_scaled = model.predict(example_sequence)

# 예측 결과 역스케일링
predicted_price = y_scaler.inverse_transform(predicted_price_scaled)
print(f"예측된 최저가: {predicted_price[0][0]}원")

# 학습과정 그래프 출력 유지하기
plt.show()

