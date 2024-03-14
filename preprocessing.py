from sklearn.preprocessing import MinMaxScaler
import pandas as pd

file_path = 'ICN_to_KIX4.csv'  # 업데이트된 파일 경로

# 파일 로드
df = pd.read_csv(file_path)

# 데이터의 처음 몇 줄 확인
print(df.head())

# 데이터 전처리 단계

# '가격' 컬럼 처리
df['가격'] = df['가격'].str.replace('원', '').str.replace(',', '').astype(float)

# 날짜와 시간 처리
df['날짜'] = pd.to_datetime(df['날짜'].str[:10])
df['출발시간_시'] = df['출발시간'].str.split(':').str[0].astype(int)
df['출발시간_분'] = df['출발시간'].str.split(':').str[1].astype(int)
df['도착시간_시'] = df['도착시간'].str.split(':').str[0].astype(int)
df['도착시간_분'] = df['도착시간'].str.split(':').str[1].astype(int)

# 필요없는 컬럼 제거
df.drop(['출발시간', '도착시간', '소요시간'], axis=1, inplace=True)

# 원-핫 인코딩을 위한 처리
df_encoded = pd.get_dummies(df, columns=['항공사', '출발지', '도착지', '최저가 결제수단'])

# 수치 데이터 스케일링
scaler = MinMaxScaler()
numeric_features = ['출발시간_시', '출발시간_분', '도착시간_시', '도착시간_분', '가격']
df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])

# 전처리된 데이터 확인
df_encoded.head()


# 수치 데이터 스케일링
scaler = MinMaxScaler()
numeric_features = ['출발시간_시', '출발시간_분', '도착시간_시', '도착시간_분', '가격']
df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])

# 전처리된 데이터 확인
df_encoded.head()