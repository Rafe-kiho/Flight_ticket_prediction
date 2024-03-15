#!pip install pandas seaborn
import pandas as pd
import csv


# 한글폰트 설정
import matplotlib.pyplot as plt

plt.rc("font", family="Malgun Gothic")
plt.rc("axes", unicode_minus=False)

# 폰트가 선명하게 보이도록 retina 설정
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats("retina")

# 한글폰트와 마이너스 폰트 설정 확인
pd.Series([-1, 0, 1, 3, 5]).plot(title="한글폰트")


df = pd.read_csv("crawling_results.csv")
# 데이터 확인
df.head()
# 데이터 형태 확인
df.shape
# 데이터 정보 확인
df.info()
# 결측치 보기
df.isnull().sum() # 결과 데이터 이상 무

print(df["항공사"].unique())
print(df["가격"].unique())
df['가격'] = df['가격'].str.extract('([\d,]+원)')

# '원'과 숫자 사이의 쉼표(,) 제거
df['가격'] = df['가격'].str.replace(',', '')
# df_pn = df["가격"].unique()
# print(len(df_pn))

df_nd = df[~df["가격"].str.contains('장애인|학생')]

final_df = df_nd[df_nd['가격'].str.contains('원')]
print(final_df)
final_df['가격'] = final_df['가격'].str.extract('(\d+원)')
# sorted_df = final_df.copy()
# sorted_df['가격_길이'] = sorted_df['가격'].apply(len)
# sorted_df = sorted_df.sort_values(by='가격_길이', ascending=False).drop('가격_길이', axis=1)
# print("-"*40)
# print(sorted_df.head())
#
# sorted_df = final_df.copy()
# sorted_df['가격_길이'] = sorted_df['가격'].apply(len)
# sorted_df = sorted_df.sort_values(by='가격_길이', ascending=True).drop('가격_길이', axis=1)
# print("-"*40)
# print(sorted_df.head())
# print(len(sorted_df))


# final_df = df_nd[df_nd['가격'].str.contains('원')]
# print(final_df)
#
#
#
# Correcting the regex to ensure only the numeric value followed by "원" is kept, and other details are removed
# final_df['가격'] = final_df['가격'].str.extract(r'(\d+,\d+,\d+)원')

#
# # Adding back the "원" suffix to maintain the "N원" format
# final_df['가격'] = final_df['가격'] + '원'
#
# sorted_df = final_df.copy()
# sorted_df['가격_길이'] = sorted_df['가격'].apply(len)
# sorted_df = sorted_df.sort_values(by='가격_길이', ascending=False).drop('가격_길이', axis=1)
# print("-"*40)
# print(sorted_df.head())
#
# print(final_df['가격'].unique())


#-------------------------------------------
# df['가격'] = df['가격'].str.extract('([\d,]+원)')
#
# # '원'과 숫자 사이의 쉼표(,) 제거
# df['가격'] = df['가격'].str.replace(',', '')
#
# df_nd = df[~df["가격"].str.contains('장애인|학생')]
#
# df_a = df_nd[df_nd['가격'].str.contains('원')]
# df_a.head()