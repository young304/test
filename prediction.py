# Importing ToolKits
import pandas as pd
import numpy as np
import plotly.express as px

import streamlit as st
import warnings

def creat_matrix_score_cards(card_image="", card_title="Card Title", card_value=None, percent=False):
    st.image(card_image,
             caption="", width=70)

    st.subheader(
        card_title)

    if percent:
        st.subheader(
            f"{card_value}%")

    else:
        st.subheader(
            f"{card_value}")


def create_comparison_df(y_actual, y_pred):
    predected_df = pd.DataFrame()
    predected_df["Actual Spent Values"] = y_actual
    predected_df.reset_index(
        drop=True, inplace=True)
    predected_df["Predicted Spent Value"] = y_pred

    return predected_df


def create_residules_scatter(predected_df):
    fig = px.scatter(
        predected_df,
        x=predected_df.iloc[:, 0],
        y=predected_df.iloc[:, 1],
        color=predected_df.iloc[:, 1] - predected_df.iloc[:, 0],
        opacity=0.8,
        title="Predicted Vs. Actual",
        template="plotly_dark",
        trendline="ols",
        height=650,
        labels={"x": "Actual Value", "y": "Predicted Value"}
    )

    fig.update_layout(
        title={
            "font": {
                "size": 28,
                "family": "tahoma"
            }
        }
    )
    return fig
# 라이브러리 및 툴킷 임포트
import pandas as pd  # 데이터 처리 및 분석을 위한 pandas 라이브러리
import numpy as np  # 수치 계산 및 배열 작업을 위한 numpy 라이브러리
import plotly.express as px  # 상호작용 가능한 그래프를 생성하기 위한 Plotly Express 라이브러리

import streamlit as st  # 웹 애플리케이션을 구축하기 위한 Streamlit 라이브러리
import warnings  # 경고 메시지를 처리하기 위한 라이브러리

# 점수 카드 UI를 생성하는 함수
def creat_matrix_score_cards(card_image="", card_title="Card Title", card_value=None, percent=False):
    # 점수 카드 이미지 삽입
    st.image(card_image, caption="", width=70)
    
    # 점수 카드 제목 표시
    st.subheader(card_title)
    
    # 퍼센트 여부에 따라 카드 값 표시
    if percent:
        # 퍼센트로 표시할 경우 값 뒤에 '%' 추가
        st.subheader(f"{card_value}%")
    else:
        # 일반 값일 경우 값만 표시
        st.subheader(f"{card_value}")

# 실제 값과 예측 값을 비교하는 데이터프레임을 생성하는 함수
def create_comparison_df(y_actual, y_pred):
    # 빈 데이터프레임 생성
    predected_df = pd.DataFrame()
    
    # 실제 연간 소비 금액을 데이터프레임에 추가
    predected_df["Actual Spent Values"] = y_actual
    
    # 데이터프레임의 인덱스 재설정
    predected_df.reset_index(drop=True, inplace=True)
    
    # 예측된 연간 소비 금액을 데이터프레임에 추가
    predected_df["Predicted Spent Value"] = y_pred

    # 비교용 데이터프레임 반환
    return predected_df

# 예측 값과 실제 값의 차이를 시각화하는 잔차 산포도(Residual Scatter Plot)를 생성하는 함수
def create_residules_scatter(predected_df):
    # 산포도 그래프 생성
    fig = px.scatter(
        predected_df,  # 비교 데이터프레임 사용
        x=predected_df.iloc[:, 0],  # 실제 값 (첫 번째 열)
        y=predected_df.iloc[:, 1],  # 예측 값 (두 번째 열)
        color=predected_df.iloc[:, 1] - predected_df.iloc[:, 0],  # 실제 값과 예측 값의 차이를 색상으로 표시
        opacity=0.8,  # 투명도 설정
        title="Predicted Vs. Actual",  # 그래프 제목 설정
        template="plotly_dark",  # 다크 테마 적용
        trendline="ols",  # 최적의 선형 회귀선을 추가
        height=650,  # 그래프의 높이 설정
        labels={"x": "Actual Value", "y": "Predicted Value"}  # x축, y축 라벨 설정
    )

    # 그래프 레이아웃 업데이트 (제목 폰트 설정)
    fig.update_layout(
        title={
            "font": {
                "size": 28,  # 제목 폰트 크기 설정
                "family": "tahoma"  # 제목 폰트 글꼴 설정
            }
        }
    )
    
    # 생성된 잔차 산포도 그래프 반환
    return fig
