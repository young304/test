# 라이브러리 및 툴킷 임포트
import pandas as pd  # 데이터 처리 및 분석을 위한 pandas 라이브러리
import numpy as np  # 수치 계산 및 배열 작업을 위한 numpy 라이브러리
import plotly.express as px  # 상호작용 가능한 그래프를 생성하기 위한 Plotly Express 라이브러리

import streamlit as st  # 웹 애플리케이션을 구축하기 위한 Streamlit 라이브러리
import warnings  # 경고 메시지를 처리하기 위한 라이브러리

# 상관 행렬 히트맵을 생성하는 함수
def create_heat_map(the_df):
    # 입력된 데이터프레임의 수치형 데이터에 대한 상관 계수 계산
    correlation = the_df.corr(numeric_only=True)

    # 히트맵 그래프 생성
    fig = px.imshow(
        correlation,  # 상관 계수 행렬을 이미지로 변환
        template="plotly_dark",  # 다크 테마 적용
        text_auto="0.2f",  # 각 셀에 소수점 둘째 자리까지 텍스트 표시
        aspect=1,  # 그래프의 가로세로 비율을 1:1로 설정
        color_continuous_scale="PuBu",  # 연속 색상 스케일 설정
        title="Correlation Heatmap of Data",  # 히트맵 제목 설정
        height=650,  # 그래프의 높이 설정
    )
    
    # 그래프 내 텍스트 설정 (폰트 크기 및 글꼴 스타일)
    fig.update_traces(
        textfont={
            "size": 18,  # 텍스트 크기 설정
            "family": "consolas"  # 텍스트 글꼴 설정
        },
    )
    
    # 그래프 레이아웃 업데이트 (제목과 호버 라벨 설정)
    fig.update_layout(
        title={
            "font": {
                "size": 30,  # 제목 폰트 크기 설정
                "family": "tahoma"  # 제목 폰트 글꼴 설정
            }
        },
        hoverlabel={
            "bgcolor": "#111",  # 호버 라벨의 배경색 설정
            "font_size": 16,  # 호버 라벨의 폰트 크기 설정
            "font_family": "consolas"  # 호버 라벨의 글꼴 설정
        }
    )
    
    # 생성된 히트맵 그래프 반환
    return fig

# 산포 행렬 (scatter matrix) 그래프를 생성하는 함수
def create_scatter_matrix(the_df):
    # 산포 행렬 그래프 생성
    fig = px.scatter_matrix(
        the_df,  # 입력 데이터프레임
        dimensions=the_df.select_dtypes(include="number").columns,  # 숫자형 열만 사용
        height=800,  # 그래프의 높이 설정
        color=the_df.iloc[:, -1],  # 마지막 열을 기준으로 색상 지정 (대개 타겟 변수)
        opacity=0.65,  # 투명도 설정
        title="Relationships Between Numerical Data",  # 산포 행렬 제목 설정
        template="plotly_dark"  # 다크 테마 적용
    )

    # 그래프 레이아웃 업데이트 (제목과 호버 라벨 설정)
    fig.update_layout(
        title={
            "font": {
                "size": 30,  # 제목 폰트 크기 설정
                "family": "tahoma"  # 제목 폰트 글꼴 설정
            }
        },
        hoverlabel={
            "bgcolor": "#111",  # 호버 라벨의 배경색 설정
            "font_size": 14,  # 호버 라벨의 폰트 크기 설정
            "font_family": "consolas"  # 호버 라벨의 글꼴 설정
        }
    )
    
    # 생성된 산포 행렬 그래프 반환
    return fig

# 두 변수 간의 관계를 시각화하는 산포도 (scatter plot) 그래프를 생성하는 함수
def create_relation_scatter(the_df, x, y):
    # 산포도 그래프 생성
    fig = px.scatter(
        data_frame=the_df,  # 입력 데이터프레임
        x=x,  # x축에 사용할 변수
        y=y,  # y축에 사용할 변수
        color=y,  # y 변수에 따른 색상 지정
        opacity=0.78,  # 투명도 설정
        title="Predicted Vs. Actual",  # 그래프 제목 설정
        template="plotly_dark",  # 다크 테마 적용
        trendline="ols",  # 최적의 선형 회귀선을 추가
        height=650  # 그래프의 높이 설정
    )

    # 그래프 레이아웃 업데이트 (제목 설정)
    fig.update_layout(
        title={
            "font": {
                "size": 28,  # 제목 폰트 크기 설정
                "family": "tahoma"  # 제목 폰트 글꼴 설정
            }
        }
    )
    
    # 생성된 산포도 그래프 반환
    return fig
