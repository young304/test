# 라이브러리 및 모듈 임포트
import re  # 정규 표현식 사용을 위한 라이브러리
import relations  # 관계 분석을 위한 사용자 정의 모듈 (상관관계, 산포도 등)
import prediction  # 예측 및 평가를 위한 사용자 정의 모듈

from time import sleep  # 지연 시간을 추가하기 위한 sleep 함수
import pandas as pd  # 데이터 처리 및 분석을 위한 pandas 라이브러리
import numpy as np  # 수치 계산 및 배열 작업을 위한 numpy 라이브러리
import plotly.express as px  # 상호작용 가능한 그래프를 그리기 위한 plotly 라이브러리

from sklearn.metrics import mean_absolute_error  # 모델의 평가 지표 중 하나인 평균 절대 오차(MAE)를 구하기 위한 라이브러리

import streamlit as st  # Streamlit을 사용하여 웹 애플리케이션을 구축하기 위한 라이브러리
from streamlit.components.v1 import html  # HTML을 사용한 웹 페이지 구성 요소 생성
from streamlit_option_menu import option_menu  # 옵션 메뉴 생성을 위한 라이브러리
import warnings  # 경고 메시지 무시를 위한 라이브러리

def run():  # 애플리케이션 실행 함수 정의
    # 애플리케이션 설정 (페이지 제목, 아이콘, 레이아웃)
    st.set_page_config(
        page_title="Yearly Spent Prediction",  # 페이지 제목
        page_icon="💰",  # 페이지 아이콘
        layout="wide"  # 레이아웃 설정
    )

    # FutureWarning 경고를 무시하기 위한 설정
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # 데이터셋을 불러오는 함수 정의 (캐싱하여 성능 최적화)
    @st.cache_data
    def load_data(the_file_path):
        # CSV 파일 읽기
        df = pd.read_csv(the_file_path)
        # 열 이름의 공백 및 특수 문자를 제거하고 변경
        df.columns = df.columns.str.replace(" ",  "_").str.replace(".", "")
        # 열 이름을 더 직관적으로 변경
        df.rename(columns={
            "Time_on_App": "App_Usage",  # 앱 사용 시간을 App_Usage로 변경
            "Time_on_Website": "Website_Usage",  # 웹사이트 사용 시간을 Website_Usage로 변경
            "Length_of_Membership": "Membership_Length",  # 멤버십 기간을 Membership_Length로 변경
            "Yearly_Amount_Spent": "Yearly_Spent"  # 연간 소비 금액을 Yearly_Spent로 변경
        }, inplace=True)
        return df  # 정리된 데이터프레임 반환

    # 선형 회귀 모델을 불러오는 함수 정의 (캐싱하여 성능 최적화)
    @st.cache_data
    def load_linear_regression_model(model_path):
        return pd.read_pickle(model_path)  # 저장된 선형 회귀 모델 파일을 불러옴

    # 데이터 로드 및 모델 로드
    df = load_data("Ecommerce_Customers.csv")  # 예시 데이터 로드
    model = load_linear_regression_model(
        "linear_regression_yearly_spent_predictor_v1.pkl")  # 저장된 선형 회귀 모델 로드

    # 입력 데이터의 유효성을 검증하는 함수 정의
    @st.cache_data
    def is_valid_data(d):
        letters = list("qwertyuiopasdfghjklzxcvbnm@!#$%^&*-+~")  # 입력 데이터에 포함되지 말아야 할 문자들
        # 입력 데이터의 길이가 2 이상이며, 문자들이 포함되어 있지 않으면 유효한 데이터로 간주
        return len(d) >= 2 and not any([i in letters for i in list(d)])

    # 테스트 파일의 유효성을 검증하는 함수 정의
    @st.cache_data
    def validate_test_file(test_file_columns):
        # 테스트 파일의 열 이름을 소문자로 변환하고, 줄바꿈 처리
        col = "\n".join(test_file_columns).lower()
        # 열 이름에 세션, 앱, 웹, 멤버십과 관련된 패턴이 있는지 확인하는 정규 표현식 패턴
        pattern = re.compile(
            "\w*\W*session\W*\w*\W*app\W*\w*\W*web\W*\w*\W*membership\W*\w*")
        # 정규 표현식을 이용해 해당 패턴을 찾고 매칭된 패턴의 개수를 확인하여 유효성 검증
        matches = pattern.findall(col)
        return len("\n".join(matches).split("\n")) == 4

    # 애플리케이션의 스타일을 정의하는 CSS
    st.markdown(
        """
    <style>
         .main {
            text-align: center; 
         }
         .st-emotion-cache-16txtl3 h1 {
         font: bold 29px arial;
         text-align: center;
         margin-bottom: 15px
            
         }
         div[data-testid=stSidebarContent] {
         background-color: #111;
         border-right: 4px solid #222;
         padding: 8px!important
         
         }

         div.block-containers{
            padding-top: 0.5rem
         }

         .st-emotion-cache-z5fcl4{
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 1.1rem;
            padding-right: 2.2rem;
            overflow-x: hidden;
         }

         .st-emotion-cache-16txtl3{
            padding: 2.7rem 0.6rem;
         }

         .plot-container.plotly{
            border: 1px solid #333;
            border-radius: 6px;
         }

         div.st-emotion-cache-1r6slb0 span.st-emotion-cache-10trblm{
            font: bold 24px tahoma;
         }
         div [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }

        div[data-baseweb=select]>div{
            cursor: pointer;
            background-color: #111;
            border: 2px solid #0079FF;
        }

        div[data-baseweb=base-input]{
            background-color: #111;
            border: 4px solid #444;
            border-radius: 5px;
            padding: 5px;
        }

        div[data-testid=stFormSubmitButton]> button{
            width: 40%;
            background-color: #111;
            border: 2px solid #0079FF;
            padding: 18px;
            border-radius: 30px;
            opacity: 0.76;
        }
        div[data-testid=stFormSubmitButton]  p{
            font-weight: bold;
            font-size : 20px
        }

        div[data-testid=stFormSubmitButton]> button:hover{
            opacity: 1;
            border: 2px solid #0079FF;
            color: #fff
        }


    </style>
    """,
        unsafe_allow_html=True  # HTML 스타일을 안전하게 렌더링
    )

    # 사이드바 스타일 설정
    side_bar_options_style = {
        "container": {"padding": "0!important", "background-color": 'transparent'},  # 메뉴 컨테이너 스타일
        "icon": {"color": "white", "font-size": "18px"},  # 아이콘 스타일
        "nav-link": {"color": "white", "font-size": "16px", "text-align": "left", "margin": "0px", "margin-bottom": "15px"},  # 링크 스타일
        "nav-link-selected": {"background-color": "#0079FF", "font-size": "15px"},  # 선택된 링크 스타일
    }

    # 서브 옵션 메뉴 스타일 설정
    sub_options_style = {
        "container": {"padding": "3!important", "background-color": '#101010', "border": "2px solid #0079FF"},
        "nav-link": {"color": "white", "padding": "12px", "font-size": "18px", "text-align": "center", "margin": "0px", },
        "nav-link-selected": {"background-color": "#0079FF"},
    }

    # 컨테이너 생성
    header = st.container()
    content = st.container()

    # 사이드바 구성
    with st.sidebar:
        st.title("SPENT MONEY :blue[PREDICTION]")  # 사이드바 제목 설정
        st.image("imgs/money.png", caption="", width=60)  # 아이콘 이미지 삽입
        # 사이드바 옵션 메뉴 설정
        page = option_menu(
            menu_title=None,
            options=['Home', 'Relations & Correlarions', 'Prediction'],  # 메뉴 옵션 설정
            icons=['diagram-3-fill', 'bar-chart-line-fill', "cpu"],  # 아이콘 설정
            menu_icon="cast",  # 메뉴 아이콘
            default_index=0,  # 기본 선택된 메뉴
            styles=side_bar_options_style  # 스타일 설정
        )
        st.write("***")  # 구분선

        # CSV 파일 업로드 기능 추가
        data_file = st.file_uploader("Upload Your Dataset (CSV)📂", type="csv")

        if data_file is not None:  # 업로드된 파일이 있을 경우
            if data_file.name.split(".")[-1].lower() != "csv":  # 파일 확장자가 CSV가 아닌 경우
                st.error("Please, Upload CSV FILE ONLY")  # 오류 메시지 출력
            else:
                df = pd.read_csv(data_file)  # CSV 파일 읽기

        # Home 페이지 구성
        if page == "Home":

            with header:  # 헤더 구성
                st.header('Customer Expected Annual Spend 📈💰')  # 헤더 제목 설정

            with content:  # 내용 구성
                # 데이터프레임을 표 형식으로 표시
                st.dataframe(df.sample(frac=0.25, random_state=35).reset_index(drop=True),
                             use_container_width=True)

                st.write("***")  # 구분선

                st.subheader("Data Summary Overview 🧐")  # 데이터 요약 섹션

                # 숫자형 데이터와 문자열 데이터의 수를 계산
                len_numerical_data = df.select_dtypes(
                    include="number").shape[1]
                len_string_data = df.select_dtypes(include="object").shape[1]

                # 숫자형 데이터 통계 요약 표시
                if len_numerical_data > 0:
                    st.subheader("Numerical Data [123]")

                    data_stats = df.describe().T  # 숫자형 데이터 통계 정보
                    st.table(data_stats)  # 표로 표시

                # 문자열 데이터 통계 요약 표시
                if len_string_data > 0:
                    st.subheader("String Data [𝓗]")

                    data_stats = df.select_dtypes(
                        include="object").describe().T  # 문자열 데이터 통계 정보
                    st.table(data_stats)  # 표로 표시

        # Relations & Correlations 페이지 구성
        if page == "Relations & Correlarions":

            with header:  # 헤더 구성
                st.header("Correlations Between Data 📉🚀")  # 상관관계 섹션

            with content:  # 내용 구성
                # 상관 관계 히트맵을 생성하여 표시
                st.plotly_chart(relations.create_heat_map(df),
                                use_container_width=True)

                # 산포 행렬을 생성하여 표시
                st.plotly_chart(relations.create_scatter_matrix(
                    df), use_container_width=True)

                st.write("***")  # 구분선

                # 상관관계 분석을 위한 변수 선택
                col1, col2 = st.columns(2)
                with col1:
                    # 첫 번째 변수 선택
                    first_feature = st.selectbox(
                        "First Feature", options=(df.select_dtypes(
                            include="number").columns.tolist()), index=0).strip()

                # 두 번째 변수 선택을 위한 변수 목록에서 첫 번째 변수를 제거
                temp_columns = df.select_dtypes(
                    include="number").columns.to_list().copy()

                temp_columns.remove(first_feature)

                with col2:
                    # 두 번째 변수 선택
                    second_feature = st.selectbox(
                        "Second Feature", options=(temp_columns), index=0).strip()

                # 선택한 두 변수의 산포도 그래프를 생성하여 표시
                st.plotly_chart(relations.create_relation_scatter(
                    df, first_feature, second_feature), use_container_width=True)

        # Prediction 페이지 구성
        if page == "Prediction":
            with header:  # 헤더 구성
                st.header("Prediction Model 💰🛍️")  # 예측 모델 섹션
                # 예측 옵션 메뉴 설정 (단일 값 예측 또는 파일을 통한 예측)
                prediction_option = option_menu(menu_title=None, options=["One Value", 'From File'],
                                                icons=[" "]*4, menu_icon="cast", default_index=0,
                                                orientation="horizontal", styles=sub_options_style)

            with content:  # 내용 구성
                # 단일 값 예측 옵션 선택 시
                if prediction_option == "One Value":
                    with st.form("Predict_value"):  # 입력 폼 구성
                        c1, c2 = st.columns(2)
                        with c1:
                            # 평균 세션 길이 입력
                            session_length = st.number_input(label="Average Session Length (Minute)",
                                                             min_value=5.0, max_value=50.0, value=30.0,
                                                             )
                            # 웹사이트 사용 시간 입력
                            web_usage_length = st.number_input(label="Time of Website (Minute)",
                                                               min_value=10.0,  value=30.0)
                        with c2:
                            # 앱 사용 시간 입력
                            app_usage_length = st.number_input(label="Time of APP (Minute)",
                                                               min_value=8.0, max_value=30.0, value=15.0)
                            # 멤버십 기간 입력
                            membership_length = st.number_input(label="Membership Length (Months)",
                                                                min_value=1.0, value=2.0)

                        st.write("")  # 빈 공간 추가

                        # 예측 버튼 추가
                        predict_button = st.form_submit_button(
                            label='Predict', use_container_width=True)

                        st.write("***")  # 구분선 추가

                        # 예측 버튼이 눌렸을 때
                        if predict_button:

                            # 예측 진행 중 스피너 표시
                            with st.spinner(text='Predict The Value..'):
                                # 사용자가 입력한 데이터를 리스트로 구성
                                new_data = [
                                    session_length, app_usage_length, web_usage_length, membership_length]
                                # 모델을 사용해 예측 수행
                                predicted_value = model.predict([new_data])
                                sleep(1.2)  # 지연 시간 추가

                                # 예측 결과와 모델 정확도를 표시하는 두 개의 열 구성
                                predicted_col, score_col = st.columns(2)

                                with predicted_col:
                                    st.image("imgs/money-bag.png",
                                             caption="", width=70)

                                    st.subheader(
                                        "Expected To Spent")  # 예측된 소비 금액
                                    st.subheader(
                                        f"${np.round(predicted_value[0], 2)}")  # 예측 결과 출력

                                with score_col:
                                    st.image("imgs/star.png",
                                             caption="", width=70)
                                    st.subheader("Model Accuracy")  # 모델 정확도
                                    st.subheader(f"{np.round(98.27, 2)}%")  # 정확도 출력

                # 파일로부터 예측 옵션 선택 시
                if prediction_option == "From File":
                    st.info("Please upload your file with the following columns' names in the same order\n\
                            [Avg_Session_Length, App_Usage, Website_Usage, Membership_Length]", icon="ℹ️")

                    # 테스트 파일 업로드 기능 추가
                    test_file = st.file_uploader(
                        "Upload Your Test File 📂", type="csv")

                    if test_file is not None:
                        extention = test_file.name.split(".")[-1]
                        if extention.lower() != "csv":  # 업로드된 파일이 CSV가 아닌 경우 오류 출력
                            st.error("Please, Upload CSV FILE ONLY")

                        else:
                            # 테스트 파일 읽기 및 열 이름 정리
                            X_test = pd.read_csv(test_file)
                            X_test.columns = X_test.columns.str.replace(
                                " ",  "_").str.replace(".", "")

                            # 결측값 제거
                            X_test.dropna(inplace=True)

                            # 테스트 파일의 유효성을 검증하고 예측 수행
                            if validate_test_file(X_test.columns.to_list()):
                                all_predicted_values = model.predict(
                                    X_test)
                                # 예측 결과를 포함한 데이터프레임 생성
                                final_complete_file = pd.concat([X_test, pd.DataFrame(all_predicted_values,
                                                                                      columns=["Predicted_Yearly_Spent"])], axis=1)
                                st.write("")
                                # 예측 결과 표시
                                st.dataframe(final_complete_file,
                                             use_container_width=True)
                            else:
                                st.warning(
                                    "Please, Check That Your Test File Has The Mention Columns in The Same Order", icon="⚠️")

                    # 비교 기능을 위한 폼 생성
                    with st.form("comaprison_form"):

                        # 실제 값과 비교 버튼 클릭 시
                        if st.form_submit_button("Compare Predicted With Actual Values"):
                            st.info(
                                "Be Sure Your Actual Values File HAS ONLY ONE COLUMN (Yearly_Spent)", icon="ℹ️")

                            # 실제 값 파일 업로드 기능 추가
                            actual_file = st.file_uploader(
                                "Upload Your Actual Data File 📂", type="csv")

                            if actual_file is not None and test_file is not None:
                                y_test = pd.read_csv(actual_file)  # 실제 값 파일 읽기
                                if y_test.shape[1] == 1:  # 실제 값 파일이 한 열만 가지고 있는지 확인

                                    col1, col2 = st.columns(2)

                                    with col1:
                                        # 예측 정확도 계산 및 표시
                                        test_score = np.round(
                                            model.score(X_test, y_test) * 100, 2)
                                        prediction.creat_matrix_score_cards("imgs/star.png",
                                                                            "Prediction Accuracy",
                                                                            test_score,
                                                                            True
                                                                            )

                                    with col2:
                                        # 평균 절대 오차 계산 및 표시
                                        mae = mean_absolute_error(
                                            y_test, all_predicted_values)
                                        prediction.creat_matrix_score_cards("imgs/sort.png",
                                                                            "Error Ratio",
                                                                            np.round(
                                                                                mae, 2),
                                                                            False)

                                    # 예측 값과 실제 값을 비교하는 데이터프레임 생성 및 표시
                                    predicted_df = prediction.create_comparison_df(
                                        y_test, all_predicted_values)
                                    st.dataframe(
                                        predicted_df, use_container_width=True, height=300)

                                    # 잔차 그래프 생성 및 표시
                                    st.plotly_chart(prediction.create_residules_scatter(predicted_df),
                                                    use_container_width=True)

                                else:
                                    st.warning(
                                        "Please, Check That Your Test File Has The One Column.", icon="⚠️")

                            else:
                                st.warning(
                                    "Please, Check That You Upload The Test File & Actual Value", icon="⚠️")


run()  # 애플리케이션 실행
