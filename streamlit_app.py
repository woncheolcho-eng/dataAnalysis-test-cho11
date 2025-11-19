#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

#######################
# Page configuration
st.set_page_config(
    page_title="US Population Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("default")

#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetric"] {
    background-color: #f5f5f5;   /* 밝은 회색 */
    color: #000000;              /* 글자가 보이도록 검정 */
    text-align: center;
    padding: 15px 0;
    border-radius: 10px;
    border: 1px solid #e0e0e0;
}

</style>
""", unsafe_allow_html=True)

#######################
# Load data
df_reshaped = pd.read_csv('ugv_mission_dataset_220rows.csv') ## 분석 데이터 넣기


#######################
# Sidebar
with st.sidebar:

    st.title("UGV Mission Dashboard")
    st.markdown("### 🔧 데이터 필터링")

    # TerrainType 선택
    terrain_filter = st.multiselect(
        "Terrain Type 선택",
        options=sorted(df_reshaped["TerrainType"].unique()),
        default=sorted(df_reshaped["TerrainType"].unique())
    )

    # 배터리 범위
    battery_range = st.slider(
        "Battery Level 범위",
        int(df_reshaped["BatteryLevel"].min()),
        int(df_reshaped["BatteryLevel"].max()),
        (int(df_reshaped["BatteryLevel"].min()),
         int(df_reshaped["BatteryLevel"].max()))
    )

    # 장애물 밀도 선택
    obstacle_filter = st.multiselect(
        "Obstacle Density 선택",
        options=sorted(df_reshaped["ObstacleDensity"].unique()),
        default=sorted(df_reshaped["ObstacleDensity"].unique())
    )

    # Sensor Health 범위
    sensor_range = st.slider(
        "Sensor Health 범위",
        int(df_reshaped["SensorHealth"].min()),
        int(df_reshaped["SensorHealth"].max()),
        (int(df_reshaped["SensorHealth"].min()),
         int(df_reshaped["SensorHealth"].max()))
    )

    # 통신 품질 범위
    comm_range = st.slider(
        "Comm Quality 범위",
        float(df_reshaped["CommQuality"].min()),
        float(df_reshaped["CommQuality"].max()),
        (float(df_reshaped["CommQuality"].min()),
         float(df_reshaped["CommQuality"].max()))
    )

    st.markdown("---")

    # 머신러닝 모델 선택
    st.markdown("### 🤖 머신러닝 모델 선택")
    ml_model_choice = st.selectbox(
        "사용할 모델을 선택하세요",
        ("분류 모델: 미션 성공 예측",
         "회귀 모델: 미션 시간 예측",
         "군집 모델: 임무 패턴 분석(K-Means)")
    )

    st.markdown("---")

    # 컬러 테마 선택
    st.markdown("### 🎨 시각화 컬러 테마")
    viz_theme = st.selectbox(
        "컬러 테마 선택",
        ("Blues", "Viridis", "Plasma", "Inferno", "Cividis")
    )

#######################
# Plots



#######################
# Dashboard Main Panel
col = st.columns((1.5, 4.5, 2), gap='medium')

with col[0]:
    st.markdown("## 📊 UGV 요약 지표")

    # 필터링된 데이터 생성
    df_filtered = df_reshaped[
        (df_reshaped["TerrainType"].isin(terrain_filter)) &
        (df_reshaped["BatteryLevel"].between(battery_range[0], battery_range[1])) &
        (df_reshaped["ObstacleDensity"].isin(obstacle_filter)) &
        (df_reshaped["SensorHealth"].between(sensor_range[0], sensor_range[1])) &
        (df_reshaped["CommQuality"].between(comm_range[0], comm_range[1]))
    ]

    # 데이터가 없을 경우 메시지 출력
    if df_filtered.empty:
        st.warning("⚠ 현재 필터 조건에 해당되는 데이터가 없습니다.")
    else:
        # Metric 카드 4개
        avg_battery = df_filtered["BatteryLevel"].mean()
        avg_speed = df_filtered["Speed"].mean()
        avg_sensor = df_filtered["SensorHealth"].mean()
        success_rate = df_filtered["MissionSuccess"].mean() * 100

        st.metric("평균 배터리(%)", f"{avg_battery:.1f}")
        st.metric("평균 속도", f"{avg_speed:.2f} m/s")
        st.metric("평균 센서 상태", f"{avg_sensor:.1f}")
        st.metric("성공률", f"{success_rate:.1f}%")

    st.markdown("---")
    st.markdown("## 🤖 선택된 머신러닝 모델")

    # 선택한 모델 안내
    if ml_model_choice == "분류 모델: 미션 성공 예측":
        st.info("🔍 **분류 모델(예: Logistic Regression / RandomForestClassifier)** 을 사용해 미션 성공 확률을 예측합니다.")
    elif ml_model_choice == "회귀 모델: 미션 시간 예측":
        st.info("⏱ **회귀 모델(예: Linear Regression / RandomForestRegressor)** 을 사용해 예상 미션 시간을 예측합니다.")
    else:
        st.info("🧭 **군집 모델(K-Means)** 을 사용해 UGV 임무 패턴을 분석하고 그룹화합니다.")

with col[1]:

    st.markdown("## 🔍 데이터 시각화")

    if df_filtered.empty:
        st.warning("⚠ 시각화를 표시할 데이터가 없습니다.")
    else:

        # -----------------------------
        # 1) 상관관계 히트맵
        # -----------------------------
        st.markdown("### 📌 변수 상관관계 히트맵")

        # corr 계산 시 numeric_only 제거 (버전 충돌 방지)
        corr_matrix = df_filtered.corr()

        corr_df = corr_matrix.reset_index().melt('index', var_name='variable', value_name='value')

        # 테마를 소문자로 변환 (Altair 호환)
        theme_safe = viz_theme.lower() if isinstance(viz_theme, str) else "blues"

        corr_chart = (
            alt.Chart(corr_df)
            .mark_rect()
            .encode(
                x=alt.X("index:O", title=""),
                y=alt.Y("variable:O", title=""),
                color=alt.Color("value:Q", scale=alt.Scale(scheme=theme_safe)),
                tooltip=["index", "variable", "value"]
            )
            .properties(height=350)
        )

        st.altair_chart(corr_chart, use_container_width=True)

        st.markdown("---")

        # -----------------------------
        # 2) 군집 분석 선택 시
        # -----------------------------
        if ml_model_choice == "군집 모델: 임무 패턴 분석(K-Means)":

            st.markdown("### 🧭 K-Means 군집 시각화")

            # 클러스터 개수는 데이터 크기에 따라 자동 결정
            n_samples = len(df_filtered)
            n_clusters = min(3, max(1, n_samples // 10))  # 최소 1, 최대 3

            from sklearn.cluster import KMeans
            X = df_filtered[["Speed", "BatteryLevel", "ObstacleDensity"]]

            # n_init="auto" → 버전 충돌 방지 위해 n_init=10
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df_filtered["Cluster"] = kmeans.fit_predict(X)

            cluster_chart = (
                alt.Chart(df_filtered)
                .mark_circle(size=80)
                .encode(
                    x="Speed:Q",
                    y="BatteryLevel:Q",
                    color=alt.Color("Cluster:N"),
                    tooltip=["Speed", "BatteryLevel", "ObstacleDensity", "Cluster"]
                )
                .properties(height=350)
            )

            st.altair_chart(cluster_chart, use_container_width=True)

        # -----------------------------
        # 3) 일반 시각화(분류/회귀)
        # -----------------------------
        else:
            st.markdown("### 🚗 속도 vs 배터리 분포")

            scatter_chart = (
                alt.Chart(df_filtered)
                .mark_circle(size=80, opacity=0.8)
                .encode(
                    x="Speed:Q",
                    y="BatteryLevel:Q",
                    color=alt.Color("MissionSuccess:N", scale=alt.Scale(scheme=theme_safe)),
                    tooltip=["Speed", "BatteryLevel", "ObstacleDensity", "MissionSuccess"]
                )
                .properties(height=350)
            )

            st.altair_chart(scatter_chart, use_container_width=True)


#with col[2]:
with col[2]:

    st.markdown("## 🏅 Top Performance 랭킹")

    if df_filtered.empty:
        st.warning("⚠ 랭킹 정보를 표시할 데이터가 없습니다.")
    else:
        # -----------------------------------
        # 1) 속도 상위 TOP 5
        # -----------------------------------
        st.markdown("### 🚀 속도 상위 TOP 5")

        top_speed = df_filtered.nlargest(5, "Speed")[["Speed", "BatteryLevel", "MissionTime"]]
        st.dataframe(top_speed, use_container_width=True)

        # -----------------------------------
        # 2) 미션 시간 짧은 TOP 5
        # -----------------------------------
        st.markdown("### ⏱ 미션 시간 짧은 TOP 5")

        top_fast = df_filtered.nsmallest(5, "MissionTime")[["MissionTime", "Speed", "BatteryLevel"]]
        st.dataframe(top_fast, use_container_width=True)

    st.markdown("---")

    # -----------------------------------
    # 3) TerrainType 성공률 비교
    # -----------------------------------
    st.markdown("## 🌍 Terrain Type별 성공률")

    terrain_success = (
        df_filtered.groupby("TerrainType")["MissionSuccess"]
        .mean()
        .reset_index()
    )
    terrain_success["MissionSuccess"] *= 100  # 퍼센트 변환

    terrain_chart = (
        alt.Chart(terrain_success)
        .mark_bar()
        .encode(
            x=alt.X("TerrainType:O", title="Terrain Type"),
            y=alt.Y("MissionSuccess:Q", title="Success Rate (%)"),
            color=alt.Color("TerrainType:O", scale=alt.Scale(scheme=viz_theme.lower())),
            tooltip=["TerrainType", "MissionSuccess"]
        )
        .properties(height=250)
    )

    st.altair_chart(terrain_chart, use_container_width=True)

    st.markdown("---")

    # -----------------------------
    # 4) About 섹션
    # -----------------------------
    st.markdown("## ℹ️ About")
    st.write("""
    이 대시보드는 **UGV(무인 지상 차량) 임무 데이터**를 기반으로 구성되었으며,  
    TerrainType, 센서 상태, 통신 품질, 배터리 상태 등 여러 변수와  
    미션 성공률 및 미션 시간 간의 관계를 분석합니다.

    또한 다음과 같은 머신러닝 기법을 포함합니다:

    - **분류(Classification)**: 미션 성공 여부 예측  
    - **회귀(Regression)**: 미션 수행 시간 예측  
    - **군집(Clustering)**: 임무 패턴 분석(K-Means)  

    """)

