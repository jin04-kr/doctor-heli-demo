# ============================================================
# 경기도 닥터헬기 이송수단 의사결정지원시스템
# 사용자 중심 UI / 발표·시연용
#
# 핵심 연구모형은 기존과 동일
# ------------------------------------------------------------
# 1. 사고 위치 지도 선택
# 2. Naver Directions API 실시간 도로시간
# 3. 현행 인계점 181개 비교
# 4. Monte Carlo 10,000회
# 5. 기대시간 최소 인계점 선택
# 6. 구급차 vs 닥터헬기 추천
#
# 지도
# - 사고 → 병원: 실제 Naver 도로경로
# - 사고 → 최적 인계점: 실제 Naver 도로경로
# - 병원 ↔ 인계점: 헬기 직선 비행
# ============================================================


import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium


# ============================================================
# 1. 페이지 설정
# ============================================================

st.set_page_config(
    page_title="닥터헬기 이송 의사결정",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. UI CSS
# ============================================================

st.markdown(
    """
    <style>

    /* 전체 화면 여백 축소 */
    .block-container {
        max-width: 1480px;
        padding-top: 1.1rem;
        padding-bottom: 2rem;
    }

    /* 기본 텍스트 */
    html, body, [class*="css"] {
        font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        "Apple SD Gothic Neo",
        "Noto Sans KR",
        sans-serif;
    }

    /* 상단 Hero */
    .hero {
        padding: 22px 26px;
        border-radius: 18px;
        background:
            linear-gradient(
                120deg,
                #eef7ff 0%,
                #f7fbff 55%,
                #ffffff 100%
            );
        border: 1px solid #dce9f5;
        margin-bottom: 18px;
    }

    .hero-title {
        font-size: 31px;
        font-weight: 800;
        color: #16324f;
        margin-bottom: 5px;
        line-height: 1.25;
    }

    .hero-sub {
        font-size: 15px;
        color: #52667a;
        line-height: 1.65;
        margin-bottom: 12px;
    }

    .badge {
        display: inline-block;
        padding: 5px 10px;
        margin-right: 5px;
        margin-bottom: 4px;
        border-radius: 999px;
        background: #ffffff;
        border: 1px solid #d8e4ee;
        color: #3e5367;
        font-size: 12px;
        font-weight: 600;
    }

    /* 섹션 제목 */
    .section-title {
        font-size: 20px;
        font-weight: 800;
        color: #20364a;
        margin-top: 4px;
        margin-bottom: 8px;
    }

    .section-desc {
        color: #6b7c8c;
        font-size: 13px;
        margin-bottom: 10px;
    }

    /* 결과 배너 */
    .decision-hems {
        padding: 19px 22px;
        border-radius: 16px;
        background: #fff4f1;
        border: 1px solid #f1c4b8;
        margin-top: 4px;
        margin-bottom: 14px;
    }

    .decision-ambulance {
        padding: 19px 22px;
        border-radius: 16px;
        background: #eef6ff;
        border: 1px solid #bfd8f2;
        margin-top: 4px;
        margin-bottom: 14px;
    }

    .decision-title {
        font-size: 25px;
        font-weight: 800;
        color: #1f3448;
        margin-bottom: 5px;
    }

    .decision-sub {
        color: #586d7e;
        font-size: 14px;
        line-height: 1.55;
    }

    /* KPI 카드 */
    .kpi-card {
        background: white;
        border: 1px solid #e1e7ec;
        border-radius: 14px;
        padding: 15px 16px;
        min-height: 110px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.025);
    }

    .kpi-label {
        color: #758493;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .kpi-value {
        color: #1e3448;
        font-size: 25px;
        font-weight: 800;
        line-height: 1.25;
    }

    .kpi-sub {
        color: #8b98a4;
        font-size: 11px;
        margin-top: 5px;
    }

    /* 작은 안내 카드 */
    .info-card {
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid #e0e6eb;
        background: #fafcfd;
        color: #4e6274;
        font-size: 13px;
        line-height: 1.6;
        margin-bottom: 10px;
    }

    /* 경로 라벨 */
    .route-title {
        font-size: 15px;
        font-weight: 800;
        margin-bottom: 2px;
    }

    .route-desc {
        font-size: 13px;
        color: #657787;
        margin-bottom: 12px;
        line-height: 1.55;
    }

    /* 모바일 */
    @media (max-width: 700px) {
        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
            padding-top: 0.8rem;
        }

        .hero {
            padding: 17px 17px;
        }

        .hero-title {
            font-size: 24px;
        }

        .hero-sub {
            font-size: 13px;
        }

        .decision-title {
            font-size: 21px;
        }

        .kpi-value {
            font-size: 21px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. 연구 기본값
# ============================================================

HOSPITAL_NAME = "아주대학교병원"

HOSPITAL_LAT = 37.2793433
HOSPITAL_LON = 127.0463045

HELI_SPEED_KMH = 267.0

HANDOVER_TIME_MIN = 4.0

N_SIM = 10_000

RANDOM_SEED = 20260913

MAX_WORKERS = 4

NAVER_ENDPOINT = (
    "https://maps.apigw.ntruss.com/"
    "map-direction/v1/driving"
)


# ============================================================
# 4. API Key
# ============================================================

try:

    NAVER_MAPS_KEY_ID = st.secrets[
        "NAVER_MAPS_KEY_ID"
    ]

    NAVER_MAPS_KEY = st.secrets[
        "NAVER_MAPS_KEY"
    ]

except Exception:

    st.error(
        "Naver Maps API 설정을 불러오지 못했습니다. "
        "Streamlit Secrets 설정을 확인해 주세요."
    )

    st.stop()


# ============================================================
# 5. 파일
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent


RP_FILE = (
    BASE_DIR
    / "data"
    / "경기도응급의료전용헬기인계점현황(2).csv"
)


# ============================================================
# 6. 유틸
# ============================================================

def read_csv_auto(file_path):

    encodings = [
        "utf-8-sig",
        "cp949",
        "euc-kr",
        "utf-8",
    ]

    last_error = None

    for encoding in encodings:

        try:

            df = pd.read_csv(
                file_path,
                encoding=encoding,
            )

            return df, encoding

        except UnicodeDecodeError as e:

            last_error = e


    raise RuntimeError(
        f"CSV 파일을 읽을 수 없습니다.\n{last_error}"
    )


def kpi_card(
    label,
    value,
    sub="",
):

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">
                {escape(str(label))}
            </div>
            <div class="kpi-value">
                {escape(str(value))}
            </div>
            <div class="kpi-sub">
                {escape(str(sub))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 7. 인계점 데이터
# ============================================================

@st.cache_data(show_spinner=False)
def load_rendezvous_points():

    if not RP_FILE.exists():

        raise FileNotFoundError(
            f"인계점 파일을 찾을 수 없습니다.\n"
            f"{RP_FILE}"
        )


    df, encoding = read_csv_auto(
        RP_FILE
    )


    required_columns = {
        "시군명",
        "지점명",
        "위도",
        "경도",
    }


    missing = (
        required_columns
        - set(df.columns)
    )


    if missing:

        raise ValueError(
            f"인계점 파일에 필요한 컬럼이 없습니다: {missing}"
        )


    df["위도"] = pd.to_numeric(
        df["위도"],
        errors="coerce",
    )

    df["경도"] = pd.to_numeric(
        df["경도"],
        errors="coerce",
    )


    bad_rows = df[
        df[
            [
                "위도",
                "경도",
            ]
        ]
        .isna()
        .any(axis=1)
    ]


    if len(bad_rows) > 0:

        raise ValueError(
            "인계점 원자료에 위도/경도 결측 또는 "
            "비정상 값이 있습니다. "
            "원자료를 임의 수정하지 않고 분석을 중단합니다."
        )


    df = df.reset_index(
        drop=True
    )


    df["인계점ID"] = [
        f"RP{i + 1:03d}"
        for i in range(len(df))
    ]


    return df, encoding


try:

    rp_df, detected_encoding = (
        load_rendezvous_points()
    )

except Exception as e:

    st.error(
        f"인계점 데이터를 불러오지 못했습니다.\n\n{e}"
    )

    st.stop()


# ============================================================
# 8. Haversine
# ============================================================

def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2,
):

    R = 6371.0088

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)

    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1_rad)
        * np.cos(lat2_rad)
        * np.sin(dlon / 2) ** 2
    )

    c = (
        2
        * np.arctan2(
            np.sqrt(a),
            np.sqrt(1 - a),
        )
    )

    return R * c


# ============================================================
# 9. 출동시간 Monte Carlo
# ============================================================

@st.cache_data(show_spinner=False)
def make_dispatch_samples():

    rng = np.random.default_rng(
        RANDOM_SEED
    )


    counts = np.array(
        [
            28,
            177,
            171,
            23,
            11,
        ],
        dtype=float,
    )


    probs = (
        counts
        / counts.sum()
    )


    categories = rng.choice(
        5,
        size=N_SIM,
        p=probs,
    )


    samples = np.zeros(
        N_SIM,
        dtype=float,
    )


    # ≤5
    idx = categories == 0

    samples[idx] = rng.uniform(
        0,
        5,
        size=idx.sum(),
    )


    # 6~8
    idx = categories == 1

    samples[idx] = rng.uniform(
        6,
        8,
        size=idx.sum(),
    )


    # 9~11
    idx = categories == 2

    samples[idx] = rng.uniform(
        9,
        11,
        size=idx.sum(),
    )


    # 12~15
    idx = categories == 3

    samples[idx] = rng.uniform(
        12,
        15,
        size=idx.sum(),
    )


    # ≥16 tail
    first_means = np.array(
        [
            2.5,
            7.0,
            10.0,
            13.5,
        ]
    )


    target_mean = 9.0


    tail_mean = (

        target_mean

        - np.sum(
            probs[:4]
            * first_means
        )

    ) / probs[4]


    exponential_mean = (
        tail_mean
        - 16.0
    )


    idx = categories == 4


    samples[idx] = (

        16.0

        + rng.exponential(
            scale=exponential_mean,
            size=idx.sum(),
        )

    )


    return samples


DISPATCH_SAMPLES = (
    make_dispatch_samples()
)


# ============================================================
# 10. Naver Directions
# ============================================================

def naver_route(
    start_lat,
    start_lon,
    goal_lat,
    goal_lon,
    retries=3,
):

    headers = {

        "x-ncp-apigw-api-key-id":
            NAVER_MAPS_KEY_ID,

        "x-ncp-apigw-api-key":
            NAVER_MAPS_KEY,

    }


    params = {

        "start":
            f"{start_lon},{start_lat}",

        "goal":
            f"{goal_lon},{goal_lat}",

        "option":
            "trafast",

    }


    last_error = None


    for attempt in range(
        retries
    ):

        try:

            response = requests.get(
                NAVER_ENDPOINT,
                headers=headers,
                params=params,
                timeout=15,
            )


            response.raise_for_status()


            data = response.json()


            if (
                "route" not in data
                or
                "trafast" not in data["route"]
                or
                len(data["route"]["trafast"]) == 0
            ):

                raise RuntimeError(
                    "Naver Directions API에서 "
                    "경로를 반환하지 않았습니다."
                )


            route = (
                data["route"]["trafast"][0]
            )


            summary = route[
                "summary"
            ]


            duration_min = (
                summary["duration"]
                / 1000
                / 60
            )


            distance_km = (
                summary["distance"]
                / 1000
            )


            raw_path = route.get(
                "path",
                []
            )


            # Naver = [경도, 위도]
            # Folium = [위도, 경도]
            path = [

                [
                    float(lat),
                    float(lon),
                ]

                for lon, lat
                in raw_path

            ]


            if len(path) < 2:

                raise RuntimeError(
                    "Naver API 응답에 "
                    "도로 경로 좌표(path)가 없습니다."
                )


            return {

                "success": True,

                "duration_min":
                    float(duration_min),

                "distance_km":
                    float(distance_km),

                "path":
                    path,

                "error":
                    None,

            }


        except Exception as e:

            last_error = str(e)


            if (
                attempt
                < retries - 1
            ):

                time.sleep(
                    0.5
                    * (attempt + 1)
                )


    return {

        "success": False,

        "duration_min":
            None,

        "distance_km":
            None,

        "path":
            None,

        "error":
            last_error,

    }


# ============================================================
# 11. 인계점 1개 분석
# ============================================================

def analyze_one_rp(
    accident_lat,
    accident_lon,
    row,
):

    rp_lat = float(
        row["위도"]
    )

    rp_lon = float(
        row["경도"]
    )


    road_result = naver_route(

        accident_lat,
        accident_lon,

        rp_lat,
        rp_lon,

    )


    if not road_result["success"]:

        return {

            "success": False,

            "인계점ID":
                row["인계점ID"],

            "지점명":
                row["지점명"],

            "시군명":
                row["시군명"],

            "error":
                road_result["error"],

        }


    road_time = (
        road_result[
            "duration_min"
        ]
    )


    flight_distance = haversine_km(

        HOSPITAL_LAT,
        HOSPITAL_LON,

        rp_lat,
        rp_lon,

    )


    flight_time = (

        flight_distance

        / HELI_SPEED_KMH

        * 60

    )


    hems_samples = (

        np.maximum(

            road_time,

            (
                DISPATCH_SAMPLES
                + flight_time
            )

        )

        + HANDOVER_TIME_MIN

        + flight_time

    )


    return {

        "success": True,

        "인계점ID":
            row["인계점ID"],

        "지점명":
            row["지점명"],

        "시군명":
            row["시군명"],

        "인계점위도":
            rp_lat,

        "인계점경도":
            rp_lon,

        "구급차_인계점시간_분":
            road_time,

        "구급차_인계점거리_km":
            road_result[
                "distance_km"
            ],

        "비행거리_km":
            flight_distance,

        "편도비행시간_분":
            flight_time,

        "HEMS평균시간_분":
            float(
                np.mean(
                    hems_samples
                )
            ),

        "HEMS표준편차_분":
            float(
                np.std(
                    hems_samples
                )
            ),

        "HEMS_P95_분":
            float(
                np.percentile(
                    hems_samples,
                    95,
                )
            ),

        # 추가 API 호출 없이 실제 도로선 사용
        "road_path":
            road_result["path"],

    }


# ============================================================
# 12. 전체 분석
# ============================================================

def analyze_location(
    accident_lat,
    accident_lon,
    progress_callback=None,
):

    # --------------------------------------------------------
    # A. 사고 → 병원 직접 구급차
    # --------------------------------------------------------

    direct_result = naver_route(

        accident_lat,
        accident_lon,

        HOSPITAL_LAT,
        HOSPITAL_LON,

    )


    if not direct_result["success"]:

        raise RuntimeError(
            "사고지점 → 아주대학교병원 "
            "직접 경로 조회에 실패했습니다.\n\n"
            f"{direct_result['error']}"
        )


    ambulance_time = (
        direct_result[
            "duration_min"
        ]
    )


    # --------------------------------------------------------
    # B. 181개 인계점
    # --------------------------------------------------------

    results = []


    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:


        futures = []


        for _, row in rp_df.iterrows():

            future = executor.submit(

                analyze_one_rp,

                accident_lat,
                accident_lon,
                row,

            )

            futures.append(
                future
            )


        completed = 0


        for future in as_completed(
            futures
        ):

            completed += 1


            try:

                results.append(
                    future.result()
                )

            except Exception as e:

                results.append({

                    "success": False,

                    "인계점ID": "-",
                    "지점명": "-",
                    "시군명": "-",

                    "error":
                        str(e),

                })


            if (
                progress_callback
                is not None
            ):

                progress_callback(
                    completed,
                    len(futures),
                )


    successful = [

        r
        for r in results
        if r["success"]

    ]


    failed = [

        r
        for r in results
        if not r["success"]

    ]


    if len(successful) == 0:

        raise RuntimeError(
            "모든 인계점의 도로정보 조회가 실패했습니다."
        )


    # --------------------------------------------------------
    # C. 기대시간 기준 최적 인계점
    # --------------------------------------------------------

    best = min(

        successful,

        key=lambda x:
            x["HEMS평균시간_분"]

    )


    # --------------------------------------------------------
    # D. 고정된 최적 인계점에서 우위확률
    # --------------------------------------------------------

    best_hems_samples = (

        np.maximum(

            best[
                "구급차_인계점시간_분"
            ],

            (
                DISPATCH_SAMPLES

                + best[
                    "편도비행시간_분"
                ]
            )

        )

        + HANDOVER_TIME_MIN

        + best[
            "편도비행시간_분"
        ]

    )


    hems_probability = (

        np.mean(

            best_hems_samples
            < ambulance_time

        )

        * 100

    )


    best_hems_mean = (
        best[
            "HEMS평균시간_분"
        ]
    )


    # --------------------------------------------------------
    # E. 추천
    # --------------------------------------------------------

    if (
        best_hems_mean
        < ambulance_time
    ):

        recommendation = (
            "닥터헬기"
        )

        system_expected_time = (
            best_hems_mean
        )

    else:

        recommendation = (
            "구급차"
        )

        system_expected_time = (
            ambulance_time
        )


    saved_time = (

        ambulance_time
        - system_expected_time

    )


    saved_rate = (

        (
            saved_time
            / ambulance_time
            * 100
        )

        if ambulance_time > 0

        else 0.0

    )


    # 두 수단 기대시간 차이
    mode_gap = (

        ambulance_time
        - best_hems_mean

    )


    # --------------------------------------------------------
    # F. 전체 인계점 표
    # --------------------------------------------------------

    rows = []


    for r in successful:

        rows.append({

            "인계점ID":
                r["인계점ID"],

            "시군명":
                r["시군명"],

            "지점명":
                r["지점명"],

            "구급차→인계점 시간(분)":
                r[
                    "구급차_인계점시간_분"
                ],

            "구급차→인계점 거리(km)":
                r[
                    "구급차_인계점거리_km"
                ],

            "헬기 편도거리(km)":
                r[
                    "비행거리_km"
                ],

            "헬기 편도시간(분)":
                r[
                    "편도비행시간_분"
                ],

            "HEMS 기대시간(분)":
                r[
                    "HEMS평균시간_분"
                ],

            "HEMS 표준편차(분)":
                r[
                    "HEMS표준편차_분"
                ],

            "HEMS P95(분)":
                r[
                    "HEMS_P95_분"
                ],

        })


    table_df = pd.DataFrame(
        rows
    )


    table_df = (

        table_df

        .sort_values(
            "HEMS 기대시간(분)"
        )

        .reset_index(
            drop=True
        )

    )


    return {

        "accident_lat":
            accident_lat,

        "accident_lon":
            accident_lon,

        "ambulance_time":
            ambulance_time,

        "ambulance_distance":
            direct_result[
                "distance_km"
            ],

        "direct_path":
            direct_result[
                "path"
            ],

        "best":
            best,

        "recommendation":
            recommendation,

        "hems_probability":
            float(
                hems_probability
            ),

        "system_expected_time":
            float(
                system_expected_time
            ),

        "saved_time":
            float(
                saved_time
            ),

        "saved_rate":
            float(
                saved_rate
            ),

        "mode_gap":
            float(
                mode_gap
            ),

        "failed":
            failed,

        "successful_count":
            len(successful),

        "table_df":
            table_df,

    }


# ============================================================
# 13. 위치 선택 지도
# ============================================================

def make_selection_map(
    selected_lat=None,
    selected_lon=None,
):

    m = folium.Map(

        location=[
            37.42,
            127.15
        ],

        zoom_start=9,

        control_scale=True,

    )


    # 아주대병원
    folium.Marker(

        [
            HOSPITAL_LAT,
            HOSPITAL_LON
        ],

        tooltip="아주대학교병원",

        icon=folium.Icon(
            color="darkred",
            icon="plus-sign",
        ),

    ).add_to(m)


    # 현재 선택 위치
    if (
        selected_lat is not None
        and
        selected_lon is not None
    ):

        folium.Marker(

            [
                selected_lat,
                selected_lon
            ],

            tooltip="선택한 사고지점",

            icon=folium.Icon(
                color="red",
                icon="info-sign",
            ),

        ).add_to(m)


    return m


# ============================================================
# 14. 결과 지도
# ============================================================

def make_result_map(
    result
):

    accident_lat = (
        result[
            "accident_lat"
        ]
    )

    accident_lon = (
        result[
            "accident_lon"
        ]
    )

    best = result[
        "best"
    ]

    rp_lat = best[
        "인계점위도"
    ]

    rp_lon = best[
        "인계점경도"
    ]


    m = folium.Map(

        location=[
            accident_lat,
            accident_lon
        ],

        zoom_start=10,

        control_scale=True,

    )


    # --------------------------------------------------------
    # 레이어
    # --------------------------------------------------------

    direct_layer = (
        folium.FeatureGroup(
            name="구급차 직접이송",
            show=True,
        )
    )

    rp_layer = (
        folium.FeatureGroup(
            name="사고→최적 인계점",
            show=True,
        )
    )

    heli_layer = (
        folium.FeatureGroup(
            name="닥터헬기 비행",
            show=True,
        )
    )


    # --------------------------------------------------------
    # 사고지점
    # --------------------------------------------------------

    folium.Marker(

        [
            accident_lat,
            accident_lon
        ],

        tooltip="사고지점",

        popup=(
            f"<b>사고지점</b><br>"
            f"위도: {accident_lat:.6f}<br>"
            f"경도: {accident_lon:.6f}"
        ),

        icon=folium.Icon(
            color="red",
            icon="info-sign",
        ),

    ).add_to(m)


    # --------------------------------------------------------
    # 병원
    # --------------------------------------------------------

    folium.Marker(

        [
            HOSPITAL_LAT,
            HOSPITAL_LON
        ],

        tooltip=HOSPITAL_NAME,

        icon=folium.Icon(
            color="darkred",
            icon="plus-sign",
        ),

    ).add_to(m)


    # --------------------------------------------------------
    # 최적 인계점
    # --------------------------------------------------------

    rp_name = escape(
        str(
            best["지점명"]
        )
    )

    rp_city = escape(
        str(
            best["시군명"]
        )
    )


    folium.Marker(

        [
            rp_lat,
            rp_lon
        ],

        tooltip=(
            f"최적 인계점: "
            f"{best['지점명']}"
        ),

        popup=(

            f"<b>최적 인계점</b><br>"
            f"{rp_name}<br>"
            f"{rp_city}<br><br>"

            f"사고→인계점: "
            f"{best['구급차_인계점시간_분']:.1f}분<br>"

            f"HEMS 기대시간: "
            f"{best['HEMS평균시간_분']:.1f}분"

        ),

        icon=folium.Icon(
            color="green",
            icon="flag",
        ),

    ).add_to(m)


    # --------------------------------------------------------
    # 사고 → 병원 실제 도로
    # --------------------------------------------------------

    folium.PolyLine(

        result[
            "direct_path"
        ],

        color="#737c84",

        weight=5,

        opacity=0.72,

        tooltip=(
            "구급차 직접이송 "
            "(Naver 실제 도로경로)"
        ),

    ).add_to(
        direct_layer
    )


    # --------------------------------------------------------
    # 사고 → 인계점 실제 도로
    # --------------------------------------------------------

    folium.PolyLine(

        best[
            "road_path"
        ],

        color="#1479c9",

        weight=6,

        opacity=0.92,

        tooltip=(
            "사고지점 → 최적 인계점 "
            "(Naver 실제 도로경로)"
        ),

    ).add_to(
        rp_layer
    )


    # --------------------------------------------------------
    # 병원 ↔ 인계점 헬기
    # --------------------------------------------------------

    folium.PolyLine(

        [

            [
                HOSPITAL_LAT,
                HOSPITAL_LON
            ],

            [
                rp_lat,
                rp_lon
            ],

        ],

        color="#e24a33",

        weight=5,

        opacity=0.9,

        dash_array="10,8",

        tooltip=(
            "닥터헬기 비행구간"
        ),

    ).add_to(
        heli_layer
    )


    direct_layer.add_to(m)
    rp_layer.add_to(m)
    heli_layer.add_to(m)


    folium.LayerControl(
        collapsed=True
    ).add_to(m)


    # --------------------------------------------------------
    # 범례
    # --------------------------------------------------------

    legend = """

    <div style="
        position: fixed;
        left: 24px;
        bottom: 24px;
        z-index: 9999;
        background: rgba(255,255,255,0.95);
        padding: 10px 12px;
        border: 1px solid #aaa;
        border-radius: 8px;
        font-size: 12px;
        line-height: 1.75;
    ">

        <b>이송 경로</b><br>

        <span style="
            color:#1479c9;
            font-weight:bold;
        ">
        ━━━
        </span>

        사고 → 인계점<br>

        <span style="
            color:#737c84;
            font-weight:bold;
        ">
        ━━━
        </span>

        사고 → 병원<br>

        <span style="
            color:#e24a33;
            font-weight:bold;
        ">
        ┄┄┄
        </span>

        닥터헬기 비행

    </div>

    """


    m.get_root().html.add_child(
        folium.Element(
            legend
        )
    )


    # --------------------------------------------------------
    # 전체 범위
    # --------------------------------------------------------

    m.fit_bounds(

        [

            [
                min(
                    accident_lat,
                    HOSPITAL_LAT,
                    rp_lat,
                ),

                min(
                    accident_lon,
                    HOSPITAL_LON,
                    rp_lon,
                ),
            ],

            [
                max(
                    accident_lat,
                    HOSPITAL_LAT,
                    rp_lat,
                ),

                max(
                    accident_lon,
                    HOSPITAL_LON,
                    rp_lon,
                ),
            ],

        ]

    )


    return m


# ============================================================
# 15. Session State
# ============================================================

if (
    "selected_lat"
    not in st.session_state
):

    st.session_state[
        "selected_lat"
    ] = None


if (
    "selected_lon"
    not in st.session_state
):

    st.session_state[
        "selected_lon"
    ] = None


if (
    "analysis_result"
    not in st.session_state
):

    st.session_state[
        "analysis_result"
    ] = None


# ============================================================
# 16. Sidebar
# ============================================================

with st.sidebar:

    st.markdown(
        "## 연구 기준"
    )

    st.markdown(
        f"""
**최종 병원**  
{HOSPITAL_NAME}

**현행 인계점**  
{len(rp_df)}개

**Monte Carlo**  
{N_SIM:,}회

**인계점 체류시간**  
{HANDOVER_TIME_MIN:.0f}분

**헬기 속도**  
{HELI_SPEED_KMH:.1f} km/h

**도로시간**  
Naver Directions `trafast`
"""
    )


    st.divider()


    st.caption(
        "본 시스템은 이송시간 비교를 위한 "
        "의사결정지원 연구모형이며 실제 의료·운항 "
        "판단을 대체하지 않습니다."
    )


# ============================================================
# ============================================================
# 17. Hero
# ============================================================

st.markdown(
    """
<div class="hero">
<div class="hero-title">
🚁 닥터헬기 이송수단 의사결정지원시스템
</div>

<div class="hero-sub">
사고 위치를 선택하면 실시간 도로상황과
헬기 출동시간의 불확실성을 함께 고려하여
<b>구급차와 닥터헬기 중 기대 이송시간이 더 짧은 수단</b>을 제안합니다.
</div>

<span class="badge">① 사고 위치 선택</span>
<span class="badge">② 181개 인계점 분석</span>
<span class="badge">③ 이송수단 추천</span>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 18. 사고 위치 선택
# ============================================================

st.markdown(
    '<div class="section-title">사고 위치 선택</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-desc">'
    '지도에서 실제 사고가 발생했다고 가정할 위치를 한 번 클릭하세요.'
    '</div>',
    unsafe_allow_html=True,
)


map_col, action_col = st.columns(
    [
        2.35,
        1
    ],
    gap="large",
)


# ------------------------------------------------------------
# 지도
# ------------------------------------------------------------

with map_col:

    picker_map = make_selection_map(

        st.session_state[
            "selected_lat"
        ],

        st.session_state[
            "selected_lon"
        ],

    )


    map_data = st_folium(

        picker_map,

        width=None,

        height=430,

        key="accident_picker",

    )


# ------------------------------------------------------------
# 클릭 처리
# ------------------------------------------------------------

if (
    map_data
    and
    map_data.get(
        "last_clicked"
    )
):

    new_lat = float(
        map_data[
            "last_clicked"
        ]["lat"]
    )

    new_lon = float(
        map_data[
            "last_clicked"
        ]["lng"]
    )


    old_lat = (
        st.session_state[
            "selected_lat"
        ]
    )

    old_lon = (
        st.session_state[
            "selected_lon"
        ]
    )


    changed = (

        old_lat is None

        or old_lon is None

        or abs(
            new_lat
            - old_lat
        ) > 1e-8

        or abs(
            new_lon
            - old_lon
        ) > 1e-8

    )


    if changed:

        st.session_state[
            "selected_lat"
        ] = new_lat

        st.session_state[
            "selected_lon"
        ] = new_lon

        st.session_state[
            "analysis_result"
        ] = None

        st.rerun()


# ------------------------------------------------------------
# 우측 조작 패널
# ------------------------------------------------------------

with action_col:

    with st.container(
        border=True
    ):

        st.markdown(
            "### 선택 위치"
        )


        selected = (

            st.session_state[
                "selected_lat"
            ]
            is not None

            and

            st.session_state[
                "selected_lon"
            ]
            is not None

        )


        if selected:

            st.success(
                "사고 위치가 선택되었습니다."
            )


            st.markdown(
                f"""
**위도**  
`{st.session_state['selected_lat']:.6f}`

**경도**  
`{st.session_state['selected_lon']:.6f}`
"""
            )

        else:

            st.info(
                "왼쪽 지도에서 사고 위치를 클릭하세요."
            )


        analyze_clicked = st.button(

            "🚁 이송수단 분석하기",

            type="primary",

            use_container_width=True,

            disabled=not selected,

        )


        if st.button(

            "선택 위치 초기화",

            use_container_width=True,

            disabled=not selected,

        ):

            st.session_state[
                "selected_lat"
            ] = None

            st.session_state[
                "selected_lon"
            ] = None

            st.session_state[
                "analysis_result"
            ] = None

            st.rerun()


        st.caption(
            "분석 시 사고→병원 1개 경로와 "
            "사고→인계점 181개 경로를 실시간 조회합니다."
        )


# ============================================================
# 19. 분석 실행
# ============================================================

if analyze_clicked:

    progress = st.progress(

        0,

        text=(
            "실시간 도로정보를 조회하고 있습니다..."
        )

    )


    def update_progress(
        completed,
        total,
    ):

        ratio = (
            completed
            / total
        )


        progress.progress(

            ratio,

            text=(
                f"인계점 분석 중 "
                f"{completed}/{total}"
            )

        )


    try:

        result = analyze_location(

            st.session_state[
                "selected_lat"
            ],

            st.session_state[
                "selected_lon"
            ],

            progress_callback=
                update_progress,

        )


        st.session_state[
            "analysis_result"
        ] = result


        progress.progress(

            1.0,

            text="분석 완료"

        )


        time.sleep(
            0.25
        )


        progress.empty()


    except Exception as e:

        progress.empty()

        st.error(
            f"분석 중 오류가 발생했습니다.\n\n{e}"
        )


# ============================================================
# 20. 결과
# ============================================================

result = st.session_state[
    "analysis_result"
]


if result is not None:

    st.divider()


    recommendation = result[
        "recommendation"
    ]

    best = result[
        "best"
    ]

    mode_gap = result[
        "mode_gap"
    ]


    # --------------------------------------------------------
    # 추천 배너
    # --------------------------------------------------------

    if recommendation == "닥터헬기":

        banner_class = (
            "decision-hems"
        )

        banner_title = (
            "🚁 닥터헬기를 추천합니다"
        )

        banner_sub = (
            f"기대시간 기준으로 구급차 직접이송보다 "
            f"약 {abs(mode_gap):.1f}분 빠릅니다. "
            f"최적 인계점은 "
            f"{best['지점명']}입니다."
        )

    else:

        banner_class = (
            "decision-ambulance"
        )

        banner_title = (
            "🚑 구급차 직접이송을 추천합니다"
        )

        banner_sub = (
            f"기대시간 기준으로 닥터헬기보다 "
            f"약 {abs(mode_gap):.1f}분 빠릅니다. "
            f"이 위치에서는 인계·출동 과정으로 인한 "
            f"추가시간이 더 큽니다."
        )


    st.markdown(

        f"""
        <div class="{banner_class}">

            <div class="decision-title">
                {escape(banner_title)}
            </div>

            <div class="decision-sub">
                {escape(banner_sub)}
            </div>

        </div>
        """,

        unsafe_allow_html=True,

    )


    # --------------------------------------------------------
    # 핵심 KPI 4개
    # --------------------------------------------------------

    k1, k2, k3, k4 = st.columns(
        4,
        gap="small",
    )


    with k1:

        kpi_card(

            "구급차 직접이송",

            f"{result['ambulance_time']:.1f}분",

            (
                f"{result['ambulance_distance']:.1f} km"
            ),

        )


    with k2:

        kpi_card(

            "닥터헬기 기대시간",

            f"{best['HEMS평균시간_분']:.1f}분",

            "최적 인계점 기준",

        )


    with k3:

        kpi_card(

            "HEMS 우위확률",

            f"{result['hems_probability']:.1f}%",

            "Monte Carlo 10,000회",

        )


    with k4:

        kpi_card(

            "시스템 기대 절감",

            f"{result['saved_time']:.1f}분",

            (
                f"{result['saved_rate']:.1f}% 절감"
            ),

        )


    # --------------------------------------------------------
    # API 실패 알림
    # --------------------------------------------------------

    if len(
        result[
            "failed"
        ]
    ) > 0:

        st.warning(

            f"현행 {len(rp_df)}개 인계점 중 "
            f"{result['successful_count']}개 경로 조회에 성공했습니다. "
            f"{len(result['failed'])}개는 API 조회에 실패했으므로 "
            f"현재 최적화 결과는 성공한 인계점을 기준으로 계산되었습니다."

        )


    # ========================================================
    # 21. Tabs
    # ========================================================

    tab_route, tab_compare, tab_rp, tab_method = (
        st.tabs(
            [
                "🗺️ 경로",
                "⚖️ 수단 비교",
                "📍 인계점 순위",
                "ℹ️ 분석 방법",
            ]
        )
    )


    # ========================================================
    # TAB 1 - 경로
    # ========================================================

    with tab_route:

        route_map_col, route_info_col = (
            st.columns(
                [
                    2.35,
                    1,
                ],
                gap="large",
            )
        )


        with route_map_col:

            result_map = (
                make_result_map(
                    result
                )
            )


            st_folium(

                result_map,

                width=None,

                height=540,

                key="result_route_map",

            )


        with route_info_col:

            with st.container(
                border=True
            ):

                st.markdown(
                    "### 최적 인계점"
                )

                st.markdown(
                    f"**{best['지점명']}**"
                )

                st.caption(
                    best[
                        "시군명"
                    ]
                )


                st.divider()


                st.markdown(
                    '<div class="route-title">'
                    '🚑 사고 → 인계점'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
<div class="route-desc">
Naver 실제 도로경로<br>
<b>{best['구급차_인계점시간_분']:.1f}분</b>
· {best['구급차_인계점거리_km']:.1f} km
</div>
""",
                    unsafe_allow_html=True,
                )


                st.markdown(
                    '<div class="route-title">'
                    '🚁 병원 ↔ 인계점'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
<div class="route-desc">
직선 비행거리 기반<br>
편도 <b>{best['편도비행시간_분']:.1f}분</b>
· {best['비행거리_km']:.1f} km
</div>
""",
                    unsafe_allow_html=True,
                )


                st.markdown(
                    '<div class="route-title">'
                    '🏥 최종 도착'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    """
<div class="route-desc">
아주대학교병원
</div>
""",
                    unsafe_allow_html=True,
                )


            st.caption(
                "파란색·회색 실선은 Naver API가 반환한 "
                "실제 자동차 도로경로이며, 빨간 점선은 "
                "헬기의 거리 기반 비행구간입니다."
            )


    # ========================================================
    # TAB 2 - 수단 비교
    # ========================================================

    with tab_compare:

        compare_left, compare_right = (
            st.columns(
                2,
                gap="large",
            )
        )


        with compare_left:

            with st.container(
                border=True
            ):

                st.markdown(
                    "### 🚑 구급차 직접이송"
                )

                st.metric(

                    "예상 이송시간",

                    f"{result['ambulance_time']:.1f}분",

                )


                st.metric(

                    "도로거리",

                    f"{result['ambulance_distance']:.1f} km",

                )


                st.caption(
                    "사고지점에서 아주대학교병원까지 "
                    "Naver 실시간 도로정보를 이용합니다."
                )


        with compare_right:

            with st.container(
                border=True
            ):

                st.markdown(
                    "### 🚁 닥터헬기"
                )


                st.metric(

                    "기대 이송시간",

                    f"{best['HEMS평균시간_분']:.1f}분",

                )


                st.metric(

                    "95% 백분위 시간",

                    f"{best['HEMS_P95_분']:.1f}분",

                )


                st.caption(
                    "출동시간 불확실성을 Monte Carlo로 반영한 "
                    "최적 인계점 기준 결과입니다."
                )


        st.markdown(
            "#### 닥터헬기 우위확률"
        )


        probability = float(
            result[
                "hems_probability"
            ]
        )


        st.progress(
            min(
                max(
                    probability / 100,
                    0.0,
                ),
                1.0,
            )
        )


        st.caption(
            f"10,000회 시뮬레이션 중 약 "
            f"{probability:.1f}%에서 닥터헬기 총 이송시간이 "
            f"구급차 직접이송시간보다 짧았습니다."
        )


        st.info(
            "최종 추천은 우위확률 50% 같은 임의 기준이 아니라 "
            "**두 수단의 기대 이송시간 비교**를 기준으로 결정합니다."
        )


    # ========================================================
    # TAB 3 - 인계점 순위
    # ========================================================

    with tab_rp:

        st.markdown(
            f"**현행 {len(rp_df)}개 인계점 중 "
            f"HEMS 기대시간이 가장 짧은 순서입니다.**"
        )


        display_df = (

            result[
                "table_df"
            ]

            .head(10)

            .copy()

        )


        numeric_cols = [

            "구급차→인계점 시간(분)",
            "구급차→인계점 거리(km)",
            "헬기 편도거리(km)",
            "헬기 편도시간(분)",
            "HEMS 기대시간(분)",
            "HEMS 표준편차(분)",
            "HEMS P95(분)",

        ]


        for col in numeric_cols:

            display_df[col] = (
                display_df[col]
                .round(2)
            )


        st.dataframe(

            display_df,

            use_container_width=True,

            hide_index=True,

            height=380,

        )


        csv_data = (

            result[
                "table_df"
            ]

            .to_csv(
                index=False
            )

            .encode(
                "utf-8-sig"
            )

        )


        st.download_button(

            "전체 인계점 분석결과 CSV 다운로드",

            data=csv_data,

            file_name=(
                "닥터헬기_인계점_분석결과.csv"
            ),

            mime="text/csv",

            use_container_width=True,

        )


        # ========================================================
    # TAB 4 - 분석 방법
    # ========================================================

    with tab_method:

        method_left, method_right = st.columns(
            2,
            gap="large",
        )

        with method_left:

            st.markdown("### 계산 구조")

            st.markdown("**구급차 직접이송**")

            st.write(
                "사고지점에서 아주대학교병원까지의 "
                "Naver 실시간 도로 이동시간을 사용합니다."
            )

            st.markdown("**닥터헬기 총 이송시간**")

            st.code(
                "max(\n"
                "    사고→인계점 구급차시간,\n"
                "    출동시간 + 병원→인계점 비행시간\n"
                ")\n"
                "+ 인계점 체류시간\n"
                "+ 인계점→병원 비행시간",
                language="text",
            )

            st.write(
                "구급차와 닥터헬기가 동시에 출발한다고 가정하며, "
                "먼저 인계점에 도착한 수단은 상대 수단이 도착할 때까지 기다립니다."
            )

        with method_right:

            st.markdown("### 기준 시나리오")

            st.write(f"• Monte Carlo: **{N_SIM:,}회**")
            st.write(f"• 고정 시드: **{RANDOM_SEED}**")
            st.write(
                f"• 인계점 체류시간: "
                f"**{HANDOVER_TIME_MIN:.0f}분**"
            )
            st.write(
                f"• 헬기 속도: "
                f"**{HELI_SPEED_KMH:.1f} km/h**"
            )
            st.write(
                f"• 현행 인계점 후보: "
                f"**{len(rp_df)}개**"
            )
            st.write(
                f"• 최종 도착병원: "
                f"**{HOSPITAL_NAME}**"
            )
            st.write(
                "• 도로시간 및 경로: "
                "**Naver Directions `trafast`**"
            )

        st.info(
            "최종 이송수단은 HEMS 우위확률에 임의의 기준값을 "
            "적용하여 결정하지 않습니다. "
            "구급차 직접이송시간과 최적 인계점의 "
            "닥터헬기 기대 이송시간을 비교하여 결정합니다."
        )

        st.warning(
            "Naver 도로시간은 일반 도로교통 상황을 기반으로 하며 "
            "응급차량의 신호 우선 또는 긴급주행 효과를 직접 반영하지 않습니다. "
            "또한 지도상의 헬기 직선은 실제 세부 비행궤적이 아니라 "
            "본 연구의 거리 기반 비행시간 모형을 시각적으로 나타낸 것입니다."
        )

        st.caption(
            "본 시스템은 연구 및 의사결정지원용 시뮬레이션이며 "
            "실제 의료진·운항관리사의 출동 및 의료 판단을 대체하지 않습니다."
        )
