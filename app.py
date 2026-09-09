# ============================================================
# 경기도 닥터헬기 이송수단 의사결정지원 시스템
#
# 기능
# 1. 지도에서 사고 위치 클릭
# 2. Naver Directions API로 실시간 도로시간 조회
# 3. 사고지점 → 181개 인계점 전체 평가
# 4. 아주대학교병원 ↔ 인계점 헬기 비행시간 계산
# 5. 닥터헬기 출동시간 Monte Carlo 10,000회
# 6. 기대 HEMS 시간이 최소인 인계점 선택
# 7. 구급차 직접이송 vs 닥터헬기 비교
# 8. 최종 추천수단과 HEMS 우위확률 제시
#
# [중요]
# - 연구용 프로토타입
# - 실제 의료/출동 판단용 공식 시스템이 아님
# ============================================================


# ============================================================
# 0. 라이브러리
# ============================================================

import os
import time

from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

import streamlit as st
import folium

from streamlit_folium import st_folium


# ============================================================
# 1. Streamlit 기본 설정
# ============================================================

st.set_page_config(

    page_title="닥터헬기 이송수단 의사결정",

    page_icon="🚁",

    layout="wide"

)


# ============================================================
# 2. 연구 고정값
# ============================================================

# 최종 병원 및 헬기 출발지
HOSPITAL_NAME = "아주대학교병원"

HOSPITAL_LAT = 37.2793433
HOSPITAL_LON = 127.0463045


# 인계점 체류시간
# HEMS on-scene time
HANDOVER_TIME = 4.0


# Monte Carlo 반복 횟수
N_SIM = 10000


# Random Seed
RANDOM_SEED = 20260913


# ============================================================
# 3. Naver Directions API
# ============================================================

NAVER_URL = (
    "https://maps.apigw.ntruss.com/"
    "map-direction/v1/driving"
)


# 기존 연구와 동일한 경로 옵션
ROUTE_OPTION = "trafast"


# ============================================================
# 4. API Key 읽기
#
# .streamlit/secrets.toml
#
# NAVER_MAPS_KEY_ID = "..."
# NAVER_MAPS_KEY = "..."
#
# ============================================================

def read_secret(name):

    try:

        return str(
            st.secrets[name]
        )

    except Exception:

        return os.getenv(
            name,
            ""
        )


NAVER_KEY_ID = read_secret(
    "NAVER_MAPS_KEY_ID"
)

NAVER_KEY = read_secret(
    "NAVER_MAPS_KEY"
)


# ============================================================
# 5. CSV 인코딩 자동 읽기
#
# Excel/Windows에서 저장된 CSV의 경우
# CP949일 수 있으므로 자동으로 확인
# ============================================================

def read_csv_auto(file_path):

    encodings = [

        "utf-8-sig",

        "cp949",

        "euc-kr",

        "utf-8"

    ]


    last_error = None


    for encoding in encodings:

        try:

            df = pd.read_csv(

                file_path,

                encoding=encoding

            )


            return (
                df,
                encoding
            )


        except UnicodeDecodeError as e:

            last_error = e


    # 모든 인코딩 실패
    raise RuntimeError(

        "CSV 파일의 인코딩을 읽을 수 없습니다.\n"
        f"마지막 오류: {last_error}"

    )


# ============================================================
# 6. Haversine 직선거리
# ============================================================

def haversine_distance(

    lat1,
    lon1,

    lat2,
    lon2

):

    EARTH_RADIUS_KM = 6371.0


    lat1 = radians(lat1)
    lon1 = radians(lon1)

    lat2 = radians(lat2)
    lon2 = radians(lon2)


    dlat = (
        lat2
        - lat1
    )

    dlon = (
        lon2
        - lon1
    )


    a = (

        sin(
            dlat / 2
        ) ** 2

        +

        cos(lat1)
        * cos(lat2)
        * sin(
            dlon / 2
        ) ** 2

    )


    c = (

        2

        * atan2(

            sqrt(a),

            sqrt(
                1 - a
            )

        )

    )


    return (

        EARTH_RADIUS_KM
        * c

    )


# ============================================================
# 7. 인계점 데이터 불러오기
#
# 아래 파일 중 존재하는 파일 자동 탐색
# ============================================================

@st.cache_data
def load_rendezvous_points():


    candidate_files = [

        Path(
            "data/경기도응급의료전용헬기인계점현황.csv"
        ),

        Path(
            "data/경기도응급의료전용헬기인계점현황(2).csv"
        ),

        Path(
            "경기도응급의료전용헬기인계점현황.csv"
        ),

        Path(
            "경기도응급의료전용헬기인계점현황(2).csv"
        ),

        Path(
            "data/네이버API_도로이동시간_최종.csv"
        ),

        Path(
            "네이버API_도로이동시간_최종.csv"
        )

    ]


    selected_file = None


    for file in candidate_files:

        if file.exists():

            selected_file = file

            break


    if selected_file is None:

        raise FileNotFoundError(

            "인계점 CSV 파일을 찾을 수 없습니다.\n\n"

            "data 폴더 안에\n"

            "경기도응급의료전용헬기인계점현황(2).csv\n"

            "파일이 있는지 확인해주세요."

        )


    # ========================================================
    # 인코딩 자동 탐색
    # ========================================================

    df, detected_encoding = (
        read_csv_auto(
            selected_file
        )
    )


    # ========================================================
    # 경우 A
    #
    # 기존 네이버 API 최종 파일을 이용하는 경우
    # ========================================================

    if {

        "경로유형",
        "목적지ID",
        "목적지명",
        "도착위도",
        "도착경도"

    }.issubset(
        df.columns
    ):


        rp = df[

            df[
                "경로유형"
            ]
            == "구급차_인계점"

        ][

            [
                "목적지ID",
                "목적지시군",
                "목적지명",
                "도착위도",
                "도착경도"
            ]

        ].copy()


        # 한 인계점이 사고 100개에 반복되어 있으므로
        # 목적지ID 기준 중복 제거

        rp = (

            rp

            .drop_duplicates(
                subset=[
                    "목적지ID"
                ]
            )

            .reset_index(
                drop=True
            )

        )


        rp = rp.rename(

            columns={

                "목적지ID":
                    "인계점ID",

                "목적지시군":
                    "시군명",

                "목적지명":
                    "인계점명",

                "도착위도":
                    "위도",

                "도착경도":
                    "경도"

            }

        )


    # ========================================================
    # 경우 B
    #
    # 경기도 인계점 원본 CSV를 이용하는 경우
    # ========================================================

    elif {

        "지점명",
        "위도",
        "경도"

    }.issubset(
        df.columns
    ):


        rp = df.copy()


        rp = rp.rename(

            columns={

                "지점명":
                    "인계점명"

            }

        )


        if (
            "시군명"
            not in rp.columns
        ):

            rp[
                "시군명"
            ] = ""


        rp = rp[

            [
                "시군명",
                "인계점명",
                "위도",
                "경도"
            ]

        ].copy()


        # 인계점 ID 생성

        rp.insert(

            0,

            "인계점ID",

            [

                f"RP{i:03d}"

                for i in range(
                    1,
                    len(rp) + 1
                )

            ]

        )


    else:

        raise ValueError(

            "인계점 CSV의 컬럼 구조를 확인할 수 없습니다.\n\n"

            f"현재 컬럼:\n{list(df.columns)}"

        )


    # ========================================================
    # 위도/경도 숫자형 확인
    # ========================================================

    rp[
        "위도"
    ] = pd.to_numeric(

        rp[
            "위도"
        ],

        errors="coerce"

    )


    rp[
        "경도"
    ] = pd.to_numeric(

        rp[
            "경도"
        ],

        errors="coerce"

    )


    # 좌표 결측값은 삭제하지 않고
    # 먼저 개수 확인

    invalid_coordinate_count = (

        rp[
            [
                "위도",
                "경도"
            ]
        ]

        .isna()

        .any(
            axis=1
        )

        .sum()

    )


    if (
        invalid_coordinate_count
        > 0
    ):

        raise ValueError(

            f"인계점 중 좌표가 없는 행이 "
            f"{invalid_coordinate_count}개 있습니다.\n"

            "원본 데이터를 확인해주세요."

        )


    rp = rp.reset_index(
        drop=True
    )


    return (

        rp,

        str(
            selected_file
        ),

        detected_encoding

    )


# ============================================================
# 8. 출동시간 Monte Carlo 표본 생성
#
# 최종 연구모형과 동일
# ============================================================

@st.cache_data
def generate_dispatch_samples():


    # 2024 경기도 닥터헬기
    # 결정 → 이륙시간 410건

    counts = np.array(

        [
            28,
            177,
            171,
            23,
            11
        ],

        dtype=float

    )


    probabilities = (

        counts

        / counts.sum()

    )


    # 공식 평균
    OFFICIAL_MEAN = 9.0


    # 유한 구간 평균
    bounded_means = np.array(

        [
            2.5,    # 0~5
            7.0,    # 6~8
            10.0,   # 9~11
            13.5    # 12~15
        ]

    )


    known_time_sum = np.sum(

        counts[:4]

        * bounded_means

    )


    # 전체 평균 9분을 만족하도록
    # 16분 이상 구간 평균 역산

    tail_mean = (

        OFFICIAL_MEAN
        * counts.sum()

        - known_time_sum

    ) / counts[4]


    TAIL_START = 16.0


    tail_excess_mean = (

        tail_mean

        - TAIL_START

    )


    # ========================================================
    # Random Seed 고정
    # ========================================================

    rng = np.random.default_rng(

        RANDOM_SEED

    )


    # 먼저 구간 선택

    selected_bins = rng.choice(

        5,

        size=N_SIM,

        p=probabilities

    )


    dispatch_samples = np.empty(

        N_SIM,

        dtype=float

    )


    # --------------------------------------------------------
    # 5분 이하
    # --------------------------------------------------------

    mask = (
        selected_bins == 0
    )


    dispatch_samples[
        mask
    ] = rng.uniform(

        0,
        5,

        size=mask.sum()

    )


    # --------------------------------------------------------
    # 6~8분
    # --------------------------------------------------------

    mask = (
        selected_bins == 1
    )


    dispatch_samples[
        mask
    ] = rng.uniform(

        6,
        8,

        size=mask.sum()

    )


    # --------------------------------------------------------
    # 9~11분
    # --------------------------------------------------------

    mask = (
        selected_bins == 2
    )


    dispatch_samples[
        mask
    ] = rng.uniform(

        9,
        11,

        size=mask.sum()

    )


    # --------------------------------------------------------
    # 12~15분
    # --------------------------------------------------------

    mask = (
        selected_bins == 3
    )


    dispatch_samples[
        mask
    ] = rng.uniform(

        12,
        15,

        size=mask.sum()

    )


    # --------------------------------------------------------
    # 16분 이상
    # --------------------------------------------------------

    mask = (
        selected_bins == 4
    )


    dispatch_samples[
        mask
    ] = (

        16

        +

        rng.exponential(

            scale=tail_excess_mean,

            size=mask.sum()

        )

    )


    return dispatch_samples


# ============================================================
# 9. Naver Directions API 요청
# ============================================================

def naver_route(

    start_lat,
    start_lon,

    goal_lat,
    goal_lon

):


    headers = {

        "x-ncp-apigw-api-key-id":
            NAVER_KEY_ID,

        "x-ncp-apigw-api-key":
            NAVER_KEY

    }


    params = {

        # 중요:
        # Naver API 좌표순서는
        # 경도,위도

        "start":
            f"{start_lon},{start_lat}",

        "goal":
            f"{goal_lon},{goal_lat}",

        "option":
            ROUTE_OPTION

    }


    last_error = None


    # 최대 3회 재시도

    for attempt in range(3):


        try:


            query_time = (

                datetime.now()

                .strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

            )


            response = requests.get(

                NAVER_URL,

                headers=headers,

                params=params,

                timeout=20

            )


            # =================================================
            # 성공
            # =================================================

            if (
                response.status_code
                == 200
            ):


                data = response.json()


                route_object = (

                    data.get(
                        "route",
                        {}
                    )

                )


                routes = (

                    route_object.get(
                        ROUTE_OPTION,
                        []
                    )

                )


                # 혹시 응답 key가 다를 경우
                # 사용 가능한 첫 경로 탐색

                if (
                    len(routes)
                    == 0
                ):


                    for (
                        key,
                        value
                    ) in route_object.items():


                        if (

                            isinstance(
                                value,
                                list
                            )

                            and

                            len(value) > 0

                        ):

                            routes = value

                            break


                if (
                    len(routes)
                    == 0
                ):

                    raise RuntimeError(

                        "사용 가능한 도로경로가 없습니다."

                    )


                summary = (

                    routes[0][
                        "summary"
                    ]

                )


                # duration = millisecond

                duration_min = (

                    float(
                        summary[
                            "duration"
                        ]
                    )

                    / 1000

                    / 60

                )


                # distance = meter

                distance_km = (

                    float(
                        summary[
                            "distance"
                        ]
                    )

                    / 1000

                )


                return {

                    "time_min":
                        duration_min,

                    "distance_km":
                        distance_km,

                    "query_time":
                        query_time

                }


            # =================================================
            # 일시적 서버 오류 / 요청 제한
            # =================================================

            if (
                response.status_code
                in
                [
                    429,
                    500,
                    502,
                    503,
                    504
                ]
            ):


                last_error = RuntimeError(

                    f"API 상태코드 "
                    f"{response.status_code}"

                )


                time.sleep(

                    1.5
                    * (attempt + 1)

                )


                continue


            # =================================================
            # 인증 등의 오류
            # =================================================

            raise RuntimeError(

                f"Naver API 오류\n"

                f"HTTP 상태코드: "
                f"{response.status_code}\n"

                f"응답: "
                f"{response.text[:300]}"

            )


        except Exception as e:


            last_error = e


            if (
                attempt < 2
            ):

                time.sleep(

                    1.0
                    * (attempt + 1)

                )


    raise RuntimeError(

        str(
            last_error
        )

    )


# ============================================================
# 10. 인계점 한 곳의 도로시간 계산
# ============================================================

def calculate_rp_route(

    accident_lat,
    accident_lon,

    rp_row

):


    route = naver_route(

        accident_lat,
        accident_lon,

        float(
            rp_row[
                "위도"
            ]
        ),

        float(
            rp_row[
                "경도"
            ]
        )

    )


    return {

        "인계점ID":
            rp_row[
                "인계점ID"
            ],

        "인계점명":
            rp_row[
                "인계점명"
            ],

        "시군명":
            rp_row[
                "시군명"
            ],

        "위도":
            float(
                rp_row[
                    "위도"
                ]
            ),

        "경도":
            float(
                rp_row[
                    "경도"
                ]
            ),

        "도로시간_분":
            route[
                "time_min"
            ],

        "도로거리_km":
            route[
                "distance_km"
            ]

    }


# ============================================================
# 11. 사고 위치 전체 분석
# ============================================================

def analyze_location(

    accident_lat,
    accident_lon,

    helicopter_speed,

    rp_master

):


    # ========================================================
    # A.
    # 사고 → 아주대학교병원
    # 구급차 직접이송
    # ========================================================

    direct_route = naver_route(

        accident_lat,
        accident_lon,

        HOSPITAL_LAT,
        HOSPITAL_LON

    )


    ambulance_time = (

        direct_route[
            "time_min"
        ]

    )


    # ========================================================
    # B.
    # 사고 → 모든 인계점
    # ========================================================

    progress_bar = st.progress(
        0
    )

    status_text = st.empty()


    successful = []

    failed = []


    # 너무 많은 동시 API 호출 방지
    MAX_WORKERS = 4


    with ThreadPoolExecutor(

        max_workers=MAX_WORKERS

    ) as executor:


        future_map = {}


        for (
            index,
            row
        ) in rp_master.iterrows():


            future = executor.submit(

                calculate_rp_route,

                accident_lat,
                accident_lon,

                row

            )


            future_map[
                future
            ] = {

                "index":
                    index,

                "인계점명":
                    row[
                        "인계점명"
                    ]

            }


        total = len(
            future_map
        )


        completed = 0


        for future in as_completed(
            future_map
        ):


            completed += 1


            try:


                successful.append(

                    future.result()

                )


            except Exception as e:


                failed.append({

                    "인계점":
                        future_map[
                            future
                        ][
                            "인계점명"
                        ],

                    "오류":
                        str(e)

                })


            progress_bar.progress(

                completed
                / total

            )


            status_text.text(

                f"실시간 도로경로 계산 중 "
                f"{completed}/{total}"

            )


    progress_bar.empty()

    status_text.empty()


    if (
        len(successful)
        == 0
    ):

        raise RuntimeError(

            "모든 인계점 도로경로 계산에 실패했습니다."

        )


    rp_result = pd.DataFrame(

        successful

    )


    # ========================================================
    # C.
    # 출동시간 Monte Carlo
    # ========================================================

    dispatch_samples = (

        generate_dispatch_samples()

    )


    evaluations = []


    # ========================================================
    # D.
    # 인계점별 HEMS 시간
    # ========================================================

    for (
        _,
        rp
    ) in rp_result.iterrows():


        # ----------------------------------------------------
        # 아주대병원 ↔ 인계점 직선거리
        # ----------------------------------------------------

        flight_distance = (

            haversine_distance(

                HOSPITAL_LAT,
                HOSPITAL_LON,

                float(
                    rp[
                        "위도"
                    ]
                ),

                float(
                    rp[
                        "경도"
                    ]
                )

            )

        )


        # ----------------------------------------------------
        # 편도 헬기 비행시간
        # ----------------------------------------------------

        flight_time = (

            flight_distance

            / helicopter_speed

            * 60

        )


        # ----------------------------------------------------
        # 헬기 인계점 도착시간
        #
        # 결정→이륙시간
        # +
        # 비행시간
        # ----------------------------------------------------

        helicopter_arrival = (

            dispatch_samples

            + flight_time

        )


        # ----------------------------------------------------
        # 구급차와 헬기 중 늦게 도착한 시간
        # ----------------------------------------------------

        rendezvous_time = np.maximum(

            float(
                rp[
                    "도로시간_분"
                ]
            ),

            helicopter_arrival

        )


        # ----------------------------------------------------
        # HEMS 총 이송시간
        #
        # max(
        #   사고→인계점 구급차시간,
        #   출동시간 + 헬기비행시간
        # )
        #
        # + 인계점 체류시간
        # + 병원 복귀 비행시간
        # ----------------------------------------------------

        hems_samples = (

            rendezvous_time

            + HANDOVER_TIME

            + flight_time

        )


        # ====================================================
        # 통계량
        # ====================================================

        hems_mean = float(

            hems_samples.mean()

        )


        hems_std = float(

            hems_samples.std(
                ddof=1
            )

        )


        hems_median = float(

            np.median(
                hems_samples
            )

        )


        hems_p90 = float(

            np.percentile(
                hems_samples,
                90
            )

        )


        hems_p95 = float(

            np.percentile(
                hems_samples,
                95
            )

        )


        hems_probability = float(

            (

                hems_samples

                < ambulance_time

            ).mean()

            * 100

        )


        evaluations.append({


            "인계점ID":
                rp[
                    "인계점ID"
                ],

            "인계점명":
                rp[
                    "인계점명"
                ],

            "시군명":
                rp[
                    "시군명"
                ],

            "위도":
                rp[
                    "위도"
                ],

            "경도":
                rp[
                    "경도"
                ],


            "사고→인계점_구급차시간_분":
                float(
                    rp[
                        "도로시간_분"
                    ]
                ),


            "사고→인계점_도로거리_km":
                float(
                    rp[
                        "도로거리_km"
                    ]
                ),


            "헬기직선거리_km":
                flight_distance,


            "편도헬기시간_분":
                flight_time,


            "HEMS평균시간_분":
                hems_mean,


            "HEMS중앙값_분":
                hems_median,


            "HEMS표준편차_분":
                hems_std,


            "HEMS90분위수_분":
                hems_p90,


            "HEMS95분위수_분":
                hems_p95,


            "HEMS우위확률_%":
                hems_probability

        })


    evaluation_df = pd.DataFrame(

        evaluations

    )


    # ========================================================
    # E.
    # 기대시간이 가장 짧은 인계점 선택
    # ========================================================

    best_index = (

        evaluation_df[
            "HEMS평균시간_분"
        ]

        .idxmin()

    )


    best = (

        evaluation_df.loc[
            best_index
        ]

    )


    best_hems_time = float(

        best[
            "HEMS평균시간_분"
        ]

    )


    # ========================================================
    # F.
    # 구급차시간 - HEMS시간
    #
    # 양수 = HEMS 유리
    # 음수 = 구급차 유리
    # ========================================================

    time_difference = (

        ambulance_time

        - best_hems_time

    )


    # ========================================================
    # G.
    # 최종 의사결정
    #
    # 기대시간 최소 기준
    # ========================================================

    if (

        best_hems_time

        < ambulance_time

    ):

        recommendation = (
            "닥터헬기"
        )

    else:

        recommendation = (
            "구급차"
        )


    system_time = min(

        ambulance_time,

        best_hems_time

    )


    # ========================================================
    # H.
    # 결과 반환
    # ========================================================

    return {

        "accident_lat":
            accident_lat,

        "accident_lon":
            accident_lon,


        "ambulance_time":
            ambulance_time,

        "ambulance_distance":
            direct_route[
                "distance_km"
            ],


        "api_time":
            direct_route[
                "query_time"
            ],


        "recommendation":
            recommendation,


        "system_time":
            system_time,


        "time_difference":
            time_difference,


        "best_rp":
            best.to_dict(),


        "evaluation":
            evaluation_df,


        "failed":
            failed,


        "failed_count":
            len(failed),


        "helicopter_speed":
            helicopter_speed

    }


# ============================================================
# 12. 앱 제목
# ============================================================

st.title(

    "🚁 경기도 닥터헬기 이송수단 의사결정 시스템"

)


st.write(

    """
지도에서 사고가 발생했다고 가정할 위치를 클릭하세요.

현재 도로상황과 닥터헬기 출동시간의 불확실성을 반영하여
**구급차 직접이송과 닥터헬기 연계이송의 예상 병원도착시간을 비교합니다.**
"""

)


st.info(

    "연구용 프로토타입입니다. "
    "실제 의료·출동 판단을 위한 공식 시스템이 아닙니다."

)


# ============================================================
# 13. API Key 확인
# ============================================================

if (

    not NAVER_KEY_ID

    or

    not NAVER_KEY

):


    st.error(

        """
Naver Maps API Key가 설정되어 있지 않습니다.

프로젝트 폴더의

.streamlit/secrets.toml

파일에 다음 두 값을 설정해주세요.

NAVER_MAPS_KEY_ID
NAVER_MAPS_KEY
"""

    )


    st.stop()


# ============================================================
# 14. 인계점 데이터 읽기
# ============================================================

try:


    (
        rp_master,
        rp_file_name,
        rp_encoding

    ) = load_rendezvous_points()


except Exception as e:


    st.error(

        f"인계점 데이터 오류:\n\n{e}"

    )


    st.stop()


# ============================================================
# 15. Sidebar
# ============================================================

st.sidebar.header(

    "분석 설정"

)


speed_option = st.sidebar.selectbox(

    "헬기 운항속도",

    [

        "기준 100% (267.0 km/h)",

        "90% (240.3 km/h)",

        "80% (213.6 km/h)"

    ]

)


speed_dict = {

    "기준 100% (267.0 km/h)":
        267.0,

    "90% (240.3 km/h)":
        240.3,

    "80% (213.6 km/h)":
        213.6

}


helicopter_speed = (

    speed_dict[
        speed_option
    ]

)


st.sidebar.write(

    f"인계점 후보: "
    f"**{len(rp_master)}개**"

)


st.sidebar.write(

    f"Monte Carlo: "
    f"**{N_SIM:,}회**"

)


st.sidebar.write(

    f"인계점 체류시간: "
    f"**{HANDOVER_TIME:.0f}분**"

)


st.sidebar.write(

    f"Random Seed: "
    f"**{RANDOM_SEED}**"

)


st.sidebar.divider()


st.sidebar.caption(

    f"인계점 데이터:\n"
    f"{rp_file_name}"

)


st.sidebar.caption(

    f"CSV 인코딩: "
    f"{rp_encoding}"

)


# ============================================================
# 16. 속도를 변경하면 기존 결과 초기화
# ============================================================

if (
    "last_speed"
    not in st.session_state
):

    st.session_state[
        "last_speed"
    ] = helicopter_speed


if (

    st.session_state[
        "last_speed"
    ]

    != helicopter_speed

):


    st.session_state[
        "last_speed"
    ] = helicopter_speed


    st.session_state.pop(

        "analysis_result",

        None

    )


# ============================================================
# 17. 사고 위치 저장
# ============================================================

if (
    "selected_point"
    not in st.session_state
):

    st.session_state[
        "selected_point"
    ] = None


# ============================================================
# 18. 사고 위치 선택 지도
# ============================================================

st.subheader(

    "① 사고 발생 위치 선택"

)


select_map = folium.Map(

    location=[
        37.40,
        127.05
    ],

    zoom_start=9,

    tiles="OpenStreetMap"

)


# 아주대학교병원

folium.Marker(

    [
        HOSPITAL_LAT,
        HOSPITAL_LON
    ],

    tooltip="아주대학교병원",

    popup="아주대학교병원",

    icon=folium.Icon(

        color="darkred",

        icon="plus-sign"

    )

).add_to(
    select_map
)


# 이미 선택한 사고 위치가 있을 경우

if (

    st.session_state[
        "selected_point"
    ]

    is not None

):


    (
        selected_lat,
        selected_lon

    ) = (

        st.session_state[
            "selected_point"
        ]

    )


    folium.Marker(

        [
            selected_lat,
            selected_lon
        ],

        tooltip="선택한 사고 위치",

        popup=(

            f"사고 위치<br>"

            f"위도: "
            f"{selected_lat:.6f}<br>"

            f"경도: "
            f"{selected_lon:.6f}"

        ),

        icon=folium.Icon(

            color="red"

        )

    ).add_to(
        select_map
    )


map_data = st_folium(

    select_map,

    height=520,

    use_container_width=True,

    key="location_map"

)


# ============================================================
# 19. 지도 클릭 좌표
# ============================================================

clicked = (

    map_data.get(
        "last_clicked"
    )

)


if (
    clicked is not None
):


    new_point = (

        round(
            clicked[
                "lat"
            ],
            6
        ),

        round(
            clicked[
                "lng"
            ],
            6
        )

    )


    if (

        new_point

        !=

        st.session_state[
            "selected_point"
        ]

    ):


        st.session_state[
            "selected_point"
        ] = new_point


        # 위치가 바뀌면
        # 이전 분석결과 제거

        st.session_state.pop(

            "analysis_result",

            None

        )


        st.rerun()


# ============================================================
# 20. 선택 위치 표시
# ============================================================

if (

    st.session_state[
        "selected_point"
    ]

    is not None

):


    lat, lon = (

        st.session_state[
            "selected_point"
        ]

    )


    st.success(

        f"선택 위치: "
        f"위도 {lat:.6f}, "
        f"경도 {lon:.6f}"

    )


    st.caption(

        f"분석 버튼을 누르면 "
        f"구급차 직접경로 1개와 "
        f"인계점 {len(rp_master)}개 경로를 "
        f"실시간으로 조회합니다."

    )


    # ========================================================
    # 분석 시작
    # ========================================================

    if st.button(

        "🚁 구급차 vs 닥터헬기 분석 시작",

        type="primary",

        use_container_width=True

    ):


        try:


            with st.spinner(

                "실시간 도로정보와 "
                "닥터헬기 이송시간을 계산하고 있습니다..."

            ):


                analysis_result = (

                    analyze_location(

                        lat,
                        lon,

                        helicopter_speed,

                        rp_master

                    )

                )


                st.session_state[
                    "analysis_result"
                ] = analysis_result


        except Exception as e:


            st.error(

                f"분석 중 오류가 발생했습니다.\n\n{e}"

            )


else:


    st.warning(

        "지도에서 사고 위치를 클릭해주세요."

    )


# ============================================================
# 21. 분석 결과
# ============================================================

if (

    "analysis_result"
    in st.session_state

):


    analysis_result = (

        st.session_state[
            "analysis_result"
        ]

    )


    best = (

        analysis_result[
            "best_rp"
        ]

    )


    st.divider()


    st.header(

        "② 분석 결과"

    )


    # ========================================================
    # 추천수단
    # ========================================================

    if (

        analysis_result[
            "recommendation"
        ]

        == "닥터헬기"

    ):


        st.success(

            "🚁 추천 이송수단: 닥터헬기"

        )


    else:


        st.info(

            "🚑 추천 이송수단: 구급차"

        )


    # ========================================================
    # 핵심 결과 카드
    # ========================================================

    col1, col2, col3, col4 = st.columns(
        4
    )


    col1.metric(

        "구급차 직접이송",

        (
            f"{analysis_result['ambulance_time']:.1f}분"
        )

    )


    col2.metric(

        "HEMS 기대시간",

        (
            f"{best['HEMS평균시간_분']:.1f}분"
        )

    )


    col3.metric(

        "HEMS 시간 우위",

        (
            f"{analysis_result['time_difference']:.1f}분"
        ),

        help=(

            "구급차 직접이송시간에서 "
            "HEMS 평균시간을 뺀 값입니다. "
            "양수이면 HEMS가 더 빠릅니다."

        )

    )


    col4.metric(

        "HEMS 우위확률",

        (
            f"{best['HEMS우위확률_%']:.1f}%"
        ),

        help=(

            "Monte Carlo 10,000회 중 "
            "HEMS가 구급차보다 빨랐던 비율입니다."

        )

    )


    # ========================================================
    # 의사결정 해석
    # ========================================================

    st.subheader(

        "의사결정 해석"

    )


    if (

        analysis_result[
            "recommendation"
        ]

        == "닥터헬기"

    ):


        st.write(

            f"""
구급차로 사고지점에서 아주대학교병원까지 직접 이송할 경우
예상시간은 **{analysis_result['ambulance_time']:.1f}분**입니다.

181개 인계점을 평가한 결과,
기대 HEMS 이송시간이 가장 짧은 인계점은
**{best['인계점명']}**으로 계산되었습니다.

해당 인계점을 이용한 HEMS 기대시간은
**{best['HEMS평균시간_분']:.1f}분**입니다.

따라서 닥터헬기 연계이송은 구급차 직접이송보다
평균적으로 **{analysis_result['time_difference']:.1f}분 빠른 것으로 계산**되었습니다.

Monte Carlo {N_SIM:,}회에서
HEMS가 구급차보다 빠른 확률은
**{best['HEMS우위확률_%']:.1f}%**입니다.
"""

        )


    else:


        ambulance_advantage = (

            abs(
                analysis_result[
                    "time_difference"
                ]
            )

        )


        st.write(

            f"""
구급차로 사고지점에서 아주대학교병원까지 직접 이송할 경우
예상시간은 **{analysis_result['ambulance_time']:.1f}분**입니다.

181개 인계점 가운데 가장 빠른 HEMS 경로는
**{best['인계점명']}**을 이용하는 경우이며,
HEMS 기대시간은
**{best['HEMS평균시간_분']:.1f}분**입니다.

따라서 기대시간 기준으로는 구급차 직접이송이
약 **{ambulance_advantage:.1f}분 빠른 것으로 계산**되어
구급차를 추천합니다.

참고로 해당 최적 HEMS 경로가 실제로 구급차보다 빠른
Monte Carlo 확률은
**{best['HEMS우위확률_%']:.1f}%**입니다.
"""

        )


    # ========================================================
    # 최적 인계점
    # ========================================================

    st.subheader(

        "③ 최적 인계점"

    )


    c1, c2, c3, c4 = st.columns(
        4
    )


    c1.metric(

        "최적 인계점",

        str(
            best[
                "인계점명"
            ]
        )

    )


    c2.metric(

        "사고 → 인계점",

        (
            f"{best['사고→인계점_구급차시간_분']:.1f}분"
        )

    )


    c3.metric(

        "편도 헬기 비행",

        (
            f"{best['편도헬기시간_분']:.1f}분"
        )

    )


    c4.metric(

        "HEMS 95% 분위수",

        (
            f"{best['HEMS95분위수_분']:.1f}분"
        )

    )


    # ========================================================
    # 일부 인계점 API 실패 경고
    # ========================================================

    if (

        analysis_result[
            "failed_count"
        ]

        > 0

    ):


        st.warning(

            f"인계점 "
            f"{analysis_result['failed_count']}개의 "
            f"도로경로 조회에 실패했습니다.\n\n"

            "현재 결과는 조회에 성공한 인계점만을 대상으로 "
            "산출한 결과입니다."

        )


    # ========================================================
    # 결과 지도
    # ========================================================

    st.subheader(

        "④ 이송 경로 개념도"

    )


    accident_lat = (

        analysis_result[
            "accident_lat"
        ]

    )

    accident_lon = (

        analysis_result[
            "accident_lon"
        ]

    )


    result_map = folium.Map(

        location=[
            accident_lat,
            accident_lon
        ],

        zoom_start=10,

        tiles="OpenStreetMap"

    )


    # 사고지점

    folium.Marker(

        [
            accident_lat,
            accident_lon
        ],

        tooltip="사고지점",

        popup="사고지점",

        icon=folium.Icon(

            color="red",

            icon="info-sign"

        )

    ).add_to(
        result_map
    )


    # 최적 인계점

    folium.Marker(

        [
            best[
                "위도"
            ],

            best[
                "경도"
            ]
        ],

        tooltip=(

            "최적 인계점: "

            + str(
                best[
                    "인계점명"
                ]
            )

        ),

        popup=(

            f"<b>최적 인계점</b><br>"
            f"{best['인계점명']}<br><br>"

            f"사고→인계점: "
            f"{best['사고→인계점_구급차시간_분']:.1f}분<br>"

            f"편도 헬기: "
            f"{best['편도헬기시간_분']:.1f}분"

        ),

        icon=folium.Icon(

            color="green",

            icon="flag"

        )

    ).add_to(
        result_map
    )


    # 아주대학교병원

    folium.Marker(

        [
            HOSPITAL_LAT,
            HOSPITAL_LON
        ],

        tooltip=HOSPITAL_NAME,

        popup=HOSPITAL_NAME,

        icon=folium.Icon(

            color="darkred",

            icon="plus-sign"

        )

    ).add_to(
        result_map
    )


    # --------------------------------------------------------
    # 사고 → 인계점
    #
    # 실제 도로선이 아니라 위치관계 개념선
    # --------------------------------------------------------

    folium.PolyLine(

        [

            [
                accident_lat,
                accident_lon
            ],

            [
                best[
                    "위도"
                ],
                best[
                    "경도"
                ]
            ]

        ],

        tooltip=(

            "사고지점 → 인계점 "
            "(구급차)"

        ),

        weight=4

    ).add_to(
        result_map
    )


    # --------------------------------------------------------
    # 인계점 → 아주대학교병원
    # 헬기 비행 개념선
    # --------------------------------------------------------

    folium.PolyLine(

        [

            [
                best[
                    "위도"
                ],
                best[
                    "경도"
                ]
            ],

            [
                HOSPITAL_LAT,
                HOSPITAL_LON
            ]

        ],

        tooltip=(

            "인계점 → 아주대학교병원 "
            "(닥터헬기)"

        ),

        weight=4,

        dash_array="8"

    ).add_to(
        result_map
    )


    # 지도 범위

    result_map.fit_bounds(

        [

            [

                min(

                    accident_lat,

                    float(
                        best[
                            "위도"
                        ]
                    ),

                    HOSPITAL_LAT

                ),

                min(

                    accident_lon,

                    float(
                        best[
                            "경도"
                        ]
                    ),

                    HOSPITAL_LON

                )

            ],

            [

                max(

                    accident_lat,

                    float(
                        best[
                            "위도"
                        ]
                    ),

                    HOSPITAL_LAT

                ),

                max(

                    accident_lon,

                    float(
                        best[
                            "경도"
                        ]
                    ),

                    HOSPITAL_LON

                )

            ]

        ]

    )


    st_folium(

        result_map,

        height=520,

        use_container_width=True,

        key="result_map"

    )


    st.caption(

        "※ 지도 위의 선은 경로의 위치관계를 보여주기 위한 "
        "개념선입니다. 구급차 이동시간 자체는 "
        "Naver Directions API의 실제 도로경로 결과를 사용합니다."

    )


    # ========================================================
    # 인계점 TOP 10
    # ========================================================

    with st.expander(

        "⑤ 인계점별 분석 결과 TOP 10"

    ):


        top10 = (

            analysis_result[
                "evaluation"
            ]

            .sort_values(

                "HEMS평균시간_분"

            )

            .head(10)

            [

                [
                    "인계점명",
                    "시군명",
                    "사고→인계점_구급차시간_분",
                    "편도헬기시간_분",
                    "HEMS평균시간_분",
                    "HEMS표준편차_분",
                    "HEMS우위확률_%"
                ]

            ]

            .copy()

        )


        st.dataframe(

            top10.round(
                2
            ),

            use_container_width=True,

            hide_index=True

        )


    # ========================================================
    # 전체 인계점 결과 다운로드
    # ========================================================

    csv_data = (

        analysis_result[
            "evaluation"
        ]

        .to_csv(
            index=False
        )

        .encode(
            "utf-8-sig"
        )

    )


    st.download_button(

        "전체 인계점 분석 결과 CSV 다운로드",

        data=csv_data,

        file_name=(

            "닥터헬기_실시간_인계점분석.csv"

        ),

        mime="text/csv"

    )


    # ========================================================
    # API 조회 정보
    # ========================================================

    st.caption(

        f"도로정보 조회시각: "
        f"{analysis_result['api_time']}"

        f" | 헬기속도: "
        f"{analysis_result['helicopter_speed']:.1f} km/h"

        f" | Monte Carlo: "
        f"{N_SIM:,}회"

        f" | 인계점 경로 실패: "
        f"{analysis_result['failed_count']}개"

    )


# ============================================================
# 22. 연구모형 설명
# ============================================================

st.divider()


with st.expander(

    "시스템 계산방법 보기"

):


    st.markdown(

        """
### 구급차 직접이송

사고지점에서 아주대학교병원까지의 도로이동시간을
Naver Directions API를 이용하여 계산합니다.

---

### 닥터헬기 연계이송

각 인계점에 대해 다음 식으로 계산합니다.

\[
T_{HEMS,j}
=
\max
(
T_{road,j},
D + T_{flight,j}
)
+
4
+
T_{flight,j}
\]

- \(T_{road,j}\): 사고지점 → 인계점 구급차 이동시간
- \(D\): 닥터헬기 결정→이륙시간
- \(T_{flight,j}\): 아주대학교병원 ↔ 인계점 편도 비행시간
- 4분: 인계점 체류시간

구급차와 헬기는 동시에 인계점으로 이동한다고 가정하며,
먼저 도착한 수단은 상대 수단을 기다립니다.

181개 인계점 중 기대 HEMS 시간이 최소인 인계점을 선택하고,
구급차 직접이송시간과 비교하여 기대 병원도착시간이 더 짧은
수단을 추천합니다.

또한 Monte Carlo 10,000회를 통해

\[
P(T_{HEMS}<T_{AMB})
\]

를 계산하여 HEMS 우위확률로 제공합니다.
"""

    )


# ============================================================
# 23. 하단 주의사항
# ============================================================

st.divider()


st.caption(

    """
※ 본 시스템은 연구 결과의 시연을 위한 프로토타입입니다.

Naver Directions 자동차 경로를 기반으로 하므로 실제 구급차의
긴급주행 및 신호우선 효과를 직접 반영하지 않습니다.

또한 실제 기상조건, 야간운항 여부, 헬기 운항가능 여부,
인계점의 실시간 운영가능 여부는 포함하지 않습니다.

실제 환자 이송수단을 결정하기 위한 공식 의료시스템이 아닙니다.
"""

)