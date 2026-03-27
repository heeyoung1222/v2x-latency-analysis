# V2X Communication Latency Analysis

자율주행 환경에서 발생하는 **V2X 통신 지연(latency)** 을 데이터 기반으로 탐색하고, 지연 패턴을 분류 및 해석하기 위한 R 기반 분석 프로젝트입니다.

## Overview

이 프로젝트는 차량 속도, 신호 세기, 네트워크 안정성 등의 변수와 통신 지연 시간의 관계를 확인하고,
이상적인 통신 상태와 지연 위험 상태를 구분하기 위한 기초 분석을 수행합니다.

주요 목표는 다음과 같습니다.

- V2X 환경에서 latency 분포와 기본 특성을 파악하기
- 속도, 신호 세기, 네트워크 안정성과 latency의 관계를 확인하기
- K-means clustering으로 유사한 통신 상태를 그룹화하기
- Random Forest로 지연 위험 구간을 분류하기
- Apriori association rule mining으로 조건 간 패턴을 해석하기

## Tech Stack

- **Language**: R
- **Libraries**: `tidyverse`, `randomForest`, `arules`
- **Analysis Topics**: EDA, correlation analysis, clustering, classification, association rules

## Analysis Workflow

1. 데이터 압축 해제 및 CSV 로드
2. 결측치 및 기초 통계 확인
3. Latency 분포 시각화
4. 차량 속도와 latency 간 관계 분석
5. K-means 기반 네트워크 상태 군집화
6. 평균 latency 기준으로 상대적으로 안정적인 클러스터 선별
7. `delay_group` 생성 후 Random Forest 분류 수행
8. Association Rule 기반 패턴 탐색

## Key Analysis Components

### 1. Exploratory Data Analysis

- `Latency (ms)` 분포를 히스토그램으로 시각화
- 속도와 지연 시간의 관계를 산점도 및 회귀선으로 확인
- 결측치 및 기술통계를 통해 데이터 품질 점검

### 2. Correlation Check

- `Speed` 와 `Latency` 간 Pearson 상관계수를 계산하여
  선형 관계의 방향성과 강도를 확인

### 3. Clustering

다음 변수를 기반으로 K-means clustering을 수행합니다.

- `Latency (ms)`
- `Signal Strength (RSRP)`
- `Network Stability Index (SINR)`

이를 통해 유사한 통신 품질 상태를 묶고, 평균 latency가 낮은 클러스터를
상대적으로 안정적인 상태로 해석합니다.

### 4. Delay Classification

Random Forest를 사용하여 아래 변수를 기반으로 지연 수준을 분류합니다.

- `Speed`
- `Signal Strength`
- `Network Stability Index`

`Latency > 60 ms` 인 경우 `High`, 그 외는 `Normal`로 정의했습니다.

### 5. Association Rule Mining

속도 수준, latency 수준, 스케줄링 알고리즘 간 조합을 transaction 형태로 변환한 뒤,
Apriori 알고리즘을 적용하여 의미 있는 규칙을 탐색합니다.

## Repository Note

현재 저장소는 분석 전 과정을 정리한 **README 중심의 요약형 프로젝트 기록** 형태입니다.
추후 아래 항목을 추가하면 더 완성도 높은 형태로 확장할 수 있습니다.

- 실제 R 스크립트 파일 분리
- 데이터 전처리 코드와 모델링 코드 모듈화
- 시각화 결과 이미지 첨부
- 데이터셋 설명 및 컬럼 정의 문서화

## Example Packages

```r
library(tidyverse)
library(randomForest)
library(arules)
```

## Future Improvements

- 재현 가능한 실행 환경 정리
- 모델 성능 비교 실험 추가
- 특성 중요도 시각화
- 클러스터별 해석 강화
- README에 결과 이미지 및 표 추가

## Author

Heeyoung Jeong

관심 분야: Intelligent Transportation Systems, V2X, Traffic Safety, Data Analysis