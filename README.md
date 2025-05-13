# 자율 주행 V2X 통신 지연 프로젝트
# 필요한 패키지 로드
library(tidyverse)
library(randomForest)
library(arules)

# 데이터 불러오기 및 확인
unzip(zipfile = "C:/Users/jungh/archive.zip", exdir = "C:/Users/jungh")
data <- read.csv("C:/Users/jungh/vehicle_network_dataset.csv")

# 요약 및 결측 확인
summary(data)
colSums(is.na(data))

# Latency 분포 시각화
ggplot(data, aes(x = Latency..ms.)) +
  geom_histogram(binwidth = 5, fill = "skyblue", color = "black") +
  labs(title = "Latency 분포", x = "Latency (ms)", y = "빈도") +
  theme_minimal()

# 속도 vs 지연시간 산점도
ggplot(data, aes(x = Speed..km.h., y = Latency..ms.)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "속도와 지연시간의 관계", x = "Speed (km/h)", y = "Latency (ms)") +
  theme_minimal()

# 상관관계 분석
cor(data$Speed..km.h., data$Latency..ms., method = "pearson")

# 군집화 수행
clust_data <- data %>% select(Latency..ms., RSRP = Signal_Strength..dBm., SINR = Network_Stability_Index)
k <- 3
km <- kmeans(clust_data, centers = k)
data$cluster <- factor(km$cluster)

# 군집 시각화
ggplot(data, aes(x = Signal_Strength..dBm., y = Latency..ms., color = cluster)) +
  geom_point(alpha = 0.5) +
  labs(title = "K-means 클러스터링 (k=3)", x = "RSRP", y = "Latency") +
  theme_minimal()

# 평균 latency 이하 클러스터만 필터링 → 건강한 클러스터
healthy_clusters <- names(which(tapply(data$Latency..ms., data$cluster, mean) < mean(data$Latency..ms.)))
data_clean <- data %>% filter(cluster %in% healthy_clusters)

# delay_group 생성 (기준: 60ms 이상 High)
data_clean$delay_group <- ifelse(data_clean$Latency..ms. > 60, "High", "Normal")
table(data_clean$delay_group)

# 분류 예측 모델 학습
model <- randomForest(
  factor(delay_group) ~ Speed..km.h. + Signal_Strength..dBm. + Network_Stability_Index,
  data = data_clean,
  ntree = 100
)
print(model)

# 예측 및 평가
pred <- predict(model, newdata = data_clean)
conf_mat <- table(Predicted = pred, Actual = data_clean$delay_group)
print(conf_mat)

# 정확도 계산
accuracy <- sum(diag(conf_mat)) / sum(conf_mat)
print(paste("정확도:", round(accuracy * 100, 2), "%"))

# 연관 규칙 분석
trans <- as(
  data_clean %>%
    mutate(
      speed_level = factor(ifelse(Speed..km.h. > 50, "Fast", "Slow")),
      latency_level = factor(ifelse(Latency..ms. > 100, "Delayed", "Normal")),
      Scheduling_Algorithm = factor(Scheduling_Algorithm)
    ) %>% select(speed_level, latency_level, Scheduling_Algorithm),
  "transactions"
)

rules <- apriori(trans, parameter = list(supp = 0.01, conf = 0.6))
inspect(rules[1:5])
