set.seed(1222)

root_dir <- normalizePath(file.path(getwd()), winslash = "/", mustWork = TRUE)
data_dir <- file.path(root_dir, "data")
results_dir <- file.path(root_dir, "results")

if (!dir.exists(data_dir)) dir.create(data_dir, recursive = TRUE)
if (!dir.exists(results_dir)) dir.create(results_dir, recursive = TRUE)

n <- 1500

speed_kmh <- round(rnorm(n, mean = 68, sd = 18), 1)
speed_kmh <- pmax(speed_kmh, 0)

signal_strength_dbm <- round(rnorm(n, mean = -83, sd = 11), 1)
signal_strength_dbm <- pmin(pmax(signal_strength_dbm, -120), -55)

network_stability_index <- round(runif(n, min = 0.35, max = 0.99), 3)
vehicle_density <- sample(c("Low", "Medium", "High"), n, replace = TRUE, prob = c(0.28, 0.47, 0.25))
scheduling_algorithm <- sample(
  c("RoundRobin", "WeightedFair", "PriorityAware"),
  n,
  replace = TRUE,
  prob = c(0.34, 0.32, 0.34)
)

density_effect <- ifelse(vehicle_density == "High", 15, ifelse(vehicle_density == "Medium", 7, 1))
algorithm_effect <- ifelse(
  scheduling_algorithm == "PriorityAware",
  -5,
  ifelse(scheduling_algorithm == "WeightedFair", 1.5, 5)
)

latency_ms <- 42 +
  0.20 * speed_kmh +
  0.32 * abs(signal_strength_dbm + 75) +
  26 * (1 - network_stability_index) +
  density_effect +
  algorithm_effect +
  rnorm(n, mean = 0, sd = 8)

latency_ms <- round(pmax(latency_ms, 8), 1)
high_latency <- ifelse(latency_ms >= 80, "High", "Normal")

dataset <- data.frame(
  speed_kmh = speed_kmh,
  signal_strength_dbm = signal_strength_dbm,
  network_stability_index = network_stability_index,
  vehicle_density = factor(vehicle_density, levels = c("Low", "Medium", "High")),
  scheduling_algorithm = factor(
    scheduling_algorithm,
    levels = c("RoundRobin", "WeightedFair", "PriorityAware")
  ),
  latency_ms = latency_ms,
  high_latency = factor(high_latency, levels = c("Normal", "High"))
)

write.csv(dataset, file.path(data_dir, "simulated_v2x_dataset.csv"), row.names = FALSE)

summary_metrics <- data.frame(
  metric = c(
    "sample_size",
    "mean_latency_ms",
    "median_latency_ms",
    "sd_latency_ms",
    "high_latency_share",
    "mean_speed_kmh",
    "mean_signal_strength_dbm",
    "mean_network_stability_index"
  ),
  value = c(
    nrow(dataset),
    round(mean(dataset$latency_ms), 2),
    round(median(dataset$latency_ms), 2),
    round(sd(dataset$latency_ms), 2),
    round(mean(dataset$high_latency == "High"), 4),
    round(mean(dataset$speed_kmh), 2),
    round(mean(dataset$signal_strength_dbm), 2),
    round(mean(dataset$network_stability_index), 3)
  )
)

write.csv(summary_metrics, file.path(results_dir, "summary_metrics.csv"), row.names = FALSE)

png(file.path(results_dir, "latency_distribution.png"), width = 1200, height = 800, res = 140)
hist(
  dataset$latency_ms,
  breaks = 30,
  col = "#6BAED6",
  border = "white",
  main = "Distribution of Simulated V2X Latency",
  xlab = "Latency (ms)"
)
abline(v = mean(dataset$latency_ms), col = "#CB181D", lwd = 3)
legend(
  "topright",
  legend = paste("Mean latency:", round(mean(dataset$latency_ms), 1), "ms"),
  bty = "n"
)
dev.off()

png(file.path(results_dir, "speed_vs_latency.png"), width = 1200, height = 800, res = 140)
plot(
  dataset$speed_kmh,
  dataset$latency_ms,
  pch = 19,
  cex = 0.6,
  col = rgb(44, 127, 184, 110, maxColorValue = 255),
  xlab = "Vehicle Speed (km/h)",
  ylab = "Latency (ms)",
  main = "Speed vs. V2X Latency"
)
abline(lm(latency_ms ~ speed_kmh, data = dataset), col = "#CB181D", lwd = 3)
dev.off()

cluster_input <- scale(dataset[, c("latency_ms", "signal_strength_dbm", "network_stability_index")])
kmeans_result <- kmeans(cluster_input, centers = 3, nstart = 20)
dataset$cluster <- factor(kmeans_result$cluster)

cluster_colors <- c("#1B9E77", "#D95F02", "#7570B3")

png(file.path(results_dir, "cluster_map.png"), width = 1200, height = 800, res = 140)
plot(
  dataset$signal_strength_dbm,
  dataset$latency_ms,
  pch = 19,
  cex = 0.65,
  col = cluster_colors[dataset$cluster],
  xlab = "Signal Strength (dBm)",
  ylab = "Latency (ms)",
  main = "K-means Cluster Map of Network States"
)
legend(
  "topright",
  legend = paste("Cluster", levels(dataset$cluster)),
  col = cluster_colors,
  pch = 19,
  bty = "n"
)
dev.off()

glm_model <- glm(
  high_latency ~ speed_kmh + signal_strength_dbm + network_stability_index +
    vehicle_density + scheduling_algorithm,
  data = dataset,
  family = binomial()
)

coeff_table <- summary(glm_model)$coefficients
coeff_df <- data.frame(
  term = rownames(coeff_table)[-1],
  estimate = coeff_table[-1, "Estimate"]
)

coeff_colors <- ifelse(coeff_df$estimate >= 0, "#CB181D", "#2171B5")

png(file.path(results_dir, "model_coefficients.png"), width = 1200, height = 800, res = 140)
par(mar = c(5, 11, 4, 2))
barplot(
  rev(coeff_df$estimate),
  horiz = TRUE,
  col = rev(coeff_colors),
  names.arg = rev(coeff_df$term),
  las = 1,
  main = "Logistic Regression Coefficients for High Latency Risk",
  xlab = "Coefficient Estimate"
)
abline(v = 0, lty = 2)
dev.off()

message("Analysis completed successfully.")
