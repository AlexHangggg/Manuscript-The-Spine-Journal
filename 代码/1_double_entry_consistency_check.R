# =============================================================================
# Double-entry consistency check script
# =============================================================================
# Purpose: Assess inter-rater consistency for double-entry data
# Date: 2026-01-19
# =============================================================================

# Load required packages
if (!require("readxl")) install.packages("readxl")
if (!require("irr")) install.packages("irr")

library(readxl)
library(irr)

# =============================================================================
# 1. Data loading
# =============================================================================

# ---------------------------------------------------------------------------
# Input file path — auto-discovery (same pattern as 5_lmm_vas_odi_joa_analysis.R)
# Directory layout assumed:
#   F:\李子航毕业论文原始数据\代码\   <- this script lives here
#   F:\李子航毕业论文原始数据\文件\   <- data files live here (sibling folder)
#
# Candidate paths tried in order:
#   1. <script_dir>/../文件/Double entry2.28.xlsx  (standard layout)
#   2. <working_dir>/文件/Double entry2.28.xlsx     (run from project root)
#   3. Double entry2.28.xlsx                        (data in working dir, fallback)
# ---------------------------------------------------------------------------
TARGET_FILE <- "Double entry2.28.xlsx"

script_args <- commandArgs(trailingOnly = FALSE)
file_arg    <- grep("^--file=", script_args, value = TRUE)
script_path <- if (length(file_arg) >= 1) {
  normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = FALSE)
} else {
  NA_character_
}
script_dir <- if (!is.na(script_path) && nzchar(script_path)) {
  dirname(script_path)
} else {
  getwd()
}

input_candidates <- c(
  file.path(script_dir, "..", "\u6587\u4ef6", TARGET_FILE),   # sibling 文件 folder (standard)
  file.path(getwd(),    "\u6587\u4ef6", TARGET_FILE),          # 文件 sub-folder of working dir
  file.path(getwd(),    TARGET_FILE)                             # working dir fallback
)
input_candidates <- normalizePath(input_candidates, winslash = "/", mustWork = FALSE)

existing_inputs <- input_candidates[file.exists(input_candidates)]
file_path <- if (length(existing_inputs) > 0) {
  existing_inputs[[1]]
} else {
  stop(
    "Input file \'", TARGET_FILE, "\' not found. Paths tried:\n",
    paste0("  ", input_candidates, collapse = "\n")
  )
}
cat("Reading data from:", file_path, "\n")

# Read data
cat("Reading data...\n")
raw_data <- read_excel(file_path)

# Show basic structure
cat("\nRaw data dimensions:", dim(raw_data), "\n")
cat("\nRaw data column names:\n")
print(colnames(raw_data))
cat("\nRaw data preview (head):\n")
print(head(raw_data))

# =============================================================================
# 2. Variable setup and parser
# =============================================================================

# Categorical variables (Cohen's Kappa)
categorical_vars <- c(
  "Pfirrmann",
  "Iwabuchi",
  "Modic",
  "Komori",
  "MSU",
  "Spinal_canal_stenosis",
  "Bull_eye"
)

# Continuous variables (ICC)
continuous_vars <- c(
  "SS",
  "Upper_VB_Posterior_Height_CM",
  "Lower_VB_Posterior_Height_CM",
  "Initial_volume",
  "DHI",
  "RSI"
)

# Parse one variable into paired Res1/Res2 columns
parse_double_entry <- function(raw_data, var_name, is_categorical = FALSE) {
  # Exact match of <var>_Res1 and <var>_Res2
  res1_pattern <- paste0("^", var_name, "_Res1$")
  res2_pattern <- paste0("^", var_name, "_Res2$")

  res1_col <- grep(res1_pattern, colnames(raw_data), value = TRUE)
  res2_col <- grep(res2_pattern, colnames(raw_data), value = TRUE)

  if (length(res1_col) == 0 || length(res2_col) == 0) {
    warning(paste("Variable", var_name, "Res1 or Res2 not found"))
    return(NULL)
  }

  # Extract data
  res1 <- raw_data[[res1_col]]
  res2 <- raw_data[[res2_col]]

  # Convert continuous variables to numeric
  if (!is_categorical) {
    res1 <- as.numeric(as.character(res1))
    res2 <- as.numeric(as.character(res2))
  }

  # Remove NA pairs
  valid_idx <- complete.cases(res1, res2)

  result <- data.frame(
    Rater1 = res1[valid_idx],
    Rater2 = res2[valid_idx]
  )

  return(result)
}

# =============================================================================
# 3. Categorical consistency (Cohen's Kappa)
# =============================================================================

cat("\n")
cat("========================================\n")
cat("Categorical consistency check (Cohen's Kappa)\n")
cat("========================================\n\n")

kappa_results <- data.frame(
  Variable = character(),
  Kappa = numeric(),
  SE = numeric(),
  Z = numeric(),
  p_value = numeric(),
  Agreement = character(),
  stringsAsFactors = FALSE
)

for (var in categorical_vars) {
  cat(sprintf("\n--- Variable: %s ---\n", var))

  data <- parse_double_entry(raw_data, var, is_categorical = TRUE)

  if (is.null(data) || nrow(data) == 0) {
    cat(sprintf("  Skip %s (no valid paired data)\n", var))
    next
  }

  # Convert to factor for categorical agreement analysis
  data$Rater1 <- as.factor(data$Rater1)
  data$Rater2 <- as.factor(data$Rater2)

  # Cross table
  tab <- table(data$Rater1, data$Rater2)
  cat("  Cross table:\n")
  print(tab)

  # Cohen's Kappa
  kappa_result <- kappa2(data[, c("Rater1", "Rater2")])

  # kappa2() does not return lower.value/upper.value; compute 95% CI manually
  kappa_ci_low  <- kappa_result$value - 1.96 * kappa_result$std.error
  kappa_ci_high <- kappa_result$value + 1.96 * kappa_result$std.error

  cat(sprintf("\n  Kappa: %.3f (SE = %.3f, 95%% CI: %.3f - %.3f)\n",
              kappa_result$value, kappa_result$std.error,
              kappa_ci_low, kappa_ci_high))
  cat(sprintf("  Z statistic: %.3f, p value: %.4f\n",
              kappa_result$statistic, kappa_result$p.value))

  # Interpret Kappa
  kappa_val <- kappa_result$value
  if (is.na(kappa_val)) {
    agreement <- "Not available"
  } else if (kappa_val < 0) {
    agreement <- "Poor (<0)"
  } else if (kappa_val < 0.20) {
    agreement <- "Slight (0-0.20)"
  } else if (kappa_val < 0.40) {
    agreement <- "Fair (0.21-0.40)"
  } else if (kappa_val < 0.60) {
    agreement <- "Moderate (0.41-0.60)"
  } else if (kappa_val < 0.80) {
    agreement <- "Substantial (0.61-0.80)"
  } else {
    agreement <- "Almost perfect (0.81-1.00)"
  }

  cat(sprintf("  Agreement level: %s\n", agreement))

  # Observed agreement
  obs_agreement <- sum(diag(tab)) / sum(tab) * 100
  cat(sprintf("  Observed agreement: %.1f%%\n", obs_agreement))

  # Safe extraction to handle NULL/NA edge cases
  se_val <- ifelse(is.null(kappa_result$std.error) || is.na(kappa_result$std.error),
                   NA, kappa_result$std.error)
  z_val <- ifelse(is.null(kappa_result$statistic) || is.na(kappa_result$statistic),
                  NA, kappa_result$statistic)
  p_val <- ifelse(is.null(kappa_result$p.value) || is.na(kappa_result$p.value),
                  NA, kappa_result$p.value)

  kappa_results <- rbind(kappa_results, data.frame(
    Variable = var,
    Kappa = kappa_val,
    SE = se_val,
    Z = z_val,
    p_value = p_val,
    Agreement = agreement
  ))
}

# =============================================================================
# 4. Continuous consistency (ICC)
# =============================================================================

cat("\n")
cat("========================================\n")
cat("Continuous consistency check (ICC)\n")
cat("========================================\n\n")

icc_results <- data.frame(
  Variable = character(),
  ICC_single = numeric(),
  ICC_single_CI_low = numeric(),
  ICC_single_CI_high = numeric(),
  ICC_average = numeric(),
  ICC_average_CI_low = numeric(),
  ICC_average_CI_high = numeric(),
  p_value = numeric(),
  Agreement = character(),
  stringsAsFactors = FALSE
)

for (var in continuous_vars) {
  cat(sprintf("\n--- Variable: %s ---\n", var))

  data <- parse_double_entry(raw_data, var)

  if (is.null(data) || nrow(data) == 0) {
    cat(sprintf("  Skip %s (no valid paired data)\n", var))
    next
  }

  # ICC using two-way consistency model
  icc_single <- icc(data[, c("Rater1", "Rater2")],
                    model = "twoway",
                    type = "consistency",
                    unit = "single")
  icc_avg <- icc(data[, c("Rater1", "Rater2")],
                 model = "twoway",
                 type = "consistency",
                 unit = "average")

  cat(sprintf("\n  ICC (single measure): %.3f (95%% CI: %.3f - %.3f)\n",
              icc_single$value,
              icc_single$`lbound`[1],
              icc_single$`ubound`[1]))
  cat(sprintf("  ICC (average measure): %.3f (95%% CI: %.3f - %.3f)\n",
              icc_avg$value,
              icc_avg$`lbound`[1],
              icc_avg$`ubound`[1]))

  # Safe extraction of F statistic
  if (is.list(icc_avg$Fvalue)) {
    f_val <- icc_avg$Fvalue$F
  } else {
    f_val <- icc_avg$Fvalue[1]
  }
  cat(sprintf("  F statistic: %.3f\n", f_val))
  cat(sprintf("  p value: %.4f\n", icc_avg$p.value))

  # Interpret ICC
  icc_val <- icc_avg$value
  if (is.null(icc_val) || length(icc_val) == 0 || is.na(icc_val)) {
    agreement <- "Not available"
  } else if (icc_val < 0.50) {
    agreement <- "Poor (<0.50)"
  } else if (icc_val < 0.75) {
    agreement <- "Moderate (0.50-0.75)"
  } else if (icc_val < 0.90) {
    agreement <- "Good (0.75-0.90)"
  } else {
    agreement <- "Excellent (0.90-1.00)"
  }

  cat(sprintf("  Agreement level: %s\n", agreement))

  # Pearson correlation
  cor_result <- cor.test(data$Rater1, data$Rater2, method = "pearson")
  cat(sprintf("  Pearson correlation: %.3f (p = %.4f)\n",
              cor_result$estimate, cor_result$p.value))

  # Safe extraction for ICC output fields
  icc_single_low <- ifelse(length(icc_single$`lbound`) >= 1, icc_single$`lbound`[1], NA)
  icc_single_high <- ifelse(length(icc_single$`ubound`) >= 1, icc_single$`ubound`[1], NA)
  icc_avg_val <- ifelse(length(icc_avg$value) > 0, icc_avg$value, NA)
  icc_avg_low <- ifelse(length(icc_avg$`lbound`) >= 1, icc_avg$`lbound`[1], NA)
  icc_avg_high <- ifelse(length(icc_avg$`ubound`) >= 1, icc_avg$`ubound`[1], NA)
  p_val <- ifelse(length(icc_avg$p.value) > 0, icc_avg$p.value, NA)

  icc_results <- rbind(icc_results, data.frame(
    Variable = var,
    ICC_single = ifelse(length(icc_single$value) > 0, icc_single$value, NA),
    ICC_single_CI_low = icc_single_low,
    ICC_single_CI_high = icc_single_high,
    ICC_average = icc_avg_val,
    ICC_average_CI_low = icc_avg_low,
    ICC_average_CI_high = icc_avg_high,
    p_value = p_val,
    Agreement = agreement
  ))
}

# =============================================================================
# 5. Console summary
# =============================================================================

cat("\n")
cat("========================================\n")
cat("Summary of results\n")
cat("========================================\n\n")

if (nrow(kappa_results) > 0) {
  cat("[Categorical variables] - Cohen's Kappa summary\n\n")
  print(kappa_results, row.names = FALSE)
}

cat("\n")

if (nrow(icc_results) > 0) {
  cat("[Continuous variables] - ICC summary\n\n")
  print(icc_results, row.names = FALSE)
}

# =============================================================================
# 6. Save outputs
# =============================================================================

# Build output directory under current working directory
results_root <- file.path(getwd(), "Results")
run_id <- format(Sys.time(), "%Y%m%d_%H%M%S")
run_root <- file.path(results_root, "Manuscript_v2", paste0("run_", run_id))
output_dir <- file.path(run_root, "00_Data_Quality", "Double_Entry_Consistency")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  cat(paste("\nCreated output directory:", output_dir, "\n"))
}

# Save categorical results
if (nrow(kappa_results) > 0) {
  write.csv(kappa_results,
            file = file.path(output_dir, "categorical_kappa_results.csv"),
            row.names = FALSE,
            fileEncoding = "UTF-8")
  cat(paste("\nCategorical results saved to:",
            file.path(output_dir, "categorical_kappa_results.csv"), "\n"))
}

# Save continuous results
if (nrow(icc_results) > 0) {
  write.csv(icc_results,
            file = file.path(output_dir, "continuous_icc_results.csv"),
            row.names = FALSE,
            fileEncoding = "UTF-8")
  cat(paste("Continuous results saved to:",
            file.path(output_dir, "continuous_icc_results.csv"), "\n"))
}

# =============================================================================
# 7. Visualizations
# =============================================================================

# ICC bar plot
if (nrow(icc_results) > 0) {
  # Use CI lower bound values where available
  valid_data <- icc_results[!is.na(icc_results$ICC_average_CI_low), ]

  if (nrow(valid_data) > 0) {
    tiff(file.path(output_dir, "icc_barplot.tiff"),
         width = 8.33, height = 5, units = "in",
         res = 300, compression = "lzw")

    par(mar = c(7, 5, 4, 2))

    icc_values <- valid_data$ICC_average_CI_low
    names(icc_values) <- valid_data$Variable

    # Sort descending
    icc_values <- sort(icc_values, decreasing = TRUE)

    # Color by quality level
    colors <- ifelse(icc_values >= 0.90, "darkgreen",
                     ifelse(icc_values >= 0.75, "orange",
                            ifelse(icc_values >= 0.50, "yellow", "red")))

    barplot(icc_values,
            horiz = TRUE,
            las = 1,
            col = colors,
            xlim = c(0, 1),
            main = "Continuous consistency check (ICC average CI lower bound)",
            xlab = "ICC value (average measure, 95% CI lower bound)")

    # Reference lines
    abline(v = c(0.50, 0.75, 0.90), lty = 2, col = "gray")
    text(x = c(0.52, 0.77, 0.92), y = par("usr")[3],
         labels = c("Poor", "Moderate", "Good"), xpd = TRUE, cex = 0.8)

    dev.off()
    cat(paste("ICC bar plot saved to:", file.path(output_dir, "icc_barplot.tiff"), "\n"))
  } else {
    cat("ICC bar plot skipped: no valid data\n")
  }
}

# Kappa bar plot
if (nrow(kappa_results) > 0) {
  tiff(file.path(output_dir, "kappa_barplot.tiff"),
       width = 8.33, height = 5, units = "in",
       res = 300, compression = "lzw")

  par(mar = c(7, 5, 4, 2))

  kappa_values <- kappa_results$Kappa
  names(kappa_values) <- kappa_results$Variable

  # Sort descending
  kappa_values <- sort(kappa_values, decreasing = TRUE)

  # Color by quality level
  colors <- ifelse(kappa_values >= 0.80, "darkgreen",
                   ifelse(kappa_values >= 0.60, "orange",
                          ifelse(kappa_values >= 0.40, "yellow",
                                 ifelse(kappa_values >= 0.20, "orange", "red"))))

  barplot(kappa_values,
          horiz = TRUE,
          las = 1,
          col = colors,
          xlim = c(-0.1, 1),
          main = "Categorical consistency check (Cohen's Kappa)",
          xlab = "Kappa value")

  # Reference lines
  abline(v = c(0, 0.20, 0.40, 0.60, 0.80), lty = 2, col = "gray")

  dev.off()
  cat(paste("Kappa bar plot saved to:", file.path(output_dir, "kappa_barplot.tiff"), "\n"))
}

# Bland-Altman plots for continuous variables
if (nrow(icc_results) > 0) {
  for (var in continuous_vars) {
    data <- parse_double_entry(raw_data, var)

    if (!is.null(data) && nrow(data) > 0) {
      tiff(file.path(output_dir, paste0("bland_altman_", var, ".tiff")),
           width = 6.67, height = 5, units = "in",
           res = 300, compression = "lzw")

      # Mean and difference
      means <- (data$Rater1 + data$Rater2) / 2
      diffs <- data$Rater1 - data$Rater2

      # Bias and limits of agreement
      bias <- mean(diffs, na.rm = TRUE)
      sd_diff <- sd(diffs, na.rm = TRUE)
      loa_upper <- bias + 1.96 * sd_diff
      loa_lower <- bias - 1.96 * sd_diff

      # Plot
      plot(means, diffs,
           pch = 19, col = "blue",
           xlab = "Mean of two raters",
           ylab = "Difference (Rater1 - Rater2)",
           main = paste("Bland-Altman plot -", var))

      # Reference lines
      abline(h = 0, lty = 2, col = "gray")
      abline(h = bias, lty = 1, col = "red", lwd = 2)
      abline(h = loa_upper, lty = 2, col = "red")
      abline(h = loa_lower, lty = 2, col = "red")

      # Labels
      text(x = max(means), y = bias, labels = sprintf("Bias: %.2f", bias),
           pos = 2, col = "red")
      text(x = max(means), y = loa_upper,
           labels = sprintf("+1.96SD: %.2f", loa_upper), pos = 2, col = "red")
      text(x = max(means), y = loa_lower,
           labels = sprintf("-1.96SD: %.2f", loa_lower), pos = 2, col = "red")

      dev.off()
    }
  }
  cat(paste("Bland-Altman plots saved under:", output_dir, "\n"))
}

# =============================================================================
# 8. Compact summary table
# =============================================================================

cat("\n")
cat("========================================\n")
cat("Compact consistency summary\n")
cat("========================================\n\n")
cat("Note: categorical variables use Kappa; continuous variables use ICC (average).\n\n")

summary_results <- data.frame(
  Variable = character(),
  Kappa_or_ICC = character(),
  Agreement = character(),
  stringsAsFactors = FALSE
)

if (nrow(kappa_results) > 0) {
  summary_results <- rbind(summary_results, data.frame(
    Variable = kappa_results$Variable,
    Kappa_or_ICC = ifelse(is.na(kappa_results$Kappa),
                          NA_character_,
                          paste0("Kappa=", sprintf("%.3f", kappa_results$Kappa))),
    Agreement = kappa_results$Agreement,
    stringsAsFactors = FALSE
  ))
}

if (nrow(icc_results) > 0) {
  summary_results <- rbind(summary_results, data.frame(
    Variable = icc_results$Variable,
    Kappa_or_ICC = ifelse(is.na(icc_results$ICC_average),
                          NA_character_,
                          paste0("ICC_avg=", sprintf("%.3f", icc_results$ICC_average))),
    Agreement = icc_results$Agreement,
    stringsAsFactors = FALSE
  ))
}

if (nrow(summary_results) > 0) {
  print(summary_results, row.names = FALSE)
} else {
  cat("No usable results.\n")
}

cat("\n========================================\n")
cat("Analysis completed.\n")
cat("========================================\n")
