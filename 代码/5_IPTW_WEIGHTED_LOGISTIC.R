# IPTW weighted logistic analysis for Komori contained vs non-contained (ATE)

# ============================================================================
# KEY PARAMETERS (editable)
# ============================================================================
# Follow-up time filter in months; set NA to disable time filtering.
MAX_FOLLOW_UP_MONTHS <- NA  # Examples: NA, 12, 24, 36
# ============================================================================

# Create output directory
results_root <- file.path(getwd(), "Results")
run_id <- format(Sys.time(), "%Y%m%d_%H%M%S")
run_root <- file.path(results_root, "Manuscript_v2", paste0("run_", run_id))
iptw_root <- file.path(run_root, "07_IPTW_WEIGHTED_LOGISTIC")
overlap_dir <- file.path(iptw_root, "overlap_balance")
outcome_dir <- file.path(iptw_root, "outcomes")

for (d in c(iptw_root, overlap_dir, outcome_dir)) {
  if (!dir.exists(d)) {
    dir.create(d, recursive = TRUE)
  }
}
message("Created output directory: ", iptw_root)

required_pkgs <- c("readxl", "survey", "ggplot2", "cobalt")
missing_pkgs <- required_pkgs[!sapply(required_pkgs, requireNamespace, quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop("Missing required packages: ", paste(missing_pkgs, collapse = ", "), ". Please install them.")
}

library(readxl)
library(survey)
library(ggplot2)
library(cobalt)

# ---------------------------------------------------------------------------
# Input file path — auto-discovery (same pattern as 5_lmm_vas_odi_joa_analysis.R)
# Directory layout assumed:
#   F:\李子航毕业论文原始数据\代码\   <- this script lives here
#   F:\李子航毕业论文原始数据\文件\   <- data files live here (sibling folder)
#
# Candidate paths tried in order:
#   1. <script_dir>/../文件/Retrospective data.xlsx  (standard layout)
#   2. <working_dir>/文件/Retrospective data.xlsx     (run from project root)
#   3. Retrospective data.xlsx                        (data in working dir, fallback)
# ---------------------------------------------------------------------------
TARGET_FILE <- "Retrospective data.xlsx"

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
  file.path(script_dir, "..", "文件", TARGET_FILE),   # sibling 文件 folder (standard)
  file.path(getwd(),    "文件", TARGET_FILE),          # 文件 sub-folder of working dir
  file.path(getwd(),    TARGET_FILE)                           # working dir fallback
)
input_candidates <- normalizePath(input_candidates, winslash = "/", mustWork = FALSE)

existing_inputs <- input_candidates[file.exists(input_candidates)]
file_path <- if (length(existing_inputs) > 0) {
  existing_inputs[[1]]
} else {
  stop(
    "Input file '", TARGET_FILE, "' not found. Paths tried:\n",
    paste0("  ", input_candidates, collapse = "\n")
  )
}
message("Reading data from: ", file_path)

sheets <- readxl::excel_sheets(file_path)
if (!("Train" %in% sheets)) {
  stop("Worksheet 'Train' not found in: ", file_path)
}

raw <- readxl::read_excel(file_path, sheet = "Train")
dat <- as.data.frame(raw)

expected_cols <- c(
  "ID", "Name", "Age", "Gender", "Herniated_Level", "Pfirrmann", "Iwabuchi", "Modic", "Komori", "MSU",
  "Spinal_canal_stenosis", "Bull_eye", "SS", "Upper_VB_Posterior_Height_CM", "Lower_VB_Posterior_Height_CM",
  "RSI", "DHI", "Initial_volume", "Months_of_Review", "Last_volume", "Absorption_rate",
  "Reabsorption", "Absorption_type"
)

missing_cols <- setdiff(expected_cols, names(dat))
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

n0 <- nrow(dat)

# One-cut filter: Months_of_Review (adjustable parameter)
if (!is.na(MAX_FOLLOW_UP_MONTHS)) {
  cat(sprintf("Applying filter: Months_of_Review <= %d months\n", MAX_FOLLOW_UP_MONTHS))
  idx_months <- !is.na(dat$Months_of_Review) & dat$Months_of_Review <= MAX_FOLLOW_UP_MONTHS
  filter_desc <- sprintf("Months_of_Review <= %d", MAX_FOLLOW_UP_MONTHS)
} else {
  cat("No filter on Months_of_Review - including all samples\n")
  idx_months <- rep(TRUE, nrow(dat))
  filter_desc <- "All follow-up times"
}

dat <- dat[idx_months, , drop = FALSE]

n1 <- nrow(dat)
if (!is.na(MAX_FOLLOW_UP_MONTHS)) {
  if (n1 == 0) {
    stop(sprintf("No rows after %s filter.", filter_desc))
  }
}

# Reabsorption non-missing and binary conversion
idx_reabs <- !is.na(dat$Reabsorption)
dat <- dat[idx_reabs, , drop = FALSE]

n2 <- nrow(dat)
if (n2 == 0) {
  stop("No rows after Reabsorption non-missing filter.")
}

convert_binary <- function(x, name) {
  if (is.logical(x)) {
    return(as.integer(x))
  }
  if (is.factor(x)) {
    x <- as.character(x)
  }
  if (is.character(x)) {
    x_trim <- trimws(x)
    bad <- !(x_trim %in% c("0", "1"))
    if (any(bad, na.rm = TRUE)) {
      bad_vals <- unique(x_trim[bad])
      stop(name, " has non-binary values: ", paste(bad_vals, collapse = ", "))
    }
    return(as.integer(x_trim))
  }
  if (is.numeric(x) || is.integer(x)) {
    bad <- !(x %in% c(0, 1))
    if (any(bad, na.rm = TRUE)) {
      bad_vals <- unique(x[bad])
      stop(name, " has non-binary values: ", paste(bad_vals, collapse = ", "))
    }
    return(as.integer(x))
  }
  stop(name, " has unsupported type for binary conversion: ", class(x)[1])
}

dat$Reabsorption <- convert_binary(dat$Reabsorption, "Reabsorption")

# Define Rupture from Komori
komori_num <- suppressWarnings(as.numeric(as.character(dat$Komori)))
rupture_num <- ifelse(komori_num == 1, 0, ifelse(komori_num %in% c(2, 3, 4), 1, NA))

# Keep only rows with valid rupture classification
dat <- dat[!is.na(rupture_num), , drop = FALSE]

n3 <- nrow(dat)
if (n3 == 0) {
  stop("No rows after Komori-based Rupture definition.")
}

# Recompute rupture_num after filtering to ensure correct length
komori_num_filtered <- suppressWarnings(as.numeric(as.character(dat$Komori)))
rupture_num_filtered <- ifelse(komori_num_filtered == 1, 0, ifelse(komori_num_filtered %in% c(2, 3, 4), 1, NA))

dat$Rupture <- factor(
  rupture_num_filtered,
  levels = c(0, 1),
  labels = c("Contained (Komori 1)", "Non-contained (Komori 2-4)")
)

# Bull_eye: treat missing as explicit level "NA"
if ("Bull_eye" %in% names(dat)) {
  dat$Bull_eye <- as.character(dat$Bull_eye)
  dat$Bull_eye[is.na(dat$Bull_eye) | trimws(dat$Bull_eye) == ""] <- "NA"
  bull_levels <- unique(dat$Bull_eye)
  bull_levels <- bull_levels[bull_levels != "NA"]
  bull_levels <- c(sort(bull_levels), "NA")
  dat$Bull_eye <- factor(dat$Bull_eye, levels = bull_levels)
}

exclude_cols <- c(
  "ID", "Name", "Absorption_type", "Last_volume", "Absorption_rate",
  "Months_of_Review", "Komori", "Reabsorption", "Rupture"
)

ps_covars <- setdiff(names(dat), exclude_cols)
if (length(ps_covars) == 0) {
  stop("No covariates available for PS model.")
}

factor_vars <- c("Gender", "Herniated_Level", "Iwabuchi", "Modic", "Spinal_canal_stenosis", "Bull_eye")
numeric_vars <- c(
  "Age", "SS", "Upper_VB_Posterior_Height_CM", "Lower_VB_Posterior_Height_CM",
  "RSI", "DHI", "Initial_volume"
)
ordered_vars <- c("Pfirrmann", "MSU")  # Ordered categorical variables

for (v in ps_covars) {
  if (v %in% factor_vars) {
    dat[[v]] <- as.factor(dat[[v]])
  } else if (v %in% ordered_vars) {
    dat[[v]] <- as.ordered(dat[[v]])  # Preserve ordinal information
  } else if (v %in% numeric_vars) {
    dat[[v]] <- as.numeric(dat[[v]])
  } else {
    if (!is.numeric(dat[[v]]) && !is.factor(dat[[v]])) {
      dat[[v]] <- as.factor(dat[[v]])
    }
  }
}

complete_idx <- complete.cases(dat[, ps_covars, drop = FALSE])

n4 <- sum(complete_idx)
if (n4 == 0) {
  stop("No complete cases for PS covariates.")
}
if (n4 < nrow(dat)) {
  message("Removed ", nrow(dat) - n4, " rows with missing PS covariates.")
}

dat <- dat[complete_idx, , drop = FALSE]

ps_formula <- as.formula(paste("Rupture ~", paste(ps_covars, collapse = " + ")))
ps_model <- glm(ps_formula, data = dat, family = binomial())
ps <- predict(ps_model, type = "response")

eps <- 1e-6
ps <- pmin(pmax(ps, eps), 1 - eps)

dat$ps <- ps

pt <- mean(dat$Rupture == levels(dat$Rupture)[2])

sw <- ifelse(dat$Rupture == levels(dat$Rupture)[2], pt / ps, (1 - pt) / (1 - ps))
dat$sw <- sw

weight_diag <- function(w) {
  list(
    summary = summary(w),
    sd = sd(w, na.rm = TRUE),
    quantile = quantile(w, c(0.95, 0.99, 0.995), na.rm = TRUE),
    max = max(w, na.rm = TRUE),
    ess = (sum(w)^2) / sum(w^2)
  )
}

print_weight_diag <- function(label, diag) {
  cat(label, " summary:\n")
  print(diag$summary)
  cat(label, " sd:", diag$sd, "\n")
  print(diag$quantile)
  cat(label, " max:", diag$max, "\n")
  cat(label, " ESS:", diag$ess, "\n")
}

cat("Weight diagnostics (sw):\n")
print_weight_diag("sw", weight_diag(sw))

max_sw <- max(sw, na.rm = TRUE)
sd_sw <- sd(sw, na.rm = TRUE)

if (is.finite(max_sw) && is.finite(sd_sw) && max_sw <= 10 && sd_sw <= 2) {
  w_main <- sw
  decision <- "No trimming"
} else {
  q01 <- quantile(sw, 0.01, na.rm = TRUE)
  q99 <- quantile(sw, 0.99, na.rm = TRUE)
  w_main <- pmin(pmax(sw, q01), q99)
  decision <- "Trim 1%-99%"
}

dat$w_main <- w_main

# ============================================================================
# Diagnostics plot 1: PS overlap (unweighted + IPTW-weighted)
# ============================================================================
p_ps_unw <- ggplot(dat, aes(x = ps, fill = Rupture)) +
  geom_density(alpha = 0.35) +
  labs(
    title = "Propensity score overlap (unweighted)",
    x = "Propensity score",
    y = "Density",
    caption = "Unweighted vs IPTW-weighted"
  ) +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom")

ggsave(
  file.path(overlap_dir, "PS_overlap_density_unweighted.tiff"),
  p_ps_unw,
  width = 7,
  height = 5,
  units = "in",
  dpi = 300,
  compression = "lzw"
)

p_ps_w <- ggplot(dat, aes(x = ps, fill = Rupture, weight = w_main)) +
  geom_density(alpha = 0.35) +
  labs(
    title = "Propensity score overlap (IPTW-weighted)",
    x = "Propensity score",
    y = "Density",
    caption = "Unweighted vs IPTW-weighted"
  ) +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom")

ggsave(
  file.path(overlap_dir, "PS_overlap_density_weighted.tiff"),
  p_ps_w,
  width = 7,
  height = 5,
  units = "in",
  dpi = 300,
  compression = "lzw"
)

# ============================================================================
# Diagnostics plot 2: Love plot (SMD before vs after IPTW)
# ============================================================================
bal <- cobalt::bal.tab(
  ps_formula,
  data = dat,
  weights = dat$w_main,
  method = "weighting",
  estimand = "ATE",
  un = TRUE
)

p_love <- cobalt::love.plot(
  bal,
  abs = TRUE,
  threshold = 0.1,
  var.order = "unadjusted",
  title = "Covariate balance: unadjusted vs IPTW-weighted",
  return.plot = TRUE
)

p_love <- p_love +
  coord_cartesian(xlim = c(0, 0.30)) +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom")

tryCatch(
  ggsave(
    file.path(overlap_dir, "Love_plot_SMD_unadj_vs_adj.tiff"),
    p_love,
    width = 8,
    height = 10,
    units = "in",
    dpi = 300,
    compression = "lzw"
  ),
  error = function(e) {
    tiff(
      file.path(overlap_dir, "Love_plot_SMD_unadj_vs_adj.tiff"),
      width = 8, height = 10, units = "in",
      res = 300, compression = "lzw"
    )
    print(p_love)
    dev.off()
  }
)

# ============================================================================
# eTable2: Level-wise covariate balance diagnostics (cobalt)
# ============================================================================
bal2 <- cobalt::bal.tab(
  ps_formula,
  data = dat,
  weights = dat$w_main,
  method = "weighting",
  estimand = "ATE",
  un = TRUE
)

# extract balance table (version-robust)
bal_tbl <- NULL
if (!is.null(bal2$Balance)) {
  bal_tbl <- bal2$Balance
} else {
  bal_sum <- tryCatch(summary(bal2), error = function(e) NULL)
  if (!is.null(bal_sum) && !is.null(bal_sum$Balance)) {
    bal_tbl <- bal_sum$Balance
  }
}

if (is.null(bal_tbl)) {
  stop("Cannot find balance table in cobalt bal.tab object. Please check str(bal2).")
}

bal_df <- as.data.frame(bal_tbl)
bal_df$var_name <- rownames(bal_df)
rownames(bal_df) <- NULL

# Helper: pick adjusted SMD-like column from cobalt balance output
pick_adj_smd_col <- function(df) {
  if (!is.data.frame(df) || ncol(df) == 0) {
    return(NA_character_)
  }

  nm <- names(df)
  priority <- c("Diff.Adj", "SMD.Adj", "Adj")
  hit <- priority[priority %in% nm]
  if (length(hit) > 0) {
    return(hit[1])
  }

  hit <- grep("(?i)(diff|smd).*adj|adj.*(diff|smd)", nm, value = TRUE, perl = TRUE)
  if (length(hit) > 0) {
    return(hit[1])
  }

  hit <- grep("(?i)adj$", nm, value = TRUE, perl = TRUE)
  if (length(hit) > 0) {
    return(hit[1])
  }

  NA_character_
}

# Helper: summarize max absolute adjusted SMD by original covariate
summarize_adj_smd_by_covar <- function(balance_df, covars, adj_col) {
  if (!is.data.frame(balance_df) || nrow(balance_df) == 0 ||
      !("var_name" %in% names(balance_df)) || !(adj_col %in% names(balance_df)) ||
      length(covars) == 0) {
    return(data.frame(
      covariate = character(0),
      max_abs_adj_smd = numeric(0),
      stringsAsFactors = FALSE
    ))
  }

  vn <- as.character(balance_df$var_name)
  adj_vals <- suppressWarnings(as.numeric(balance_df[[adj_col]]))
  out <- lapply(covars, function(v) {
    idx <- which(vn == v | startsWith(vn, paste0(v, "_")))
    max_abs <- NA_real_
    if (length(idx) > 0) {
      vals <- abs(adj_vals[idx])
      if (all(is.na(vals))) {
        max_abs <- NA_real_
      } else {
        max_abs <- max(vals, na.rm = TRUE)
      }
    }
    data.frame(
      covariate = v,
      max_abs_adj_smd = max_abs,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, out)
}

# keep informative columns if present (compatible across cobalt versions)
keep_patterns <- c("^Diff", "^SMD", "Un$", "Adj$", "M\\.0", "M\\.1", "V\\.Ratio")
keep_cols <- unique(unlist(lapply(keep_patterns, function(p) grep(p, names(bal_df), value = TRUE))))
if (length(keep_cols) > 0) {
  bal_out <- bal_df[, c("var_name", keep_cols), drop = FALSE]
} else {
  bal_out <- bal_df
}

out_et2 <- file.path(outcome_dir, "eTable2_balance_levelwise.csv")
write.csv(bal_out, out_et2, row.names = FALSE, fileEncoding = "UTF-8")
cat(sprintf("Saved level-wise balance table to: %s\n", out_et2))

# Residual imbalance profiling for DR covariate selection
DR_SMD_THRESHOLD <- 0.1
adj_smd_col <- pick_adj_smd_col(bal_df)
adj_smd_summary <- summarize_adj_smd_by_covar(bal_df, ps_covars, adj_smd_col)
out_dr_screen <- file.path(outcome_dir, "DR_residual_imbalance_screening.csv")
write.csv(adj_smd_summary, out_dr_screen, row.names = FALSE, fileEncoding = "UTF-8")

if (is.na(adj_smd_col)) {
  cat("Warning: no adjusted SMD column found in balance output; DR covariate auto-selection will be skipped.\n")
} else {
  cat(sprintf(
    "DR auto-selection uses %s with threshold |SMD| > %.3f.\n",
    adj_smd_col, DR_SMD_THRESHOLD
  ))
}
cat(sprintf("Saved DR residual imbalance screening to: %s\n", out_dr_screen))

cat("Weight diagnostics (w_main):\n")
print_weight_diag("w_main", weight_diag(w_main))


# ============================================================================
# Helper formatting and SMD functions
# (defined here so they are available before first use; previously defined
#  after the eTable loop which caused "could not find function" errors)
# ============================================================================
format_n_pct <- function(n, denom) {
  if (is.na(denom) || denom <= 0) return("NA")
  sprintf("%d (%.1f)", n, 100 * n / denom)
}
format_w_n_pct <- function(w, denom) {
  if (is.na(denom) || denom <= 0) return("NA")
  sprintf("%.1f (%.1f)", w, 100 * w / denom)
}
format_mean_sd <- function(m, s) {
  if (is.na(m) || is.na(s)) return("NA")
  sprintf("%.2f (%.2f)", m, s)
}
format_p <- function(p) {
  if (is.na(p)) return("")
  if (p < 1e-4) return("<0.0001")
  formatC(p, format = "f", digits = 4)
}
smd_cont <- function(m0, sd0, m1, sd1) {
  if (any(is.na(c(m0, sd0, m1, sd1)))) return(NA_real_)
  pooled <- sqrt((sd0^2 + sd1^2) / 2)
  if (pooled == 0) return(0)
  abs(m1 - m0) / pooled
}
smd_cat <- function(p0, p1) {
  if (any(is.na(c(p0, p1)))) return(NA_real_)
  pbar <- (p0 + p1) / 2
  denom <- sum(pbar * (1 - pbar))
  if (denom == 0) return(0)
  sqrt(sum((p1 - p0)^2) / denom)
}

design <- survey::svydesign(ids = ~1, data = dat, weights = ~w_main)

# ============================================================================
# Helper: extract OR/CI/p with robust SE (svyglm provides robust SE)
# ============================================================================
extract_svyglm_effect <- function(fit, term_pattern = "Rupture") {
  s <- summary(fit)
  coef_tab <- s$coefficients
  term_names <- rownames(coef_tab)
  matched <- grep(term_pattern, term_names, value = TRUE)
  if (length(matched) == 0) {
    warning("No term matched pattern '", term_pattern, "'. Using first coefficient.")
    term_used <- term_names[1]
  } else {
    term_used <- matched[1]
  }

  beta <- coef_tab[term_used, "Estimate"]
  se_used <- coef_tab[term_used, "Std. Error"]
  z <- beta / se_used
  p <- 2 * (1 - pnorm(abs(z)))

  list(
    OR = exp(beta),
    CI_low = exp(beta - 1.96 * se_used),
    CI_high = exp(beta + 1.96 * se_used),
    p = p,
    se_used = se_used,
    beta = beta,
    term_used = term_used
  )
}

# Helper: extract OR/CI/p from standard glm
extract_glm_effect <- function(fit, term_pattern = "Rupture") {
  s <- summary(fit)
  coef_tab <- s$coefficients
  term_names <- rownames(coef_tab)
  matched <- grep(term_pattern, term_names, value = TRUE)
  if (length(matched) == 0) {
    warning("No term matched pattern '", term_pattern, "'. Using first coefficient.")
    term_used <- term_names[1]
  } else {
    term_used <- matched[1]
  }

  beta <- coef_tab[term_used, "Estimate"]
  se_used <- coef_tab[term_used, "Std. Error"]
  z <- beta / se_used
  p <- 2 * (1 - pnorm(abs(z)))

  list(
    OR = exp(beta),
    CI_low = exp(beta - 1.96 * se_used),
    CI_high = exp(beta + 1.96 * se_used),
    p = p,
    se_used = se_used,
    beta = beta,
    term_used = term_used
  )
}

format_ci <- function(ci_low, ci_high) {
  if (any(is.na(c(ci_low, ci_high)))) {
    return(NA_character_)
  }
  sprintf("[%.6f, %.6f]", ci_low, ci_high)
}

# ============================================================================
# Unweighted comparison models for Table 4
# ============================================================================
fit_unadj_logit <- glm(
  Reabsorption ~ Rupture,
  data = dat,
  family = binomial()
)

res_unadj <- extract_glm_effect(fit_unadj_logit, "Rupture")
or_unadj <- res_unadj$OR
ci_low_unadj <- res_unadj$CI_low
ci_high_unadj <- res_unadj$CI_high
pval_unadj <- res_unadj$p

unadj_line <- sprintf(
  "Unadjusted model (glm, unweighted): OR=%.4f, 95%%CI [%.4f, %.4f], p=%.4g",
  or_unadj, ci_low_unadj, ci_high_unadj, pval_unadj
)
cat(unadj_line, "\n")

multi_formula <- as.formula(
  paste("Reabsorption ~ Rupture +", paste(ps_covars, collapse = " + "))
)
fit_multi_logit <- glm(
  multi_formula,
  data = dat,
  family = binomial()
)

res_multi <- extract_glm_effect(fit_multi_logit, "Rupture")
or_multi <- res_multi$OR
ci_low_multi <- res_multi$CI_low
ci_high_multi <- res_multi$CI_high
pval_multi <- res_multi$p

multiv_line <- sprintf(
  "Multivariable model (glm, unweighted, adjusted by ps_covars): OR=%.4f, 95%%CI [%.4f, %.4f], p=%.4g",
  or_multi, ci_low_multi, ci_high_multi, pval_multi
)
cat(multiv_line, "\n")

# ============================================================================
# Primary analysis: IPTW-weighted logistic regression (robust SE)
# Outcome: Reabsorption (binary at last MRI); Exposure: Rupture
# ============================================================================
fit_iptw_logit <- survey::svyglm(
  Reabsorption ~ Rupture,
  design = design,
  family = quasibinomial()
)

res_main <- extract_svyglm_effect(fit_iptw_logit, "Rupture")
or <- res_main$OR
ci_low <- res_main$CI_low
ci_high <- res_main$CI_high
pval <- res_main$p

primary_line <- sprintf(
  "Primary analysis: IPTW-weighted logistic regression (decision=%s) OR=%.4f, 95%%CI [%.4f, %.4f], p=%.4g",
  decision, or, ci_low, ci_high, pval
)
cat(primary_line, "\n")

out_txt <- file.path(outcome_dir, "IPTW_weighted_logistic_results.txt")
writeLines(
  c(
    sprintf("Decision: %s", decision),
    "Model: IPTW-weighted logistic regression (svyglm, quasibinomial; robust SE)",
    sprintf("OR = %.6f", or),
    sprintf("95%% CI = [%.6f, %.6f]", ci_low, ci_high),
    sprintf("p = %.6g", pval)
  ),
  con = out_txt
)
cat(unadj_line, "\n", file = out_txt, append = TRUE)
cat(multiv_line, "\n", file = out_txt, append = TRUE)

# ============================================================================
# Doubly robust (DR) analysis:
# IPTW-weighted logistic regression + adjustment for residual-imbalance covariates
# Covariates are auto-selected by weighted balance: |adjusted SMD| > DR_SMD_THRESHOLD
# ============================================================================
dr_vars <- character(0)
if (!is.na(adj_smd_col) && nrow(adj_smd_summary) > 0) {
  dr_vars <- adj_smd_summary$covariate[
    is.finite(adj_smd_summary$max_abs_adj_smd) &
      (adj_smd_summary$max_abs_adj_smd > DR_SMD_THRESHOLD)
  ]
  dr_vars <- dr_vars[dr_vars %in% names(dat)]
}
if (length(dr_vars) > 0) {
  non_all_na <- sapply(dr_vars, function(v) !all(is.na(dat[[v]])))
  dr_vars <- dr_vars[non_all_na]
}

res_dr <- list(
  OR = NA_real_,
  CI_low = NA_real_,
  CI_high = NA_real_,
  p = NA_real_
)
dr_method_table4 <- "Doubly robust: IPTW-weighted logistic + outcome adjustment (svyglm)"

if (length(dr_vars) < 1) {
  dr_line <- sprintf(
    "DR analysis skipped: no available covariates met residual imbalance criterion |%s| > %.3f.",
    ifelse(is.na(adj_smd_col), "Adj.SMD", adj_smd_col),
    DR_SMD_THRESHOLD
  )
  cat(dr_line, "\n")
  cat(dr_line, "\n", file = out_txt, append = TRUE)
  dr_method_table4 <- paste0(
    dr_method_table4,
    " [skipped: no available covariates met residual imbalance criterion]"
  )
  
  # Still create eTable3 with primary only
  eTable3 <- data.frame(
    Model = c("IPTW-only"),
    TrimmingDecision = c(decision),
    OutcomeAdjustment = c("None (Reabsorption ~ Rupture)"),
    OR = c(or),
    CI_low = c(ci_low),
    CI_high = c(ci_high),
    p_value = c(pval),
    stringsAsFactors = FALSE
  )
  write.csv(eTable3, file = file.path(outcome_dir, "eTable3_IPTW_weighted_logistic_DR.csv"), row.names = FALSE)

} else {
  dr_formula <- as.formula(
    paste("Reabsorption ~ Rupture +", paste(dr_vars, collapse = " + "))
  )
  
  fit_iptw_logit_dr <- survey::svyglm(
    dr_formula,
    design = design,
    family = quasibinomial()
  )
  
  res_dr <- extract_svyglm_effect(fit_iptw_logit_dr, "Rupture")
  or_dr <- res_dr$OR
  ci_low_dr <- res_dr$CI_low
  ci_high_dr <- res_dr$CI_high
  pval_dr <- res_dr$p
  
  dr_line <- sprintf(
    "Doubly robust (IPTW + outcome adjustment for residual imbalance): Criterion=|%s|>%.3f; Adjusted=%s | OR=%.4f, 95%%CI [%.4f, %.4f], p=%.4g",
    ifelse(is.na(adj_smd_col), "Adj.SMD", adj_smd_col),
    DR_SMD_THRESHOLD,
    paste(dr_vars, collapse = "; "),
    or_dr, ci_low_dr, ci_high_dr, pval_dr
  )
  cat(dr_line, "\n")
  cat(dr_line, "\n", file = out_txt, append = TRUE)

  # eTable3 output
  eTable3 <- data.frame(
    Model = c("IPTW-only", "Doubly robust (IPTW + outcome adjustment)"),
    TrimmingDecision = c(decision, decision),
    OutcomeAdjustment = c(
      "None (Reabsorption ~ Rupture)",
      paste0("Adjusted: ", paste(dr_vars, collapse = " + "))
    ),
    OR = c(or, or_dr),
    CI_low = c(ci_low, ci_low_dr),
    CI_high = c(ci_high, ci_high_dr),
    p_value = c(pval, pval_dr),
    stringsAsFactors = FALSE
  )
  write.csv(eTable3, file = file.path(outcome_dir, "eTable3_IPTW_weighted_logistic_DR.csv"), row.names = FALSE)
}

# ============================================================================
# Table 4: unified four-model comparison
# ============================================================================
table4_out <- file.path(outcome_dir, "Table4_logistic_models_comparison.csv")

table4 <- data.frame(
  Model = c(
    "Unadjusted Model",
    "Multivariable Model",
    "IPTW-weighted Model",
    "Doubly Robust Model"
  ),
  Method = c(
    "Standard logistic regression (glm, unweighted, unadjusted)",
    "Standard logistic regression (glm, unweighted, adjusted by ps_covars)",
    "IPTW-weighted logistic regression (svyglm, quasibinomial; robust SE)",
    dr_method_table4
  ),
  OR = signif(c(or_unadj, or_multi, res_main$OR, res_dr$OR), 6),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

table4[["95% CI"]] <- c(
  format_ci(ci_low_unadj, ci_high_unadj),
  format_ci(ci_low_multi, ci_high_multi),
  format_ci(res_main$CI_low, res_main$CI_high),
  format_ci(res_dr$CI_low, res_dr$CI_high)
)
table4$P_value <- signif(c(pval_unadj, pval_multi, res_main$p, res_dr$p), 6)
table4 <- table4[, c("Model", "Method", "OR", "95% CI", "P_value")]

write.csv(table4, file = table4_out, row.names = FALSE, fileEncoding = "UTF-8")
cat(sprintf("Saved Table 4 model comparison to: %s\n", table4_out))


get_w_mean_sd <- function(design_obj, var, group_level) {
  sub <- subset(design_obj, Rupture == group_level)
  form <- as.formula(paste0("~", var))
  m <- tryCatch(as.numeric(svymean(form, sub, na.rm = TRUE)), error = function(e) NA_real_)
  v <- tryCatch(as.numeric(svyvar(form, sub, na.rm = TRUE)), error = function(e) NA_real_)
  c(mean = m, sd = sqrt(v))
}

pval_ttest <- function(x, g) {
  tryCatch(t.test(x ~ g)$p.value, error = function(e) NA_real_)
}

pval_svyttest <- function(design_obj, var) {
  form <- as.formula(paste0(var, " ~ Rupture"))
  tryCatch(svyttest(form, design_obj)$p.value, error = function(e) NA_real_)
}

pval_chisq <- function(tbl) {
  if (all(dim(tbl) == c(2, 2))) {
    expected <- suppressWarnings(chisq.test(tbl)$expected)
    if (any(expected < 5)) {
      return(tryCatch(fisher.test(tbl)$p.value, error = function(e) NA_real_))
    }
  }
  tryCatch(chisq.test(tbl)$p.value, error = function(e) NA_real_)
}

pval_svychisq <- function(design_obj, var) {
  form <- as.formula(paste0("~", var, " + Rupture"))
  tryCatch(svychisq(form, design_obj)$p.value, error = function(e) NA_real_)
}

group0 <- levels(dat$Rupture)[1]
group1 <- levels(dat$Rupture)[2]

n0_non <- sum(dat$Rupture == group0)
n0_rup <- sum(dat$Rupture == group1)

w_non <- sum(dat$w_main[dat$Rupture == group0])
w_rup <- sum(dat$w_main[dat$Rupture == group1])

etable_rows <- list()

etable_rows[[length(etable_rows) + 1]] <- data.frame(
  Characteristic = "n",
  Before_Non = as.character(n0_non),
  Before_Rup = as.character(n0_rup),
  Before_p = "",
  Before_SMD = "",
  After_Non = sprintf("%.1f", w_non),
  After_Rup = sprintf("%.1f", w_rup),
  After_p = "",
  After_SMD = "",
  stringsAsFactors = FALSE
)

for (v in ps_covars) {
  if (v %in% factor_vars || v %in% ordered_vars) {
    # Categorical variables (nominal + ordered)
    x <- dat[[v]]
    if (!is.factor(x)) {
      x <- as.factor(x)
    }
    lvls <- levels(x)

    tbl <- table(x, dat$Rupture)
    p_before <- pval_chisq(tbl)
    p_after <- pval_svychisq(design, v)

    p0 <- p1 <- numeric(length(lvls))
    for (i in seq_along(lvls)) {
      lvl <- lvls[i]
      n_g0 <- sum(x == lvl & dat$Rupture == group0)
      n_g1 <- sum(x == lvl & dat$Rupture == group1)
      p0[i] <- if (n0_non > 0) n_g0 / n0_non else NA_real_
      p1[i] <- if (n0_rup > 0) n_g1 / n0_rup else NA_real_
    }
    smd_before <- smd_cat(p0, p1)

    wp0 <- wp1 <- numeric(length(lvls))
    for (i in seq_along(lvls)) {
      lvl <- lvls[i]
      w_g0 <- sum(dat$w_main[x == lvl & dat$Rupture == group0])
      w_g1 <- sum(dat$w_main[x == lvl & dat$Rupture == group1])
      wp0[i] <- if (w_non > 0) w_g0 / w_non else NA_real_
      wp1[i] <- if (w_rup > 0) w_g1 / w_rup else NA_real_
    }
    smd_after <- smd_cat(wp0, wp1)

    etable_rows[[length(etable_rows) + 1]] <- data.frame(
      Characteristic = paste0(v, " (%)"),
      Before_Non = "",
      Before_Rup = "",
      Before_p = format_p(p_before),
      Before_SMD = ifelse(is.na(smd_before), "", formatC(smd_before, format = "f", digits = 3)),
      After_Non = "",
      After_Rup = "",
      After_p = format_p(p_after),
      After_SMD = ifelse(is.na(smd_after), "", formatC(smd_after, format = "f", digits = 3)),
      stringsAsFactors = FALSE
    )

    for (i in seq_along(lvls)) {
      lvl <- lvls[i]
      n_g0 <- sum(x == lvl & dat$Rupture == group0)
      n_g1 <- sum(x == lvl & dat$Rupture == group1)
      w_g0 <- sum(dat$w_main[x == lvl & dat$Rupture == group0])
      w_g1 <- sum(dat$w_main[x == lvl & dat$Rupture == group1])
      etable_rows[[length(etable_rows) + 1]] <- data.frame(
        Characteristic = paste0("  ", lvl),
        Before_Non = format_n_pct(n_g0, n0_non),
        Before_Rup = format_n_pct(n_g1, n0_rup),
        Before_p = "",
        Before_SMD = "",
        After_Non = format_w_n_pct(w_g0, w_non),
        After_Rup = format_w_n_pct(w_g1, w_rup),
        After_p = "",
        After_SMD = "",
        stringsAsFactors = FALSE
      )
    }
  } else if (v %in% numeric_vars) {
    # Continuous variables
    x0 <- dat[dat$Rupture == group0, v]
    x1 <- dat[dat$Rupture == group1, v]
    m0 <- mean(x0, na.rm = TRUE)
    sd0 <- sd(x0, na.rm = TRUE)
    m1 <- mean(x1, na.rm = TRUE)
    sd1 <- sd(x1, na.rm = TRUE)

    wstats0 <- get_w_mean_sd(design, v, group0)
    wstats1 <- get_w_mean_sd(design, v, group1)

    p_before <- pval_ttest(dat[[v]], dat$Rupture)
    p_after <- pval_svyttest(design, v)

    smd_before <- smd_cont(m0, sd0, m1, sd1)
    smd_after <- smd_cont(wstats0["mean"], wstats0["sd"], wstats1["mean"], wstats1["sd"])

    etable_rows[[length(etable_rows) + 1]] <- data.frame(
      Characteristic = paste0(v, " (mean (SD))"),
      Before_Non = format_mean_sd(m0, sd0),
      Before_Rup = format_mean_sd(m1, sd1),
      Before_p = format_p(p_before),
      Before_SMD = ifelse(is.na(smd_before), "", formatC(smd_before, format = "f", digits = 3)),
      After_Non = format_mean_sd(wstats0["mean"], wstats0["sd"]),
      After_Rup = format_mean_sd(wstats1["mean"], wstats1["sd"]),
      After_p = format_p(p_after),
      After_SMD = ifelse(is.na(smd_after), "", formatC(smd_after, format = "f", digits = 3)),
      stringsAsFactors = FALSE
    )
  }
}

eTable1 <- do.call(rbind, etable_rows)
colnames(eTable1) <- c(
  "Characteristic",
  paste0("Before IPTW: ", group0),
  paste0("Before IPTW: ", group1),
  "Before p-value",
  "Before SMD",
  paste0("After IPTW: ", group0),
  paste0("After IPTW: ", group1),
  "After p-value",
  "After SMD"
)

write.csv(eTable1, file.path(outcome_dir, "eTable1_IPTW.csv"), row.names = FALSE, fileEncoding = "UTF-8")

# Save analysis dataset
write.csv(dat, file.path(outcome_dir, "IPTW_analysis_dataset.csv"), row.names = FALSE)

# Write results
sw_diag <- weight_diag(sw)
w_diag <- weight_diag(w_main)
summary_txt <- file.path(outcome_dir, "IPTW_weighted_logistic_summary.txt")

out_lines <- c(
  "=================================================================",
  "IPTW Weighted Logistic Analysis: Contained vs Non-contained (Komori-based)",
  "=================================================================",
  "",
  "Parameters:",
  paste0("  Follow-up time limit: ", ifelse(!is.na(MAX_FOLLOW_UP_MONTHS), sprintf("%d months", MAX_FOLLOW_UP_MONTHS), "No limit")),
  "",
  "Sample flow:",
  paste0("  Initial n = ", n0),
  sprintf("  After %s = %d", filter_desc, n1),
  paste0("  After Reabsorption non-missing = ", n2),
  paste0("  After Komori-based Rupture definition = ", n3),
  paste0("  After complete PS covariates = ", n4),
  paste0("Decision: ", decision),
  "sw summary:",
  capture.output(print(sw_diag$summary)),
  paste0("sw sd: ", sw_diag$sd),
  paste0("sw quantile 0.95/0.99/0.995: ", paste(sw_diag$quantile, collapse = ", ")),
  paste0("sw max: ", sw_diag$max),
  paste0("sw ESS: ", sw_diag$ess),
  "w_main summary:",
  capture.output(print(w_diag$summary)),
  paste0("w_main sd: ", w_diag$sd),
  paste0("w_main quantile 0.95/0.99/0.995: ", paste(w_diag$quantile, collapse = ", ")),
  paste0("w_main max: ", w_diag$max),
  paste0("w_main ESS: ", w_diag$ess),
  "",
  primary_line,
  "Model: Reabsorption ~ Rupture",
  "Method: IPTW-weighted logistic regression (svyglm, quasibinomial; robust SE)",
  "",
  "Generated files:",
  file.path(overlap_dir, "PS_overlap_density_unweighted.tiff"),
  file.path(overlap_dir, "PS_overlap_density_weighted.tiff"),
  file.path(overlap_dir, "Love_plot_SMD_unadj_vs_adj.tiff"),
  file.path(outcome_dir, "IPTW_weighted_logistic_results.txt"),
  file.path(outcome_dir, "eTable1_IPTW.csv"),
  file.path(outcome_dir, "eTable3_IPTW_weighted_logistic_DR.csv"),
  table4_out,
  file.path(outcome_dir, "IPTW_analysis_dataset.csv"),
  file.path(outcome_dir, "IPTW_weighted_logistic_summary.txt")
)

writeLines(out_lines, summary_txt)

