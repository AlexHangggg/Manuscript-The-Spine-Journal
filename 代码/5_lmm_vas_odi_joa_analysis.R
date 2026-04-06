#!/usr/bin/env Rscript

# =========================================================
# LMM analysis pipeline for VAS / ODI / JOA
# - package check and install
# - data read and reshape
# - LMM fitting per outcome
# - Type III ANOVA (Satterthwaite)
# - pre-specified Holm-adjusted contrasts
# - export tables for manuscript use
# =========================================================

required_pkgs <- c(
  "readxl", "dplyr", "tidyr", "lme4", "lmerTest",
  "emmeans", "broom.mixed", "janitor"
)

install_and_load <- function(pkgs) {
  # Avoid transient CRAN timeout failures in constrained networks.
  options(timeout = max(300, getOption("timeout")))

  missing_pkgs <- pkgs[!vapply(pkgs, requireNamespace, FUN.VALUE = logical(1), quietly = TRUE)]
  if (length(missing_pkgs) > 0) {
    cat("Installing missing packages:\n")
    print(missing_pkgs)
    install.packages(missing_pkgs, repos = "https://cloud.r-project.org")

    # Retry once for any packages still missing after the first attempt.
    still_missing <- missing_pkgs[!vapply(missing_pkgs, requireNamespace, FUN.VALUE = logical(1), quietly = TRUE)]
    if (length(still_missing) > 0) {
      cat("Retrying installation for:\n")
      print(still_missing)
      install.packages(still_missing, repos = "https://cloud.r-project.org")
    }
  }

  invisible(lapply(pkgs, function(p) {
    suppressPackageStartupMessages(library(p, character.only = TRUE))
  }))
}

install_and_load(required_pkgs)

# Ensure Type III ANOVA is meaningful for factors.
options(contrasts = c("contr.sum", "contr.poly"))
# Keep emmeans df method aligned with lmerTest and avoid KR fallback warning.
emmeans::emm_options(lmer.df = "satterthwaite")

input_sheet <- "VAS_ODI_JOA"
alpha <- 0.05

# Resolve project root from script location so outputs are stable regardless of working dir.
script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", script_args, value = TRUE)
script_path <- if (length(file_arg) >= 1) {
  normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = FALSE)
} else {
  NA_character_
}
project_root <- if (!is.na(script_path) && nzchar(script_path)) {
  normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = FALSE)
} else {
  normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}

# Prefer the shared data folder used by the other scripts.
input_candidates <- c(
  file.path(project_root, "文件", "VAS_ODI_JOA.xlsx"),
  file.path(getwd(), "VAS_ODI_JOA.xlsx"),
  "VAS_ODI_JOA.xlsx"
)
existing_inputs <- input_candidates[file.exists(input_candidates)]
input_file <- if (length(existing_inputs) > 0) existing_inputs[[1]] else input_candidates[[1]]

# Keep output directory layout aligned with other project scripts.
results_root <- file.path(project_root, "Results")
run_id <- format(Sys.time(), "%Y%m%d_%H%M%S")
run_root <- file.path(results_root, "Manuscript_v2", paste0("run_", run_id))
output_dir <- file.path(run_root, "08_LMM_VAS_ODI_JOA")

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

safe_read_data <- function(file_path, sheet_name) {
  if (!file.exists(file_path)) {
    stop(sprintf("Input file not found: %s", file_path))
  }

  # Primary path: true .xlsx workbook via readxl.
  dat <- tryCatch({
    sheets <- readxl::excel_sheets(file_path)
    if (!(sheet_name %in% sheets)) {
      stop(sprintf(
        "Sheet '%s' not found. Available sheets: %s",
        sheet_name,
        paste(sheets, collapse = ", ")
      ))
    }
    as.data.frame(readxl::read_excel(file_path, sheet = sheet_name), stringsAsFactors = FALSE)
  }, error = function(e) {
    message("readxl read failed: ", conditionMessage(e))
    NULL
  })

  if (!is.null(dat)) {
    return(dat)
  }

  # Fallback path: tab-delimited text mislabeled as .xlsx.
  dat_fallback <- tryCatch({
    as.data.frame(read.delim(file_path, sep = "\t", check.names = FALSE, stringsAsFactors = FALSE))
  }, error = function(e) {
    stop(sprintf(
      "Both readxl and tab-delimited fallback failed. Last error: %s",
      conditionMessage(e)
    ))
  })

  message("Fallback parser used: read.delim with tab separator.")
  dat_fallback
}

raw_data <- safe_read_data(input_file, input_sheet)
clean_data <- janitor::clean_names(raw_data)

required_cols <- c(
  "reabsorption",
  "vas_baseline", "vas_7d", "vas_6m",
  "odi_baseline", "odi_7d", "odi_6m",
  "joa_baseline", "joa_7d", "joa_6m"
)

missing_cols <- setdiff(required_cols, names(clean_data))
if (length(missing_cols) > 0) {
  stop(sprintf(
    "Required columns missing after cleaning: %s",
    paste(missing_cols, collapse = ", ")
  ))
}

analysis_data <- clean_data %>%
  dplyr::select(dplyr::all_of(required_cols)) %>%
  dplyr::mutate(reabsorption = trimws(as.character(reabsorption)))

invalid_group <- unique(analysis_data$reabsorption[!(analysis_data$reabsorption %in% c("0", "1"))])
if (length(invalid_group) > 0) {
  stop(sprintf(
    "Group column 'Reabsorption' must be 0/1 only. Invalid values: %s",
    paste(invalid_group, collapse = ", ")
  ))
}

# Convert outcomes to numeric while preserving ODI 0-1 scale.
analysis_data <- analysis_data %>%
  dplyr::mutate(
    dplyr::across(
      .cols = -reabsorption,
      .fns = ~ suppressWarnings(as.numeric(.x))
    )
  )

analysis_data <- analysis_data %>%
  dplyr::mutate(
    subject = factor(dplyr::row_number()),
    group = factor(reabsorption, levels = c("0", "1"))
  ) %>%
  dplyr::select(subject, group, dplyr::everything())

# Group counts (patients, not repeated rows).
group_counts <- analysis_data %>% dplyr::count(group, name = "n")
cat("\n=== Group sample sizes ===\n")
print(group_counts)
write.csv(group_counts, file.path(output_dir, "group_counts.csv"), row.names = FALSE)

make_long_data <- function(data, prefix, label) {
  cols <- paste0(prefix, c("_baseline", "_7d", "_6m"))

  long_df <- data %>%
    dplyr::select(subject, group, dplyr::all_of(cols)) %>%
    tidyr::pivot_longer(
      cols = dplyr::all_of(cols),
      names_to = "time",
      values_to = "outcome"
    ) %>%
    dplyr::mutate(
      time = sub(paste0("^", prefix, "_"), "", time),
      time = factor(time, levels = c("baseline", "7d", "6m")),  # ordered=FALSE: uses contr.sum per global options, consistent with Type III ANOVA
      outcome_name = label,
      outcome = as.numeric(outcome)
    ) %>%
    dplyr::select(subject, group, time, outcome_name, outcome)

  long_df
}

preview_long <- dplyr::bind_rows(
  make_long_data(analysis_data, "vas", "VAS"),
  make_long_data(analysis_data, "odi", "ODI"),
  make_long_data(analysis_data, "joa", "JOA")
)

cat("\n=== Long data preview (first 30 rows) ===\n")
print(utils::head(preview_long, 30))
write.csv(utils::head(preview_long, 30), file.path(output_dir, "data_long_preview.csv"), row.names = FALSE)

extract_type3_table <- function(anova_tbl) {
  anova_df <- data.frame(Effect = rownames(anova_tbl), anova_tbl, row.names = NULL, check.names = FALSE)

  p_col <- grep("Pr\\(>F\\)", names(anova_df), value = TRUE)
  if (length(p_col) != 1) {
    stop("Could not locate p-value column in Type III ANOVA table.")
  }

  if (!("F value" %in% names(anova_df))) {
    stop("Could not locate 'F value' column in Type III ANOVA table.")
  }

  out <- anova_df %>%
    dplyr::mutate(
      F_value = .data[["F value"]],
      p_value = .data[[p_col]]
    ) %>%
    dplyr::select(Effect, NumDF, DenDF, F_value, p_value)

  out
}

analyze_outcome <- function(data, prefix, label, out_dir, alpha_level = 0.05) {
  cat(sprintf("\n\n================ %s ================\n", label))

  long_df <- make_long_data(data, prefix, label)
  na_n <- sum(is.na(long_df$outcome))
  if (na_n > 0) {
    cat(sprintf("Note: %d missing observations for %s will be omitted by lmer.\n", na_n, label))
  }

  model <- lmerTest::lmer(
    outcome ~ time * group + (1 | subject),
    data = long_df,
    REML = TRUE,  # REML=TRUE recommended for inference with Satterthwaite F-tests (unbiased variance estimates)
    na.action = na.omit
  )

  # Model summary (console + txt)
  model_summary <- summary(model)
  print(model_summary)

  summary_file <- file.path(out_dir, paste0(label, "_model_summary.txt"))
  # Use tryCatch to guarantee sink() is always closed even if print() errors.
  sink(summary_file)
  tryCatch({
    cat(sprintf("Outcome: %s\n", label))
    cat("Formula: outcome ~ time * group + (1 | subject)\n\n")
    print(model_summary)
  }, error = function(e) {
    cat(sprintf("\n[ERROR writing model summary: %s]\n", conditionMessage(e)))
  }, finally = {
    sink()
  })

  # Type III ANOVA (Satterthwaite)
  type3 <- anova(model, type = 3, ddf = "Satterthwaite")
  type3_out <- extract_type3_table(type3)
  cat("\nType III ANOVA:\n")
  print(type3_out)
  write.csv(type3_out, file.path(out_dir, paste0(label, "_type3_anova.csv")), row.names = FALSE)

  # A) Within-group time pairwise comparisons (Holm)
  emm_within <- emmeans::emmeans(model, ~ time | group)
  pw_within <- pairs(emm_within, adjust = "holm")
  within_df <- as.data.frame(pw_within)
  within_ci <- as.data.frame(confint(pw_within, adjust = "holm"))

  within_df$lower.CL <- within_ci$lower.CL
  within_df$upper.CL <- within_ci$upper.CL

  within_out <- within_df %>%
    dplyr::select(group, contrast, estimate, SE, df, t.ratio, lower.CL, upper.CL, p.value)

  cat("\nWithin-group time pairwise (Holm):\n")
  print(within_out)
  write.csv(within_out, file.path(out_dir, paste0(label, "_within_time_pairs.csv")), row.names = FALSE)

  # B) Between-group differences at each time point (Holm)
  emm_between <- emmeans::emmeans(model, ~ group | time)
  pw_between <- pairs(emm_between, adjust = "holm")
  between_df <- as.data.frame(pw_between)
  between_ci <- as.data.frame(confint(pw_between, adjust = "holm"))

  between_df$lower.CL <- between_ci$lower.CL
  between_df$upper.CL <- between_ci$upper.CL

  between_out <- between_df %>%
    dplyr::select(time, contrast, estimate, SE, df, t.ratio, lower.CL, upper.CL, p.value)

  cat("\nBetween-group at each time (Holm):\n")
  print(between_out)
  write.csv(between_out, file.path(out_dir, paste0(label, "_between_group_at_time.csv")), row.names = FALSE)

  # Extract p-values for one-sentence manuscript conclusion.
  p_time <- type3_out$p_value[type3_out$Effect == "time"]
  p_interaction <- type3_out$p_value[type3_out$Effect %in% c("time:group", "group:time")]

  if (length(p_time) == 0) p_time <- NA_real_
  if (length(p_interaction) == 0) p_interaction <- NA_real_

  sig_time <- !is.na(p_time) && (p_time < alpha_level)
  sig_inter <- !is.na(p_interaction) && (p_interaction < alpha_level)

  sig_between_times <- character(0)
  if (sig_inter) {
    sig_between_times <- as.character(between_out$time[between_out$p.value < alpha_level])
    sig_between_times <- unique(sig_between_times)
  }

  if (sig_inter && length(sig_between_times) > 0) {
    sig_text <- paste(sig_between_times, collapse = ", ")
  } else if (sig_inter) {
    sig_text <- "none after Holm correction"
  } else {
    sig_text <- "not applicable (interaction not significant)"
  }

  one_line <- sprintf(
    "%s: Time main effect %s (p=%.4g); Time*Group interaction %s (p=%.4g); significant between-group time points: %s.",
    label,
    ifelse(sig_time, "significant", "not significant"),
    ifelse(is.na(p_time), NaN, p_time),
    ifelse(sig_inter, "significant", "not significant"),
    ifelse(is.na(p_interaction), NaN, p_interaction),
    sig_text
  )

  cat("\nManuscript-style template:\n")
  cat(one_line, "\n")

  list(
    model = model,
    type3 = type3_out,
    within = within_out,
    between = between_out,
    conclusion = one_line
  )
}

outcome_specs <- list(
  list(prefix = "vas", label = "VAS"),
  list(prefix = "odi", label = "ODI"),
  list(prefix = "joa", label = "JOA")
)

results <- list()
conclusion_lines <- character(0)
failed <- character(0)

for (spec in outcome_specs) {
  res <- tryCatch({
    analyze_outcome(
      data = analysis_data,
      prefix = spec$prefix,
      label = spec$label,
      out_dir = output_dir,
      alpha_level = alpha
    )
  }, error = function(e) {
    msg <- sprintf("%s failed: %s", spec$label, conditionMessage(e))
    cat("\nERROR:\n", msg, "\n", sep = "")
    failed <<- c(failed, msg)
    NULL
  })

  results[[spec$label]] <- res
  if (!is.null(res)) {
    conclusion_lines <- c(conclusion_lines, res$conclusion)
  }
}

if (length(failed) > 0) {
  conclusion_lines <- c(
    conclusion_lines,
    "",
    "Failures:",
    failed
  )
}

cat("\n=== Paper-style one-line conclusions ===\n")
cat(paste(conclusion_lines, collapse = "\n"), "\n")

writeLines(conclusion_lines, file.path(output_dir, "paper_style_conclusions.txt"))

cat(sprintf("\nAnalysis completed. Outputs saved to: %s\n", normalizePath(output_dir, winslash = "/", mustWork = FALSE)))
