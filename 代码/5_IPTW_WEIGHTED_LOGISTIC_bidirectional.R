MAIN_MAX_FOLLOW_UP_MONTHS <- 15
SCRIPT_VARIANT <- "bidirectional"
required_pkgs <- c("readxl", "survey", "ggplot2", "cobalt")
missing_pkgs <- required_pkgs[!sapply(required_pkgs, requireNamespace, quietly = TRUE)]
if (length(missing_pkgs) > 0) stop("Missing required packages: ", paste(missing_pkgs, collapse = ", "))
library(readxl)
library(survey)
library(ggplot2)
library(cobalt)

ensure_dir <- function(path) if (!dir.exists(path)) dir.create(path, recursive = TRUE)
norm_str <- function(x) { if (is.na(x)) return(NA_character_); y <- trimws(as.character(x)); if (!nzchar(y)) NA_character_ else y }
norm_gender <- function(x) {
  if (is.na(x)) return(NA_character_)
  s <- trimws(as.character(x))
  if (s %in% c("0","0.0","Female","female","F","f")) return("Female")
  if (s %in% c("1","1.0","Male","male","M","m")) return("Male")
  NA_character_
}
norm_age <- function(x) {
  if (is.na(x)) return(NA_character_)
  n <- suppressWarnings(as.numeric(x))
  if (!is.na(n)) {
    if (abs(n - round(n)) < 1e-9) return(as.character(as.integer(round(n))))
    y <- sprintf("%.6f", n); y <- sub("0+$", "", y); y <- sub("\\.$", "", y); return(y)
  }
  norm_str(x)
}
discover_file <- function(target) {
  script_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", script_args, value = TRUE)
  script_path <- if (length(file_arg) >= 1) normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = FALSE) else NA_character_
  script_dir <- if (!is.na(script_path) && nzchar(script_path)) dirname(script_path) else getwd()
  candidates <- normalizePath(c(file.path(script_dir, "..", "文件", target), file.path(getwd(), "文件", target), file.path(getwd(), target)), winslash = "/", mustWork = FALSE)
  hit <- candidates[file.exists(candidates)]
  if (length(hit) < 1) stop("Input file not found: ", target, "\n", paste(candidates, collapse = "\n"))
  hit[[1]]
}
convert_binary <- function(x, name) {
  if (is.logical(x)) return(as.integer(x))
  if (is.factor(x)) x <- as.character(x)
  if (is.character(x)) {
    y <- trimws(x); bad <- !(y %in% c("0", "1"))
    if (any(bad, na.rm = TRUE)) stop(name, " has non-binary values: ", paste(unique(y[bad]), collapse = ", "))
    return(as.integer(y))
  }
  if (is.numeric(x) || is.integer(x)) {
    bad <- !(x %in% c(0, 1))
    if (any(bad, na.rm = TRUE)) stop(name, " has non-binary values: ", paste(unique(x[bad]), collapse = ", "))
    return(as.integer(x))
  }
  stop(name, " unsupported type: ", class(x)[1])
}
fmt_ci <- function(lo, hi) if (any(is.na(c(lo, hi)))) NA_character_ else sprintf("[%.6f, %.6f]", lo, hi)
fmt_p <- function(p) if (is.na(p)) "" else if (p < 1e-4) "<0.0001" else formatC(p, format = "f", digits = 4)
fmt_n_pct <- function(n, d) if (is.na(d) || d <= 0) "NA" else sprintf("%d (%.1f)", n, 100 * n / d)
fmt_w_pct <- function(w, d) if (is.na(d) || d <= 0) "NA" else sprintf("%.1f (%.1f)", w, 100 * w / d)
fmt_mean_sd <- function(m, s) if (is.na(m) || is.na(s)) "NA" else sprintf("%.2f (%.2f)", m, s)
smd_cont <- function(m0, sd0, m1, sd1) { if (any(is.na(c(m0, sd0, m1, sd1)))) return(NA_real_); p <- sqrt((sd0^2 + sd1^2)/2); if (p == 0) 0 else abs(m1 - m0) / p }
smd_cat <- function(p0, p1) { if (any(is.na(c(p0, p1)))) return(NA_real_); pbar <- (p0 + p1)/2; d <- sum(pbar*(1-pbar)); if (d == 0) 0 else sqrt(sum((p1-p0)^2)/d) }
weight_diag <- function(w) list(summary = summary(w), sd = sd(w, na.rm = TRUE), quantile = quantile(w, c(0.95, 0.99, 0.995), na.rm = TRUE), max = max(w, na.rm = TRUE), ess = (sum(w)^2)/sum(w^2))
pick_adj <- function(df) {
  nm <- names(df); pr <- c("Diff.Adj", "SMD.Adj", "Adj"); hit <- pr[pr %in% nm]
  if (length(hit) > 0) return(hit[1])
  hit <- grep("(?i)(diff|smd).*adj|adj.*(diff|smd)", nm, value = TRUE, perl = TRUE)
  if (length(hit) > 0) return(hit[1])
  NA_character_
}
summarize_adj <- function(balance_df, covars, adj_col) {
  if (is.na(adj_col) || !("var_name" %in% names(balance_df))) return(data.frame(covariate = character(0), max_abs_adj_smd = numeric(0)))
  vn <- as.character(balance_df$var_name); vals <- suppressWarnings(as.numeric(balance_df[[adj_col]]))
  do.call(rbind, lapply(covars, function(v) {
    idx <- which(vn == v | startsWith(vn, paste0(v, "_")))
    mx <- if (length(idx) < 1 || all(is.na(vals[idx]))) NA_real_ else max(abs(vals[idx]), na.rm = TRUE)
    data.frame(covariate = v, max_abs_adj_smd = mx)
  }))
}
extract_glm <- function(fit, pattern = "Rupture") {
  s <- summary(fit)$coefficients; rn <- rownames(s); m <- grep(pattern, rn, value = TRUE); term <- if (length(m) > 0) m[1] else rn[1]
  b <- s[term, "Estimate"]; se <- s[term, "Std. Error"]; p <- 2 * (1 - pnorm(abs(b / se)))
  list(OR = exp(b), CI_low = exp(b - 1.96 * se), CI_high = exp(b + 1.96 * se), p = p)
}
extract_svy <- function(fit, pattern = "Rupture") {
  s <- summary(fit)$coefficients; rn <- rownames(s); m <- grep(pattern, rn, value = TRUE); term <- if (length(m) > 0) m[1] else rn[1]
  b <- s[term, "Estimate"]; se <- s[term, "Std. Error"]; p <- 2 * (1 - pnorm(abs(b / se)))
  list(OR = exp(b), CI_low = exp(b - 1.96 * se), CI_high = exp(b + 1.96 * se), p = p)
}
get_w_stats <- function(design_obj, var, group_level) {
  sub <- subset(design_obj, Rupture == group_level); form <- as.formula(paste0("~", var))
  m <- tryCatch(as.numeric(svymean(form, sub, na.rm = TRUE)), error = function(e) NA_real_)
  v <- tryCatch(as.numeric(svyvar(form, sub, na.rm = TRUE)), error = function(e) NA_real_)
  c(mean = m, sd = sqrt(v))
}
write_skip <- function(root_dir, setting, reason) {
  ensure_dir(root_dir)
  writeLines(c(paste0("Analysis skipped: ", setting), paste0("Reason: ", reason), paste0("Script_Variant: ", SCRIPT_VARIANT)), file.path(root_dir, "SKIPPED.txt"))
}

load_bidirectional <- function() {
  rf <- discover_file("Retrospective data.xlsx")
  pf <- discover_file("Prospective data.xlsx")
  retro <- as.data.frame(read_excel(rf, sheet = "Train"), stringsAsFactors = FALSE, check.names = FALSE)
  pros <- as.data.frame(read_excel(pf, sheet = "Train_Pors"), stringsAsFactors = FALSE, check.names = FALSE)
  if (!identical(names(retro), names(pros))) stop("Retrospective and prospective columns are not identical.")
  std <- function(df) { for (nm in names(df)) if (is.character(df[[nm]]) || is.factor(df[[nm]])) { z <- trimws(as.character(df[[nm]])); z[z == ""] <- NA_character_; df[[nm]] <- z }; df }
  retro <- std(retro); pros <- std(pros)
  retro$Cohort <- "Retrospective"; pros$Cohort <- "Prospective"
  retro$Source_File <- basename(rf); pros$Source_File <- basename(pf)
  retro$Source_Sheet <- "Train"; pros$Source_Sheet <- "Train_Pors"
  retro$Unified_ID <- paste0(retro$Cohort, "_", retro$ID); pros$Unified_ID <- paste0(pros$Cohort, "_", pros$ID)
  rk <- ifelse(is.na(retro$Name) | is.na(retro$Age) | is.na(retro$Gender), NA_character_, paste(vapply(retro$Name, norm_str, character(1)), vapply(retro$Age, norm_age, character(1)), vapply(retro$Gender, norm_gender, character(1)), sep = "|"))
  pk <- ifelse(is.na(pros$Name) | is.na(pros$Age) | is.na(pros$Gender), NA_character_, paste(vapply(pros$Name, norm_str, character(1)), vapply(pros$Age, norm_age, character(1)), vapply(pros$Gender, norm_gender, character(1)), sep = "|"))
  overlap <- intersect(na.omit(unique(rk)), na.omit(unique(pk)))
  retro$Overlap_Key <- rk; pros$Overlap_Key <- pk
  retro_overlap <- retro[!is.na(retro$Overlap_Key) & retro$Overlap_Key %in% overlap, , drop = FALSE]
  pros_overlap <- pros[!is.na(pros$Overlap_Key) & pros$Overlap_Key %in% overlap, , drop = FALSE]
  matches <- merge(retro_overlap[, c("Unified_ID", "ID", "Name", "Age", "Gender", "Overlap_Key")], pros_overlap[, c("Unified_ID", "ID", "Name", "Age", "Gender", "Overlap_Key")], by = "Overlap_Key", suffixes = c("_retro", "_pros"))
  # Keep same-person records across cohorts as distinct longitudinal observations.
  merged <- rbind(retro[, setdiff(names(retro), "Overlap_Key"), drop = FALSE], pros[, setdiff(names(pros), "Overlap_Key"), drop = FALSE])
  dropped <- retro_overlap[0, setdiff(names(retro_overlap), "Overlap_Key"), drop = FALSE]
  list(merged = merged, matches = matches, dropped = dropped, overlap_n = length(overlap), files = c(rf = rf, pf = pf))
}

write_merge_audit <- function(root_dir, obj) {
  ensure_dir(root_dir)
  write.csv(obj$matches, file.path(root_dir, "overlap_matches.csv"), row.names = FALSE, fileEncoding = "UTF-8")
  write.csv(obj$dropped, file.path(root_dir, "dropped_retrospective_due_to_overlap.csv"), row.names = FALSE, fileEncoding = "UTF-8")
  write.csv(obj$merged, file.path(root_dir, "merged_analysis_dataset.csv"), row.names = FALSE, fileEncoding = "UTF-8")
  lines <- c(
    "Bidirectional merge summary",
    paste0("Retrospective file: ", obj$files[["rf"]]),
    paste0("Prospective file: ", obj$files[["pf"]]),
    paste0("Overlap keys detected: ", obj$overlap_n),
    "Retrospective rows dropped due to overlap: 0 (overlaps retained as distinct cohort-specific observations)",
    paste0("Merged rows with overlap retained: ", nrow(obj$merged)),
    capture.output(print(table(obj$merged$Cohort, useNA = "ifany")))
  )
  writeLines(lines, file.path(root_dir, "merge_summary.txt"))
}

run_case <- function(dat_input, analysis_root, analysis_setting, followup_limit = NA_real_) {
  ensure_dir(analysis_root); overlap_dir <- file.path(analysis_root, "overlap_balance"); outcome_dir <- file.path(analysis_root, "outcomes"); ensure_dir(overlap_dir); ensure_dir(outcome_dir)
  dat <- dat_input; n0 <- nrow(dat)
  if (!is.na(followup_limit)) { mo <- suppressWarnings(as.numeric(dat$Months_of_Review)); dat <- dat[!is.na(mo) & mo <= followup_limit, , drop = FALSE]; filter_desc <- sprintf("Months_of_Review <= %d", as.integer(followup_limit)) } else filter_desc <- "All follow-up times"
  n1 <- nrow(dat); if (n1 == 0) stop("No rows after follow-up filter.")
  dat <- dat[!is.na(dat$Reabsorption), , drop = FALSE]; n2 <- nrow(dat); if (n2 == 0) stop("No rows after Reabsorption filter.")
  dat$Reabsorption <- convert_binary(dat$Reabsorption, "Reabsorption"); if (length(unique(dat$Reabsorption)) < 2) stop("Reabsorption has no variation.")
  kom <- suppressWarnings(as.numeric(as.character(dat$Komori))); rup <- ifelse(kom == 1, 0, ifelse(kom %in% c(2,3,4), 1, NA))
  dat <- dat[!is.na(rup), , drop = FALSE]; n3 <- nrow(dat); if (n3 == 0) stop("No rows after Rupture definition.")
  dat$Rupture <- factor(rup[!is.na(rup)], levels = c(0, 1), labels = c("Contained (Komori 1)", "Non-contained (Komori 2-4)")); if (length(unique(dat$Rupture)) < 2) stop("Rupture has no variation.")
  dat$Gender <- vapply(dat$Gender, norm_gender, character(1))
  be <- as.character(dat$Bull_eye); be[is.na(be) | trimws(be) == ""] <- "NA"; lv <- unique(be); lv <- c(sort(lv[lv != "NA"]), "NA"); dat$Bull_eye <- factor(be, levels = lv)
  # Cohort is a study-design variable, not a clinical confounder.
  # Excluding it from ps_covars keeps the PS model clinically interpretable;
  # Cohort will instead be forced into the doubly-robust outcome model below.
  exclude_cols <- c("ID","Name","Unified_ID","Source_File","Source_Sheet",
                    "Absorption_type","Last_volume","Absorption_rate",
                    "Months_of_Review","Komori","Reabsorption","Rupture",
                    "Cohort")
  ps_covars <- setdiff(names(dat), exclude_cols)
  ps_covars <- ps_covars[sapply(ps_covars, function(v) length(unique(na.omit(as.character(dat[[v]])))) >= 2)]
  if (length(ps_covars) < 1) stop("No covariates available for PS model.")
  # "Cohort" removed from factor_vars: it is excluded from ps_covars (see exclude_cols above).
  factor_vars <- c("Gender","Herniated_Level","Iwabuchi","Modic","Spinal_canal_stenosis","Bull_eye")
  numeric_vars <- c("Age","SS","Upper_VB_Posterior_Height_CM","Lower_VB_Posterior_Height_CM","RSI","DHI","Initial_volume")
  ordered_vars <- c("Pfirrmann","MSU")
  for (v in ps_covars) {
    if (v %in% factor_vars) dat[[v]] <- as.factor(dat[[v]]) else if (v %in% ordered_vars) dat[[v]] <- as.ordered(dat[[v]]) else if (v %in% numeric_vars) dat[[v]] <- suppressWarnings(as.numeric(dat[[v]])) else if (!is.numeric(dat[[v]]) && !is.factor(dat[[v]]) && !is.ordered(dat[[v]])) dat[[v]] <- as.factor(dat[[v]])
  }
  cc <- complete.cases(dat[, ps_covars, drop = FALSE]); dat <- dat[cc, , drop = FALSE]; n4 <- nrow(dat); if (n4 == 0) stop("No complete cases for PS covariates.")
  if (length(unique(dat$Rupture)) < 2) stop("Rupture lost variation after complete-case filtering.")
  if (length(unique(dat$Reabsorption)) < 2) stop("Reabsorption lost variation after complete-case filtering.")
  ps_formula <- as.formula(paste("Rupture ~", paste(ps_covars, collapse = " + ")))
  ps_model <- glm(ps_formula, data = dat, family = binomial())
  ps <- pmin(pmax(predict(ps_model, type = "response"), 1e-6), 1 - 1e-6)
  dat$ps <- ps; pt <- mean(dat$Rupture == levels(dat$Rupture)[2]); sw <- ifelse(dat$Rupture == levels(dat$Rupture)[2], pt / ps, (1 - pt)/(1 - ps)); dat$sw <- sw
  if (is.finite(max(sw, na.rm = TRUE)) && is.finite(sd(sw, na.rm = TRUE)) && max(sw, na.rm = TRUE) <= 10 && sd(sw, na.rm = TRUE) <= 2) { w_main <- sw; decision <- "No trimming" } else { q01 <- quantile(sw, 0.01, na.rm = TRUE); q99 <- quantile(sw, 0.99, na.rm = TRUE); w_main <- pmin(pmax(sw, q01), q99); decision <- "Trim 1%-99%" }
  dat$w_main <- w_main; followup_label <- if (!is.na(followup_limit)) sprintf("<=%d months", as.integer(followup_limit)) else "No limit"
  p1 <- ggplot(dat, aes(x = ps, fill = Rupture)) + geom_density(alpha = 0.35) + labs(title = paste("Propensity score overlap (unweighted)", analysis_setting), caption = SCRIPT_VARIANT) + theme_bw(base_size = 11) + theme(legend.position = "bottom")
  ggsave(file.path(overlap_dir, "PS_overlap_density_unweighted.tiff"), p1, width = 7, height = 5, units = "in", dpi = 300, compression = "lzw")
  p2 <- ggplot(dat, aes(x = ps, fill = Rupture, weight = w_main)) + geom_density(alpha = 0.35) + labs(title = paste("Propensity score overlap (weighted)", analysis_setting), caption = SCRIPT_VARIANT) + theme_bw(base_size = 11) + theme(legend.position = "bottom")
  ggsave(file.path(overlap_dir, "PS_overlap_density_weighted.tiff"), p2, width = 7, height = 5, units = "in", dpi = 300, compression = "lzw")
  bal <- cobalt::bal.tab(ps_formula, data = dat, weights = dat$w_main, method = "weighting", estimand = "ATE", un = TRUE)
  love <- cobalt::love.plot(bal, abs = TRUE, threshold = 0.1, var.order = "unadjusted", title = paste("Covariate balance", analysis_setting), return.plot = TRUE) + coord_cartesian(xlim = c(0, 0.30)) + theme_bw(base_size = 11) + theme(legend.position = "bottom")
  tryCatch(ggsave(file.path(overlap_dir, "Love_plot_SMD_unadj_vs_adj.tiff"), love, width = 8, height = 10, units = "in", dpi = 300, compression = "lzw"), error = function(e) { tiff(file.path(overlap_dir, "Love_plot_SMD_unadj_vs_adj.tiff"), width = 8, height = 10, units = "in", res = 300, compression = "lzw"); print(love); dev.off() })
  bal_tbl <- if (!is.null(bal$Balance)) bal$Balance else { s <- tryCatch(summary(bal), error = function(e) NULL); if (!is.null(s) && !is.null(s$Balance)) s$Balance else NULL }
  if (is.null(bal_tbl)) stop("Cannot find balance table in cobalt bal.tab object.")
  bal_df <- as.data.frame(bal_tbl); bal_df$var_name <- rownames(bal_df); rownames(bal_df) <- NULL
  keep_cols <- unique(unlist(lapply(c("^Diff","^SMD","Un$","Adj$","M\\.0","M\\.1","V\\.Ratio"), function(p) grep(p, names(bal_df), value = TRUE))))
  write.csv(if (length(keep_cols) > 0) bal_df[, c("var_name", keep_cols), drop = FALSE] else bal_df, file.path(outcome_dir, "eTable2_balance_levelwise.csv"), row.names = FALSE, fileEncoding = "UTF-8")
  adj_col <- pick_adj(bal_df); adj_sum <- summarize_adj(bal_df, ps_covars, adj_col); write.csv(adj_sum, file.path(outcome_dir, "DR_residual_imbalance_screening.csv"), row.names = FALSE, fileEncoding = "UTF-8")
  design <- svydesign(ids = ~1, data = dat, weights = ~w_main)
  res_unadj <- extract_glm(glm(Reabsorption ~ Rupture, data = dat, family = binomial()))
  # Multivariable model: adjust ps_covars (clinical variables) + Cohort.
  # Cohort is added here to keep the adjustment strategy consistent with the
  # doubly-robust model — both treat Cohort as a design variable to be
  # controlled in the outcome model rather than balanced via IPTW weights.
  multi_covars <- ps_covars
  if ("Cohort" %in% names(dat) && length(unique(na.omit(as.character(dat[["Cohort"]])))) >= 2) {
    multi_covars <- unique(c("Cohort", multi_covars))
  }
  res_multi <- extract_glm(glm(
    as.formula(paste("Reabsorption ~ Rupture +", paste(multi_covars, collapse = " + "))),
    data = dat, family = binomial()
  ))
  res_main <- extract_svy(svyglm(Reabsorption ~ Rupture, design = design, family = quasibinomial()))
  dr_vars <- if (!is.na(adj_col) && nrow(adj_sum) > 0) adj_sum$covariate[is.finite(adj_sum$max_abs_adj_smd) & adj_sum$max_abs_adj_smd > 0.1] else character(0)
  dr_vars <- dr_vars[dr_vars %in% names(dat) & sapply(dr_vars, function(v) !all(is.na(dat[[v]])))]
  # Force Cohort into the doubly-robust outcome model to account for
  # systematic between-cohort differences (e.g. enrolment period, follow-up
  # strategy) that were intentionally kept out of the PS model.
  if ("Cohort" %in% names(dat) && length(unique(na.omit(as.character(dat[["Cohort"]])))) >= 2) {
    dr_vars <- unique(c("Cohort", dr_vars))
  }
  res_dr <- list(OR = NA_real_, CI_low = NA_real_, CI_high = NA_real_, p = NA_real_)
  if (length(dr_vars) > 0) res_dr <- extract_svy(svyglm(as.formula(paste("Reabsorption ~ Rupture +", paste(dr_vars, collapse = " + "))), design = design, family = quasibinomial()))
  writeLines(c(paste0("Script_Variant: ", SCRIPT_VARIANT), paste0("Analysis_Setting: ", analysis_setting), paste0("Followup_Window: ", followup_label), paste0("Decision: ", decision), sprintf("OR = %.6f", res_main$OR), sprintf("95%% CI = [%.6f, %.6f]", res_main$CI_low, res_main$CI_high), sprintf("p = %.6g", res_main$p)), file.path(outcome_dir, "IPTW_weighted_logistic_results.txt"))
  e3 <- if (length(dr_vars) > 0) data.frame(Script_Variant = SCRIPT_VARIANT, Analysis_Setting = analysis_setting, Followup_Window = followup_label, Model = c("IPTW-only","Doubly robust (IPTW + outcome adjustment)"), TrimmingDecision = c(decision, decision), OutcomeAdjustment = c("None (Reabsorption ~ Rupture)", paste0("Adjusted: ", paste(dr_vars, collapse = " + "))), OR = c(res_main$OR, res_dr$OR), CI_low = c(res_main$CI_low, res_dr$CI_low), CI_high = c(res_main$CI_high, res_dr$CI_high), p_value = c(res_main$p, res_dr$p), stringsAsFactors = FALSE) else data.frame(Script_Variant = SCRIPT_VARIANT, Analysis_Setting = analysis_setting, Followup_Window = followup_label, Model = "IPTW-only", TrimmingDecision = decision, OutcomeAdjustment = "None (Reabsorption ~ Rupture)", OR = res_main$OR, CI_low = res_main$CI_low, CI_high = res_main$CI_high, p_value = res_main$p, stringsAsFactors = FALSE)
  write.csv(e3, file.path(outcome_dir, "eTable3_IPTW_weighted_logistic_DR.csv"), row.names = FALSE, fileEncoding = "UTF-8")
  t4 <- data.frame(Script_Variant = SCRIPT_VARIANT, Analysis_Setting = analysis_setting, Followup_Window = followup_label, Model = c("Unadjusted","Multivariable","IPTW","Doubly robust"), Method = c("Standard logistic regression (glm, unweighted)", "Standard logistic regression (glm, unweighted, adjusted by ps_covars + Cohort)", "IPTW-weighted logistic regression (svyglm)", "Doubly robust: IPTW-weighted logistic + outcome adjustment (svyglm)"), OR = c(res_unadj$OR, res_multi$OR, res_main$OR, res_dr$OR), check.names = FALSE, stringsAsFactors = FALSE)
  t4[["95% CI"]] <- c(fmt_ci(res_unadj$CI_low, res_unadj$CI_high), fmt_ci(res_multi$CI_low, res_multi$CI_high), fmt_ci(res_main$CI_low, res_main$CI_high), fmt_ci(res_dr$CI_low, res_dr$CI_high)); t4$P_value <- signif(c(res_unadj$p, res_multi$p, res_main$p, res_dr$p), 6)
  write.csv(t4, file.path(outcome_dir, "Table4_logistic_models_comparison.csv"), row.names = FALSE, fileEncoding = "UTF-8")
  g0 <- levels(dat$Rupture)[1]; g1 <- levels(dat$Rupture)[2]; n_non <- sum(dat$Rupture == g0); n_rup <- sum(dat$Rupture == g1); w_non <- sum(dat$w_main[dat$Rupture == g0]); w_rup <- sum(dat$w_main[dat$Rupture == g1])
  rows <- list(data.frame(Characteristic = "n", Before_Non = as.character(n_non), Before_Rup = as.character(n_rup), Before_p = "", Before_SMD = "", After_Non = sprintf("%.1f", w_non), After_Rup = sprintf("%.1f", w_rup), After_p = "", After_SMD = "", stringsAsFactors = FALSE))
  for (v in ps_covars) {
    if (v %in% factor_vars || v %in% ordered_vars || is.factor(dat[[v]]) || is.ordered(dat[[v]])) {
      x <- if (is.factor(dat[[v]])) dat[[v]] else as.factor(dat[[v]]); lvls <- levels(x); tbl <- table(x, dat$Rupture); pb <- if (all(dim(tbl) == c(2, 2)) && any(suppressWarnings(chisq.test(tbl)$expected) < 5)) tryCatch(fisher.test(tbl)$p.value, error = function(e) NA_real_) else tryCatch(chisq.test(tbl)$p.value, error = function(e) NA_real_); pa <- tryCatch(svychisq(as.formula(paste0("~", v, " + Rupture")), design)$p.value, error = function(e) NA_real_); p0 <- p1 <- numeric(length(lvls)); wp0 <- wp1 <- numeric(length(lvls))
      for (i in seq_along(lvls)) { lvl <- lvls[i]; n_g0 <- sum(x == lvl & dat$Rupture == g0); n_g1 <- sum(x == lvl & dat$Rupture == g1); p0[i] <- if (n_non > 0) n_g0 / n_non else NA_real_; p1[i] <- if (n_rup > 0) n_g1 / n_rup else NA_real_; w0 <- sum(dat$w_main[x == lvl & dat$Rupture == g0]); w1 <- sum(dat$w_main[x == lvl & dat$Rupture == g1]); wp0[i] <- if (w_non > 0) w0 / w_non else NA_real_; wp1[i] <- if (w_rup > 0) w1 / w_rup else NA_real_ }
      rows[[length(rows)+1]] <- data.frame(Characteristic = paste0(v, " (%)"), Before_Non = "", Before_Rup = "", Before_p = fmt_p(pb), Before_SMD = ifelse(is.na(smd_cat(p0, p1)), "", formatC(smd_cat(p0, p1), format = "f", digits = 3)), After_Non = "", After_Rup = "", After_p = fmt_p(pa), After_SMD = ifelse(is.na(smd_cat(wp0, wp1)), "", formatC(smd_cat(wp0, wp1), format = "f", digits = 3)), stringsAsFactors = FALSE)
      for (lvl in lvls) { n_g0 <- sum(x == lvl & dat$Rupture == g0); n_g1 <- sum(x == lvl & dat$Rupture == g1); w0 <- sum(dat$w_main[x == lvl & dat$Rupture == g0]); w1 <- sum(dat$w_main[x == lvl & dat$Rupture == g1]); rows[[length(rows)+1]] <- data.frame(Characteristic = paste0("  ", lvl), Before_Non = fmt_n_pct(n_g0, n_non), Before_Rup = fmt_n_pct(n_g1, n_rup), Before_p = "", Before_SMD = "", After_Non = fmt_w_pct(w0, w_non), After_Rup = fmt_w_pct(w1, w_rup), After_p = "", After_SMD = "", stringsAsFactors = FALSE) }
    } else {
      x0 <- suppressWarnings(as.numeric(dat[dat$Rupture == g0, v])); x1 <- suppressWarnings(as.numeric(dat[dat$Rupture == g1, v])); ws0 <- get_w_stats(design, v, g0); ws1 <- get_w_stats(design, v, g1); pb <- tryCatch(t.test(suppressWarnings(as.numeric(dat[[v]])) ~ dat$Rupture)$p.value, error = function(e) NA_real_); pa <- tryCatch(svyttest(as.formula(paste0(v, " ~ Rupture")), design)$p.value, error = function(e) NA_real_)
      rows[[length(rows)+1]] <- data.frame(Characteristic = paste0(v, " (mean (SD))"), Before_Non = fmt_mean_sd(mean(x0, na.rm = TRUE), sd(x0, na.rm = TRUE)), Before_Rup = fmt_mean_sd(mean(x1, na.rm = TRUE), sd(x1, na.rm = TRUE)), Before_p = fmt_p(pb), Before_SMD = ifelse(is.na(smd_cont(mean(x0, na.rm = TRUE), sd(x0, na.rm = TRUE), mean(x1, na.rm = TRUE), sd(x1, na.rm = TRUE))), "", formatC(smd_cont(mean(x0, na.rm = TRUE), sd(x0, na.rm = TRUE), mean(x1, na.rm = TRUE), sd(x1, na.rm = TRUE)), format = "f", digits = 3)), After_Non = fmt_mean_sd(ws0["mean"], ws0["sd"]), After_Rup = fmt_mean_sd(ws1["mean"], ws1["sd"]), After_p = fmt_p(pa), After_SMD = ifelse(is.na(smd_cont(ws0["mean"], ws0["sd"], ws1["mean"], ws1["sd"])), "", formatC(smd_cont(ws0["mean"], ws0["sd"], ws1["mean"], ws1["sd"]), format = "f", digits = 3)), stringsAsFactors = FALSE)
    }
  }
  e1 <- do.call(rbind, rows); e1 <- cbind(Script_Variant = SCRIPT_VARIANT, Analysis_Setting = analysis_setting, Followup_Window = followup_label, e1, stringsAsFactors = FALSE)
  colnames(e1)[4:12] <- c("Characteristic", paste0("Before IPTW: ", g0), paste0("Before IPTW: ", g1), "Before p-value", "Before SMD", paste0("After IPTW: ", g0), paste0("After IPTW: ", g1), "After p-value", "After SMD")
  write.csv(e1, file.path(outcome_dir, "eTable1_IPTW.csv"), row.names = FALSE, fileEncoding = "UTF-8")
  write.csv(dat, file.path(outcome_dir, "IPTW_analysis_dataset.csv"), row.names = FALSE, fileEncoding = "UTF-8")
  wd1 <- weight_diag(sw); wd2 <- weight_diag(w_main)
  lines <- c("=================================================================", "IPTW Weighted Logistic Analysis: Bidirectional cohort", "=================================================================", paste0("Script_Variant: ", SCRIPT_VARIANT), paste0("Analysis_Setting: ", analysis_setting), paste0("Followup_Window: ", followup_label), "", "Sample flow:", paste0("  Initial n = ", n0), sprintf("  After %s = %d", filter_desc, n1), paste0("  After Reabsorption non-missing = ", n2), paste0("  After Komori-based Rupture definition = ", n3), paste0("  After complete PS covariates = ", n4), paste0("Decision: ", decision), "", "Cohort counts in analysis dataset:", capture.output(print(table(dat$Cohort, useNA = "ifany"))), "", "sw summary:", capture.output(print(wd1$summary)), paste0("sw sd: ", wd1$sd), paste0("sw quantile 0.95/0.99/0.995: ", paste(wd1$quantile, collapse = ", ")), paste0("sw max: ", wd1$max), paste0("sw ESS: ", wd1$ess), "w_main summary:", capture.output(print(wd2$summary)), paste0("w_main sd: ", wd2$sd), paste0("w_main quantile 0.95/0.99/0.995: ", paste(wd2$quantile, collapse = ", ")), paste0("w_main max: ", wd2$max), paste0("w_main ESS: ", wd2$ess), "", sprintf("Primary IPTW OR=%.4f, 95%%CI [%.4f, %.4f], p=%.4g", res_main$OR, res_main$CI_low, res_main$CI_high, res_main$p))
  writeLines(lines, file.path(outcome_dir, "IPTW_weighted_logistic_summary.txt"))
}

run_case_safe <- function(dat, analysis_root, analysis_setting, followup_limit = NA_real_, fail_hard = FALSE) {
  tryCatch(run_case(dat, analysis_root, analysis_setting, followup_limit), error = function(e) { write_skip(analysis_root, analysis_setting, conditionMessage(e)); if (isTRUE(fail_hard)) stop(e) })
}

results_root <- file.path(getwd(), "Results")
run_id <- format(Sys.time(), "%Y%m%d_%H%M%S")
run_root <- file.path(results_root, "Manuscript_v2", paste0("run_", run_id))
iptw_root <- file.path(run_root, "07_IPTW_WEIGHTED_LOGISTIC_bidirectional")
ensure_dir(iptw_root)
obj <- load_bidirectional()
write_merge_audit(file.path(iptw_root, "merge_audit"), obj)
run_case_safe(obj$merged, file.path(iptw_root, "main_pooled_15m"), "main_pooled_15m", MAIN_MAX_FOLLOW_UP_MONTHS, TRUE)
run_case_safe(obj$merged, file.path(iptw_root, "sensitivity_all_followup"), "sensitivity_all_followup", NA_real_, TRUE)
ensure_dir(file.path(iptw_root, "sensitivity_by_cohort"))
for (cohort_name in c("Retrospective", "Prospective")) {
  run_case_safe(obj$merged[obj$merged$Cohort == cohort_name, , drop = FALSE], file.path(iptw_root, "sensitivity_by_cohort", paste0(cohort_name, "_15m")), paste0("sensitivity_by_cohort_", cohort_name, "_15m"), MAIN_MAX_FOLLOW_UP_MONTHS, FALSE)
}
