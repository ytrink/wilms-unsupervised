

# ============================================================
# Script: figure3_fastTopics_wilms.R
#
# Purpose (Figure 3):
#   - Run multinomial topic models (fastTopics) on a gene expression matrix
#   - Fit models for k = 2:10
#   - Save fitted models and their L/F matrices
#   - Plot log-likelihood trajectories for convergence diagnostics
#   - (Optionally) run differential count analysis to identify topic-associated genes
#
# Dependencies:
#   - fastTopics
#   - ggplot2
#
# 
# ============================================================

suppressPackageStartupMessages({
  library(fastTopics)
  library(ggplot2)
})

# ------------------------------------------------------------
# Helper: run fastTopics for k = 2:10 on a counts matrix
# ------------------------------------------------------------

fastTopics.wilms <- function(counts.filepath,
                             output.filepath,
                             times.directory,
                             decomp.filepath,
                             k_min = 2,
                             k_max = 10,
                             numiter.main = 100,
                             numiter.refine = 100,
                             seed = 1) {
  # counts.filepath  : CSV with genes as rows and samples as columns
  # output.filepath  : directory to save fitted models (fit_k.RData)
  # times.directory  : directory to save runtimes per k
  # decomp.filepath  : directory to save L (omega) and F (theta) matrices
  # k_min, k_max     : range of topic numbers to fit
  # numiter.main     : main iterations in fastTopics
  # numiter.refine   : refinement iterations in fastTopics
  # seed             : random seed for reproducibility
  
  # Ensure directories exist
  if (!dir.exists(output.filepath)) {
    dir.create(output.filepath, recursive = TRUE)
  }
  if (!dir.exists(decomp.filepath)) {
    dir.create(decomp.filepath, recursive = TRUE)
  }
  if (!dir.exists(times.directory)) {
    dir.create(times.directory, recursive = TRUE)
  }
  
  # Load counts
  counts <- read.csv(counts.filepath, row.names = 1, check.names = FALSE)
  counts <- as.matrix(counts)
  
  # fastTopics expects samples as rows, features as columns
  counts.T <- t(counts)
  
  runtimes <- numeric(length = k_max - k_min + 1)
  names(runtimes) <- as.character(k_min:k_max)
  
  for (k in k_min:k_max) {
    message("Fitting fastTopics model with k = ", k, " ...")
    
    fit_path   <- file.path(output.filepath, paste0("fit_", k, ".RData"))
    omega_path <- file.path(decomp.filepath, paste0("omega_", k, ".csv"))
    theta_path <- file.path(decomp.filepath, paste0("theta_", k, ".csv"))
    
    time.start <- Sys.time()
    set.seed(seed)
    fit.wilms <- fit_topic_model(
      counts.T,
      k              = k,
      numiter.main   = numiter.main,
      numiter.refine = numiter.refine
    )
    time.end <- Sys.time()
    
    runtimes[as.character(k)] <- as.numeric(difftime(time.end, time.start, units = "mins"))
    
    # Save full fit object
    save(fit.wilms, file = fit_path)
    
    # Save L (omega) and F (theta) matrices separately
    omega <- fit.wilms$L
    theta <- fit.wilms$F
    
    write.csv(omega, file = omega_path)
    write.csv(theta, file = theta_path)
  }
  
  # Save runtimes
  times_df <- data.frame(
    k               = as.integer(names(runtimes)),
    runtime_minutes = as.numeric(runtimes)
  )
  times_file <- file.path(times.directory, "times_k.csv")
  write.csv(times_df, file = times_file, row.names = FALSE)
  
  invisible(times_df)
}

# ------------------------------------------------------------
# Helper: plot log-likelihood trajectories for each fitted model
# ------------------------------------------------------------

plot_loglik <- function(fit.directory, output.directory) {
  # fit.directory   : directory containing fit_k.RData files
  # output.directory: base directory to save plots into
  #
  # Produces:
  #   - value/loglik_fit_k.tiff      : log-likelihood per iteration
  #   - difference/loglikdiff_fit_k.tiff : change in log-likelihood per iteration
  
  if (!dir.exists(output.directory)) {
    dir.create(output.directory, recursive = TRUE)
  }
  
  value_dir <- file.path(output.directory, "value")
  diff_dir  <- file.path(output.directory, "difference")
  
  if (!dir.exists(value_dir)) {
    dir.create(value_dir, recursive = TRUE)
  }
  if (!dir.exists(diff_dir)) {
    dir.create(diff_dir, recursive = TRUE)
  }
  
  # List only the model files
  fit_files <- list.files(
    fit.directory,
    pattern   = "^fit_.*\\.RData$",
    full.names = TRUE
  )
  
  if (length(fit_files) == 0) {
    stop("No fit_*.RData files found in fit.directory: ", fit.directory)
  }
  
  plot_list.values <- vector("list", length(fit_files))
  plot_list.diff   <- vector("list", length(fit_files))
  
  for (i in seq_along(fit_files)) {
    fit_file <- fit_files[i]
    load(fit_file)  # loads 'fit.wilms'
    
    if (!exists("fit.wilms")) {
      stop("File ", fit_file, " did not contain object 'fit.wilms'.")
    }
    
    prog <- as.data.frame(fit.wilms$progress)
    n_iter <- nrow(prog)
    
    # Derive k label from file name
    k_label <- sub("^fit_(\\d+)\\.RData$", "\\1", basename(fit_file))
    
    # Log-likelihood values
    p_val <- ggplot(prog, aes(x = seq_len(n_iter), y = loglik)) +
      geom_line() +
      ggtitle(paste0("Log-likelihood (k = ", k_label, ")")) +
      xlab("Iteration") +
      ylab("Log-likelihood")
    
    plot_list.values[[i]] <- p_val
    
    # Differences in log-likelihood
    ll <- prog$loglik
    if (length(ll) > 1) {
      ll_diff <- ll[-1] - ll[-length(ll)]
      diff_df <- data.frame(iter = seq_len(length(ll_diff)), ll_diff = ll_diff)
      
      p_diff <- ggplot(diff_df, aes(x = iter, y = ll_diff)) +
        geom_line() +
        ggtitle(paste0("Δ Log-likelihood (k = ", k_label, ")")) +
        xlab("Iteration") +
        ylab("Change in log-likelihood")
      
      plot_list.diff[[i]] <- p_diff
    } else {
      plot_list.diff[[i]] <- NULL
    }
  }
  
  # Save plots
  for (i in seq_along(fit_files)) {
    base_name <- sub("\\.RData$", "", basename(fit_files[i]))
    
    # Value plot
    if (!is.null(plot_list.values[[i]])) {
      out_file_val <- file.path(value_dir, paste0(base_name, "_loglik.tiff"))
      tiff(out_file_val)
      print(plot_list.values[[i]])
      dev.off()
    }
    
    # Difference plot
    if (!is.null(plot_list.diff[[i]])) {
      out_file_diff <- file.path(diff_dir, paste0(base_name, "_loglikdiff.tiff"))
      tiff(out_file_diff)
      print(plot_list.diff[[i]])
      dev.off()
    }
  }
  
  invisible(NULL)
}

# ------------------------------------------------------------
# Example usage (Figure 3 pipeline)
# ------------------------------------------------------------
# Uncomment this block to run the full analysis on the DKFZ Wegert data.

if (FALSE) {
  # 1. Fit fastTopics models k = 2:10
  fastTopics.wilms(
    counts.filepath  = "../data/gene_expression_dkfz_noOutliers.csv",
    output.filepath  = "../user_output/fastTopics_user/model_output/",
    times.directory  = "../user_output/fastTopics_user/times/",
    decomp.filepath  = "../user_output/fastTopics_user/params/",
    k_min            = 2,
    k_max            = 10
  )
  
  # 2. Plot log-likelihood trajectories for convergence diagnostics
  plot_loglik(
    fit.directory    = "../user_output/fastTopics_user/model_output/",
    output.directory = "../user_output/fastTopics_user/log_plots/"
  )
  
  # 3. Differential count analysis for a selected k
  #    (example: k = 3; adjust as needed)
  counts   <- read.csv("../data/gene_expression_dkfz_noOutliers.csv",
                       row.names = 1, check.names = FALSE)
  counts   <- as.matrix(counts)
  counts.T <- t(counts)
  
  # Load a specific fit (e.g. k = 3)
  load("../user_output/fastTopics_user/model_output/fit_3.RData")  # loads fit.wilms
  
  dfa_out <- diff_count_analysis(fit.wilms, counts.T)
  
  # Example: write differential analysis results
  write.csv(
    dfa_out,
    "../user_output/fastTopics_user/diff_count_analysis_k3.csv",
    row.names = FALSE
  )
}
