# 01_tna_nestimate_baseline.R

suppressPackageStartupMessages({
  library(tna)
  library(Nestimate)
})

dir.create("output", showWarnings = FALSE, recursive = TRUE)

# 1. Load tna sample dataset
data("group_regulation", package = "tna")

# 2. Standard TNA model & undirected cliques
tna_model <- tna(group_regulation)
weights_mat <- as.matrix(tna_model$weights)
write.csv(weights_mat, "output/tna_weights.csv", row.names = TRUE)

count_cliques_safe <- function(model, sz, th) {
  res <- tryCatch(cliques(model, size = sz, threshold = th), error = function(e) NULL)
  if (is.null(res)) return(0L)
  if (!is.null(res$weights)) return(length(res$weights))
  if (!is.null(res$inits)) return(length(res$inits))
  0L
}

thresholds <- c(0.05, 0.10, 0.15, 0.20, 0.25)
clique_rows <- lapply(thresholds, function(th) {
  data.frame(
    threshold = th,
    n_cliques_size2 = count_cliques_safe(tna_model, 2, th),
    n_cliques_size3 = count_cliques_safe(tna_model, 3, th)
  )
})
clique_df <- do.call(rbind, clique_rows)
write.csv(clique_df, "output/tna_cliques_summary.csv", row.names = FALSE)

# 3. Nestimate estimation & bootstrap stability
nest_net <- build_network(group_regulation, method = "relative")
set.seed(42)
nest_boot <- tryCatch({
  bootstrap_network(nest_net, iter = 500, ci_level = 0.05)
}, error = function(e) {
  message("Note: bootstrap_network() returned an error: ", e$message)
  NULL
})

if (!is.null(nest_boot) && !is.null(nest_boot$significant)) {
  sig_mat <- as.matrix(nest_boot$significant)
  write.csv(sig_mat, "output/tna_boot_sig.csv", row.names = TRUE)
}

# 4. Nestimate Simplicial Complex, Betti numbers & Persistent Homology
sc <- tryCatch({
  build_simplicial(nest_net)
}, error = function(e) {
  message("Note: build_simplicial() returned an error: ", e$message)
  NULL
})

betti_res <- if (!is.null(sc)) {
  tryCatch(betti_numbers(sc), error = function(e) {
    message("Note: Nestimate::betti_numbers() returned an error: ", e$message)
    NULL
  })
} else NULL

if (!is.null(betti_res)) {
  write.csv(as.data.frame(betti_res), "output/nestimate_betti.csv", row.names = FALSE)
}

ph_res <- tryCatch({
  persistent_homology(nest_net)
}, error = function(e) {
  message("Note: Nestimate::persistent_homology() returned an error: ", e$message)
  NULL
})

if (!is.null(ph_res)) {
  tryCatch({
    if (!is.null(ph_res$diagram)) {
      write.csv(as.data.frame(ph_res$diagram), "output/nestimate_ph.csv", row.names = FALSE)
    } else {
      write.csv(as.data.frame(ph_res), "output/nestimate_ph.csv", row.names = FALSE)
    }
  }, error = function(e) NULL)
}

cat("=== Baseline R Summary ===\n")
cat("Nodes:", nrow(weights_mat), "States:", paste(rownames(weights_mat), collapse = ", "), "\n")
print(clique_df)
if (!is.null(betti_res)) {
  cat("\n=== Nestimate betti_numbers() ===\n")
  print(betti_res)
}
if (!is.null(ph_res)) {
  cat("\n=== Nestimate persistent_homology() ===\n")
  print(ph_res)
}
cat("Done. Check output/ directory.\n")
