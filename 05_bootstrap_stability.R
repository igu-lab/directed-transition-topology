# Sequence-level bootstrap stability check (B = 1000, set.seed(42))
library(tna)
data(group_regulation)
set.seed(42)
n_seq <- nrow(group_regulation)
B <- 1000

beta1_sym <- function(A) {
  U <- ((A + t(A)) > 0) * 1; diag(U) <- 0
  n <- nrow(U)
  edges <- which(U == 1 & upper.tri(U), arr.ind = TRUE)
  if (nrow(edges) == 0) return(0)
  vis <- rep(FALSE, n); cc <- 0
  for (s in 1:n) if (!vis[s]) {
    cc <- cc + 1; q <- s; vis[s] <- TRUE
    while (length(q) > 0) {
      u <- q[1]; q <- q[-1]
      nbrs <- which(U[u, ] == 1 & !vis)
      vis[nbrs] <- TRUE; q <- c(q, nbrs)
    }
  }
  tris <- list()
  if (n >= 3) {
    cmb <- combn(n, 3)
    for (k in 1:ncol(cmb)) {
      v <- cmb[, k]
      if (U[v[1],v[2]] && U[v[1],v[3]] && U[v[2],v[3]]) tris[[length(tris)+1]] <- v
    }
  }
  if (length(tris) == 0) return(nrow(edges) - n + cc)
  e_idx <- matrix(0, n, n)
  for (i in 1:nrow(edges)) e_idx[edges[i,1], edges[i,2]] <- i
  D2 <- matrix(0, nrow(edges), length(tris))
  for (j in seq_along(tris)) {
    v <- tris[[j]]
    D2[e_idx[v[1],v[2]], j] <- 1
    D2[e_idx[v[1],v[3]], j] <- -1
    D2[e_idx[v[2],v[3]], j] <- 1
  }
  (nrow(edges) - n + cc) - qr(D2)$rank
}

beta1_dfl <- function(A) {
  diag(A) <- 0; n <- nrow(A)
  edges <- which(A == 1, arr.ind = TRUE)
  if (nrow(edges) == 0) return(0)
  U <- ((A + t(A)) > 0) * 1
  vis <- rep(FALSE, n); cc <- 0
  for (s in 1:n) if (!vis[s]) {
    cc <- cc + 1; q <- s; vis[s] <- TRUE
    while (length(q) > 0) {
      u <- q[1]; q <- q[-1]
      nbrs <- which(U[u, ] == 1 & !vis)
      vis[nbrs] <- TRUE; q <- c(q, nbrs)
    }
  }
  tris <- list()
  for (i in 1:n) for (j in 1:n) if (i != j && A[i,j] == 1) {
    for (k in 1:n) if (k != i && k != j && A[i,k] == 1 && A[j,k] == 1) {
      tris[[length(tris)+1]] <- c(i, j, k)
    }
  }
  if (length(tris) == 0) return(nrow(edges) - n + cc)
  e_idx <- matrix(0, n, n)
  for (i in 1:nrow(edges)) e_idx[edges[i,1], edges[i,2]] <- i
  D2 <- matrix(0, nrow(edges), length(tris))
  for (m in seq_along(tris)) {
    v <- tris[[m]]
    D2[e_idx[v[2],v[3]], m] <- D2[e_idx[v[2],v[3]], m] + 1
    D2[e_idx[v[1],v[3]], m] <- D2[e_idx[v[1],v[3]], m] - 1
    D2[e_idx[v[1],v[2]], m] <- D2[e_idx[v[1],v[2]], m] + 1
  }
  (nrow(edges) - n + cc) - qr(D2)$rank
}

W_obs <- tna(group_regulation)$weights
A26_obs <- (W_obs >= 0.26) * 1; diag(A26_obs) <- 0

b24 <- 0; b26 <- 0; loop26 <- 0; exact26 <- 0; asym26 <- 0
drop_ac <- 0; drop_cd <- 0; add_sa <- 0; cyc16_unfilled <- 0

for (b in 1:B) {
  idx <- sample(1:n_seq, n_seq, replace = TRUE)
  W <- tna(group_regulation[idx, ])$weights
  A24 <- (W >= 0.24) * 1; diag(A24) <- 0
  A26 <- (W >= 0.26) * 1; diag(A26) <- 0
  A16 <- (W >= 0.16) * 1; diag(A16) <- 0

  if (beta1_sym(A24) == 0 && beta1_dfl(A24) == 1) b24 <- b24 + 1
  if (beta1_sym(A26) == 0 && beta1_dfl(A26) == 1) b26 <- b26 + 1
  if (A26["consensus","plan"] == 1 && A26["plan","consensus"] == 1) loop26 <- loop26 + 1
  if (all(A26 == A26_obs)) exact26 <- exact26 + 1
  if (A26["adapt","cohesion"] == 0) drop_ac <- drop_ac + 1
  if (A26["coregulate","discuss"] == 0) drop_cd <- drop_cd + 1
  if (A26["synthesis","adapt"] == 1) add_sa <- add_sa + 1
  if (sum(choose(colSums(A26), 3)) > sum(choose(rowSums(A26), 3))) asym26 <- asym26 + 1
  if (A16["consensus","coregulate"] == 1 && A16["coregulate","emotion"] == 1 &&
      A16["emotion","consensus"] == 1 && A16["coregulate","consensus"] == 0 &&
      A16["emotion","coregulate"] == 0 && A16["consensus","emotion"] == 0) {
    cyc16_unfilled <- cyc16_unfilled + 1
  }
}

cat(sprintf("loop26=%.1f%% asym26=%.1f%% blind26=%.1f%% blind24=%.1f%% cyc16_unfilled=%.1f%%\n",
    100*loop26/B, 100*asym26/B, 100*b26/B, 100*b24/B, 100*cyc16_unfilled/B))
cat(sprintf("exact26=%.1f%% drop_ac=%.1f%% drop_cd=%.1f%% add_sa=%.1f%%\n",
    100*exact26/B, 100*drop_ac/B, 100*drop_cd/B, 100*add_sa/B))
