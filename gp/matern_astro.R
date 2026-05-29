# Gaussian-process baseline for dark-matter density fields.
#
# Fits a zero-mean Gaussian process with a Matern covariance (empirical-Bayes /
# maximum likelihood over the amplitude, range and smoothness) and reports the
# log score (negative mean log-likelihood) on the held-out test fields.
#
# Run from this directory; data CSVs live in ../data/:
#   Rscript matern_astro.R
#
# Requires the R packages: fields, mvtnorm.

library(fields)

locs = read.csv("../data/locs.csv", header = FALSE)
obs = read.csv("../data/stacked.csv", header = FALSE)

N = 160
train = obs[, 0:N]
test = obs[, 160:200]

n = 4096
matern_nloglik=function(params,dat,dists){
  cov.mat=exp(params[1])*fields::Matern(dists,exp(params[2]),nu=exp(params[3]))
  -mean(mvtnorm::dmvnorm(t(dat),rep(0,n),sigma=cov.mat,log=TRUE))
}

matern_ls_ho=function(params,dat,dists,holdout){
  cov.mat=exp(params[1])*fields::Matern(dists,exp(params[2]),nu=exp(params[3]))
  ls.ho.gaussian(cov.mat,dat,holdout)
}

matern_param=function(locs,data.train,data.test,holdout=NULL){
  pb <- txtProgressBar(min = 0, max = 100, style = 3)
  dists=rdist(locs)
  matern_nloglik_with_progress <- function(params, dat, dists) {
    setTxtProgressBar(pb, getTxtProgressBar(pb) + 1)  # Increment progress
    cov.mat = exp(params[1]) * fields::Matern(dists, exp(params[2]), nu = exp(params[3]))
    -mean(mvtnorm::dmvnorm(t(dat), rep(0, n), sigma = cov.mat, log = TRUE))
  }
  opt = optim(
    log(c(1, max(dists) * 0.1, 1)),
    matern_nloglik_with_progress,
    dat = data.train,
    dists = dists,
    control = list(trace = 0, maxit = 100, reltol = 1e-3)
  )
  close(pb)
  return(opt$par)
}

train_norm = (train - colMeans(t(train)))/apply(train, 1, sd)
test_norm = (test - colMeans(t(train)))/apply(train, 1, sd)

par = matern_param(locs, train_norm, test_norm)
dists=rdist(locs)
cov_hf = exp(par[1])*fields::Matern(dists,exp(par[2]),nu=exp(par[3]))

-mean(mvtnorm::dmvnorm(t(test_norm),rep(0,4096),sigma=cov_hf,log=TRUE))
