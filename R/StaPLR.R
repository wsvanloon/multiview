#' Stacked Penalized Logistic Regression
#'
#' Fit a two-level stacked penalized logistic regression model with a single base-learner and a single meta-learner.
#' @param x input matrix of dimension nobs x nvars
#' @param y outcome vector of length nobs
#' @param view a vector of length nvars, where each entry is an integer describing to which view each feature corresponds.
#' @param alpha1 (base) alpha parameter for glmnet: lasso(1) / ridge(0)
#' @param alpha2 (meta) alpha parameter for glmnet: lasso(1) / ridge(0)
#' @param std.base should features be standardized at the base level?
#' @param std.meta should cross-validated predictions be standardized at the meta level?
#' @param ll1 lower limit(s) for each coefficient at the base-level.
#' @param ul1 upper limit(s) for each coefficient at the base-level.
#' @param ll2 lower limit(s) for each coefficient at the meta-level.
#' @param ul2 upper limit(s) for each coefficient at the meta-level.
#' @param cvloss loss to use for cross-validation.
#' @param metadat which attribute of the base learners should be used as input for the meta learner?
#' @param cvlambda value of lambda at which cross-validated predictions are made.
#' @param cvparallel whether to use 'foreach' to fit each CV fold.
#' @return TBA.
#' @keywords TBA
#' @export
#' @author Wouter van Loon <w.s.van.loon@fsw.leidenuniv.nl>
#' @examples
#' TBA

StaPLR <- function(x, y, view, alpha1 = 0, alpha2 = 1, nfolds = 5, myseed = NA,
                      std.base = FALSE, std.meta = FALSE, ll1 = -Inf, ul1 = Inf,
                      ll2 = 0, ul2 = Inf, cvloss = "deviance", metadat = "response", cvlambda = "lambda.min",
                      cvparallel = FALSE, lambda.ratio = 0.01){


  # Performs two-level Stacked Penalized Logistic Regression (StaPLR) with
  #   - a binary response variable
  #   - a single penalized logistic model (L1 and/or L2) as base learner
  #   - a single penalized logistic model (L1 and/or L2) as meta learner

  # Input
  #   x: input matrix of dimension nobs x nvars
  #   y: outcome vector of length nobs
  #   view: a vector of length nvars, where each entry is an integer describing to which view each feature corresponds.
  #   alpha1: (base) alpha parameter for glmnet: lasso(1) / ridge(0)
  #   alpha2: (meta) alpha parameter for glmnet: lasso(1) / ridge(0)
  #   std.base: should features be standardized at the base level?
  #   std.meta: should cross-validated predictions be standardized at the meta level?
  #   ll1: lower limit(s) for each coefficient at the base-level.
  #   ul1: upper limit(s) for each coefficient at the base-level.
  #   ll2: lower limit(s) for each coefficient at the meta-level.
  #   ul2: upper limit(s) for each coefficient at the meta-level.
  #   cvloss: loss to use for cross-validation.
  #   metadat: which attribute of the base learners should be used as input for the meta learner?
  #   cvlambda: value of lambda at which cross-validated predictions are made.
  #   cvparallel: whether to use 'foreach' to fit each CV fold.

  # load libraries
  # require(caret)
  # require(glmnet)
  # require(gglasso)

  # register parallel backend
  # if(cvparallel){
  #   library(doParallel)
  #   source("pardiag.R")
  #   ncores <- detectCores()
  #   if(getDoParWorkers() < ncores){
  #     registerDoParallel(ncores)
  #   }
  #   print(pardiag())
  #   cat("\n")
  # }

  # set seed
  # if(!is.na(myseed)){
  #   set.seed(myseed)
  # }

  # !!!!!!!!!!= BUILD EARLY PATH TERMINATION CHECK !!!!!!!!!!!!!

  # object initialization
  V <- length(unique(view))
  n <- length(y)
  Z <- matrix(NA, n, V)
  folds <- caret::createFolds(y, nfolds, list = FALSE)
  cv.base <- vector("list", V)

  # STEP 0: fit glmnet on each of the domains
  cat("Training base-learner on each domain...", "\n")
  for (v in 1:V){
    cat(v, "")
    cv.base[[v]] <- glmnet::cv.glmnet(x[, view == v], y, family = "binomial", nfolds = nfolds,
                              type.measure = cvloss, alpha = alpha1,
                              standardize = std.base, lower.limits = ll1,
                              upper.limits = ul1, parallel = cvparallel, lambda.min.ratio = lambda.ratio)
  }
  cat("\n")

  # STEP 1: Loop over domains and k-fold CV to obtain Z
  cat("Calculating cross-validated predictions for each domain...", "\n")
  for(v in 1:V){
    cat(v, "")
    for(k in 1:nfolds){
      cvf <- glmnet::cv.glmnet(x[folds != k, view == v], y[folds != k], family = "binomial",
                       type.measure = cvloss, alpha = alpha1,
                       standardize = std.base, lower.limits = ll1,
                       upper.limits = ul1, parallel = cvparallel, lambda.min.ratio = lambda.ratio)
      Z[folds == k,v] <- glmnet::predict.glmnet(cvf, newx = x[folds == k, view == v], s = cvlambda, type = metadat)
    } # 10fold cv
  } # domains
  cat("\n")

  # STEP 2: Train meta learner
  cat("Training meta-learner...", "\n")
  cv.meta <- glmnet::cv.glmnet(Z, y, family= "binomial", type.measure = cvloss, alpha = alpha2,
                       standardize = std.meta, lower.limits = ll2,
                       upper.limits = ul2, parallel = cvparallel, lambda.min.ratio=lambda.ratio)

  # create output list
  outlist = list(
    "base" = cv.base,
    "meta" = cv.meta,
    "CVs" = Z,
    "x" = x,
    "y" = y,
    "view" = view
  )

  # return output
  cat("DONE", "\n")
  return(outlist)
}



#' Make predictions from a "StaPLR" object.
#'
#' Fit a two-level stacked penalized logistic regression model with a single base-learner and a single meta-learner.
#' @param object Fitted "StaPLR" model object.
#' @param newx Matrix of new values for x at which predictions are to be made. Must be a matrix.
#' @param metadat The attribute of the base-learners to be used as input to the meta-learner.
#' @param predtype The type of prediction returned by the meta-learner.
#' @param cvlambda Value of the penalty parameter lambda at which predictions are to be made.
#' @return TBA.
#' @keywords TBA
#' @export
#' @author Wouter van Loon <w.s.van.loon@fsw.leidenuniv.nl>
#' @examples
#' TBA
predict.StaPLR = function(object, newx, metadat = "response", predtype = "response", cvlambda = "lambda.min"){
  # prediction based on StaPLR
  # Input
  #   object: an output object from StaPLR
  #   newx: matrix with new values for the predictors
  #   metadat: which attribute of the base learners should be used as input for the meta learner? (should this not be fixed?)
  #   predtype: type of prediction generated by the meta learner
  #   cvlambda: value of lambda at which predictions are made

  V <- length(unique(object$view))
  n <- nrow(newx)
  Z <- matrix(NA, n, V)
  for (v in 1:V){
    Z[,v] <- glmnet::predict.glmnet(object$base[[v]], newx[, object$view == v, drop=FALSE], s = cvlambda, type = metadat)
  }
  out <- glmnet::predict.glmnet(object$meta, Z, s = cvlambda, type = predtype)
  return(out)
}
