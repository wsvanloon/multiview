#' Stacked Penalized Logistic Regression
#'
#' Fit a two-level stacked penalized logistic regression model with a single base-learner and a single meta-learner.
#' @param x input matrix of dimension nobs x nvars
#' @param y outcome vector of length nobs
#' @param view a vector of length nvars, where each entry is an integer describing to which view each feature corresponds.
#' @param view.names (optional) a character vector of length nviews specifying a name for each view.
#' @param correct.for (optional) a matrix with nrow = nobs, where each column is a feature which should be included directly into the meta.learner. By default these features are not penalized (see penalty.weights) and appear at the top of the coefficient list.
#' @param alpha1 (base) alpha parameter for glmnet: lasso(1) / ridge(0)
#' @param alpha2 (meta) alpha parameter for glmnet: lasso(1) / ridge(0)
#' @param nfolds number of folds to use for all cross-validation.
#' @param seed (optional) numeric value specifying the seed. Setting the seed this way ensures the results are reproducable even when the computations are performed in parallel.
#' @param std.base should features be standardized at the base level?
#' @param std.meta should cross-validated predictions be standardized at the meta level?
#' @param ll1 lower limit(s) for each coefficient at the base-level. Defaults to -Inf.
#' @param ul1 upper limit(s) for each coefficient at the base-level. Defaults to Inf.
#' @param ll2 lower limit(s) for each coefficient at the meta-level. Defaults to 0 (non-negativity constraints).
#' @param ul2 upper limit(s) for each coefficient at the meta-level. Defaults to Inf.
#' @param cvloss loss to use for cross-validation.
#' @param metadat which attribute of the base learners should be used as input for the meta learner?
#' @param cvlambda value of lambda at which cross-validated predictions are made.
#' @param cvparallel whether to use 'foreach' to fit each CV fold.
#' @param lambda.ratio the ratio between the largest and smallest lambda value.
#' @param penalty.weights (optional) a vector of length nviews+ncol(correct.for), containing different penalty factors for the meta-learner. Defaults to rep(1,nviews) if correct.for = NULL, and c(rep(0, ncol(correct.for)), rep(1, nviews)) otherwise.
#' @param parallel whether to use foreach to fit the base-learners and obtain the cross-validated predictions in parallel. Executes sequentially unless a parallel backend is registered beforehand.
#' @param skip.fdev whether to skip checking if the fdev parameter is set to zero.
#' @param skip.version whether to skip checking the version of the glmnet package.
#' @param progress whether to show a progress bar (only supported when parallel = FALSE).
#' @return TBA.
#' @keywords TBA
#' @import foreach
#' @export
#' @author Wouter van Loon <w.s.van.loon@fsw.leidenuniv.nl>
#' @examples
#' set.seed(012)
#' n <- 1000
#' cors <- seq(0.1,0.7,0.1)
#' X <- matrix(NA, nrow=n, ncol=length(cors)+1)
#' X[,1] <- rnorm(n)
#'
#' for(i in 1:length(cors)){
#'   X[,i+1] <- X[,1]*cors[i] + rnorm(n, 0, sqrt(1-cors[i]^2))
#' }
#'
#' beta <- c(1,0,0,0,0,0,0,0)
#' eta <- X %*% beta
#' p <- exp(eta)/(1+exp(eta))
#' y <- rbinom(n, 1, p)
#' view_index <- rep(1:(ncol(X)/2), each=2)
#'
#' fit <- StaPLR(X, y, view_index)
#' coef(fit$meta, s="lambda.min")
#'
#' new_X <- matrix(rnorm(16), nrow=2)
#' predict(fit, new_X)

StaPLR <- function(x, y, view, view.names = NULL, correct.for = NULL, alpha1 = 0, alpha2 = 1, nfolds = 5, seed = NULL,
                      std.base = FALSE, std.meta = FALSE, ll1 = -Inf, ul1 = Inf,
                      ll2 = 0, ul2 = Inf, cvloss = "deviance", metadat = "response", cvlambda = "lambda.min",
                      cvparallel = FALSE, lambda.ratio = 0.01, penalty.weights = NULL, parallel = FALSE, skip.fdev = FALSE, skip.version = FALSE, progress = TRUE){

  # Check if glmnet.control parameter fdev is set to zero.
  if(skip.fdev == FALSE){
    if(glmnet.control()$fdev != 0){
      glmnet.control(fdev=0)
      message("Minimum fractional change in deviance for stopping path set to zero for the duration of this session. \n To reset to default use: glmnet.control(factory=TRUE) (NOT RECOMMENDED). \n To skip this check use: StaPLR(..., skip.fdev=TRUE) (NOT RECOMMENDED). \n")
    }
  }

  # Check current version of glmnet
  if(skip.version == FALSE){
    if(packageVersion("glmnet") != '1.9.8'){
      versionMessage <- paste0("Found glmnet version ", packageVersion("glmnet"), ". The recommended version of glmnet is 1.9-8. \n Package versions >= 2.0-1 are less stable than 1.9-8 due to a change in the cross-validation procedure. \n Because of this change, results may differ between versions 1.9-8 and >= 2.0-1. \n If you want to install glmnet 1.9-8 from source use e.g.: devtools::install_version(\"glmnet\", version=\"1.9-8\"). \n To skip this check use: StaPLR(..., skip.version=TRUE). \n")
      message(versionMessage)
    }
  }

  # object initialization
  V <- length(unique(view))
  n <- length(y)

  # SEQUENTIAL PROCESSING
  if(parallel == FALSE){

    if(!is.null(seed)){
      set.seed(seed)
      folds <- kFolds(y, nfolds)
      base.seeds <- sample(.Machine$integer.max/2, size = V)
      z.seeds <- matrix(sample(.Machine$integer.max/2, size = V*nfolds), nrow=nfolds, ncol=V)
      meta.seed <- sample(.Machine$integer.max/2, size=1)
    }
    else folds <- kFolds(y, nfolds)

    if(progress == TRUE){
      message("Training base-learner on each view...")
      pb <- txtProgressBar(min=0, max=V, style=3)
    }
    cv.base <- foreach(v=(1:V)) %do% {
      if(progress == TRUE){
        setTxtProgressBar(pb, v)
      }
      if(!is.null(seed)){
        set.seed(base.seeds[v])
      }
      glmnet::cv.glmnet(x[, view == v], y, family = "binomial", nfolds = nfolds,
                        type.measure = cvloss, alpha = alpha1,
                        standardize = std.base, lower.limits = ll1,
                        upper.limits = ul1, parallel = cvparallel, lambda.min.ratio = lambda.ratio)
    }

    if(progress == TRUE){
      message("\n Calculating cross-validated predictions...")
      pb <- txtProgressBar(min=0, max=V*nfolds, style=3)
    }

    Z <- foreach(v=(1:V), .combine=cbind) %:%
      foreach(k=(1:nfolds), .combine="+") %do% {
        if(progress == TRUE){
          setTxtProgressBar(pb, getTxtProgressBar(pb)+1)
        }
        if(!is.null(seed)){
          set.seed(z.seeds[k, v])
        }
        cvf <- glmnet::cv.glmnet(x[folds != k, view == v], y[folds != k], family = "binomial",
                                 type.measure = cvloss, alpha = alpha1,
                                 standardize = std.base, lower.limits = ll1,
                                 upper.limits = ul1, parallel = cvparallel, lambda.min.ratio = lambda.ratio)
        newy <- rep(0, length(y))
        newy[folds == k] <- predict(cvf, newx = x[folds == k, view == v], s = cvlambda, type = metadat)
        return(newy)
      }
    dimnames(Z) <- NULL
    if(!is.null(view.names)){
      colnames(Z) <- view.names
    }

    if(progress == TRUE){
      message("\n Training meta learner...")
    }
    if(!is.null(seed)){
      set.seed(meta.seed)
    }
    if(is.null(correct.for) && is.null(penalty.weights)){
      cv.meta <- glmnet::cv.glmnet(Z, y, family= "binomial", type.measure = cvloss, alpha = alpha2,
                                   standardize = std.meta, lower.limits = ll2,
                                   upper.limits = ul2, parallel = cvparallel, lambda.min.ratio=lambda.ratio)
    }
    else if(is.null(correct.for) && !is.null(penalty.weights)){
      cv.meta <- glmnet::cv.glmnet(Z, y, family= "binomial", type.measure = cvloss, alpha = alpha2,
                                   standardize = std.meta, lower.limits = ll2,
                                   upper.limits = ul2, parallel = cvparallel, lambda.min.ratio=lambda.ratio, penalty.factor=penalty.weights)
    }
    else{
      if(is.null(penalty.weights)){
        penalty.weights <- c(rep(0, ncol(correct.for)), rep(1, ncol(Z)))
      }
      Z <- cbind(correct.for, Z)
      cv.meta <- glmnet::cv.glmnet(Z, y, family= "binomial", type.measure = cvloss, alpha = alpha2,
                                   standardize = std.meta, lower.limits = ll2,
                                   upper.limits = ul2, parallel = cvparallel, lambda.min.ratio=lambda.ratio, penalty.factor=penalty.weights)
    }

  }

  # PARALLEL PROCESSING
  if(parallel == TRUE){

    if(!is.null(seed)){
      set.seed(seed)
      folds <- kFolds(y, nfolds)
      base.seeds <- sample(.Machine$integer.max/2, size = V)
      z.seeds <- matrix(sample(.Machine$integer.max/2, size = V*nfolds), nrow=nfolds, ncol=V)
      meta.seed <- sample(.Machine$integer.max/2, size=1)
    }
    else folds <- kFolds(y, nfolds)

    cv.base <- foreach(v=(1:V)) %dopar% {
      if(!is.null(seed)){
        set.seed(base.seeds[v])
      }
      glmnet::cv.glmnet(x[, view == v], y, family = "binomial", nfolds = nfolds,
                        type.measure = cvloss, alpha = alpha1,
                        standardize = std.base, lower.limits = ll1,
                        upper.limits = ul1, parallel = cvparallel, lambda.min.ratio = lambda.ratio)
    }

    Z <- foreach(v=(1:V), .combine=cbind) %:%
      foreach(k=(1:nfolds), .combine="+") %dopar% {
        if(!is.null(seed)){
          set.seed(z.seeds[k, v])
        }
        cvf <- glmnet::cv.glmnet(x[folds != k, view == v], y[folds != k], family = "binomial",
                                 type.measure = cvloss, alpha = alpha1,
                                 standardize = std.base, lower.limits = ll1,
                                 upper.limits = ul1, parallel = cvparallel, lambda.min.ratio = lambda.ratio)
        newy <- rep(0, length(y))
        newy[folds == k] <- predict(cvf, newx = x[folds == k, view == v], s = cvlambda, type = metadat)
        return(newy)
      }
    dimnames(Z) <- NULL
    if(!is.null(view.names)){
      colnames(Z) <- view.names
    }

    if(!is.null(seed)){
      set.seed(meta.seed)
    }
    if(is.null(correct.for) && is.null(penalty.weights)){
      cv.meta <- glmnet::cv.glmnet(Z, y, family= "binomial", type.measure = cvloss, alpha = alpha2,
                                   standardize = std.meta, lower.limits = ll2,
                                   upper.limits = ul2, parallel = cvparallel, lambda.min.ratio=lambda.ratio)
    }
    else if(is.null(correct.for) && !is.null(penalty.weights)){
      cv.meta <- glmnet::cv.glmnet(Z, y, family= "binomial", type.measure = cvloss, alpha = alpha2,
                                   standardize = std.meta, lower.limits = ll2,
                                   upper.limits = ul2, parallel = cvparallel, lambda.min.ratio=lambda.ratio, penalty.factor=penalty.weights)
    }
    else{
      if(is.null(penalty.weights)){
        penalty.weights <- c(rep(0, ncol(correct.for)), rep(1, ncol(Z)))
      }
      Z <- cbind(correct.for, Z)
      cv.meta <- glmnet::cv.glmnet(Z, y, family= "binomial", type.measure = cvloss, alpha = alpha2,
                                   standardize = std.meta, lower.limits = ll2,
                                   upper.limits = ul2, parallel = cvparallel, lambda.min.ratio=lambda.ratio, penalty.factor=penalty.weights)
    }
  }


  # create output list
  out = list(
    "base" = cv.base,
    "meta" = cv.meta,
    "CVs" = Z,
    "x" = x,
    "y" = y,
    "view" = view
  )

  class(out) <- "StaPLR"

  # return output
  message("DONE")
  return(out)
}



#' Make predictions from a "StaPLR" object.
#'
#' Fit a two-level stacked penalized logistic regression model with a single base-learner and a single meta-learner.
#' @param object Fitted "StaPLR" model object.
#' @param newx Matrix of new values for x at which predictions are to be made. Must be a matrix.
#' @param newcf matrix of new values of correction features, if correct.for was specified during model fitting.
#' @param metadat The attribute of the base-learners to be used as input to the meta-learner.
#' @param predtype The type of prediction returned by the meta-learner.
#' @param cvlambda Value of the penalty parameter lambda at which predictions are to be made.
#' @return TBA.
#' @keywords TBA
#' @export
#' @author Wouter van Loon <w.s.van.loon@fsw.leidenuniv.nl>
#' @examples
#' set.seed(012)
#' n <- 1000
#' cors <- seq(0.1,0.7,0.1)
#' X <- matrix(NA, nrow=n, ncol=length(cors)+1)
#' X[,1] <- rnorm(n)
#'
#' for(i in 1:length(cors)){
#'   X[,i+1] <- X[,1]*cors[i] + rnorm(n, 0, sqrt(1-cors[i]^2))
#' }
#'
#' beta <- c(1,0,0,0,0,0,0,0)
#' eta <- X %*% beta
#' p <- exp(eta)/(1+exp(eta))
#' y <- rbinom(n, 1, p)
#' view_index <- rep(1:(ncol(X)/2), each=2)
#'
#' fit <- StaPLR(X, y, view_index)
#' coef(fit$meta, s="lambda.min")
#'
#' new_X <- matrix(rnorm(16), nrow=2)
#' predict(fit, new_X)

predict.StaPLR <- function(object, newx, newcf = NULL, metadat = "response", predtype = "response", cvlambda = "lambda.min"){

  V <- length(unique(object$view))
  n <- nrow(newx)
  Z <- matrix(NA, n, V)
  for (v in 1:V){
    Z[,v] <- predict(object$base[[v]], newx[, object$view == v, drop=FALSE], s = cvlambda, type = metadat)
  }
  if(!is.null(newcf)){
    Z <- cbind(newcf, Z)
  }
  colnames(Z) <- colnames(object$CVs)
  out <- predict(object$meta, Z, s = cvlambda, type = predtype)
  return(out)
}
