# This is a quick-and-dirty script to apply StaPLR with 3 levels. Its outcome can potentially be used for automated testing of MVS later on.

StaPLR3 <- function(X, y , views, alphas, nfolds=10){

  base_learners <- StaPLR(X, y, view=views[,1], alpha1 = alphas[1], alpha2 = alphas[2], skip.meta=TRUE, ll1 = -Inf, ul1 = Inf)

  trans_learners <- StaPLR(base_learners$CVs, y, view=condense(views, level=2), alpha1 = alphas[2], alpha2 = alphas[3], ll1 = 0, ul1 = Inf,
                           ll2 = 0, ul2 = Inf)

  out <- list(base=base_learners$base,
              trans=trans_learners$base,
              meta=trans_learners$meta,
              Z1=base_learners$CVs,
              Z2=trans_learners$CVs,
              X=base_learners$x,
              y=base_learners$y,
              views1=base_learners$view,
              views2=trans_learners$view)

  class(out) <- "StaPLR3"

  return(out)
}


condense <- function(views, level){

  tab <- xtabs(~views[, level - 1] + views[, level])
  condensed_views <- rep(NA, length(unique(views[, level])))
  for(i in 1:nrow(tab)){
    condensed_views [i] <- which(tab[i,] != 0)
  }

  return(condensed_views)
}


predict.StaPLR3 <- function(object, newx, newcf = NULL, predtype = "response", cvlambda = "lambda.min"){

  V1 <- length(unique(object$views1))
  n <- nrow(newx)
  metadat <- "response" # THIS SHOULD INHERIT FROM STAPLR3 OBJECT INSTEAD!
  Z1 <- matrix(NA, n, V1)
  for (v in 1:V1){
    Z1[,v] <- predict(object$base[[v]], newx[, object$views1 == v, drop=FALSE], s = cvlambda, type = metadat)
  }
  # if(!is.null(newcf)){
  #   Z <- cbind(newcf, Z)
  # }
  colnames(Z1) <- colnames(object$CVs)

  V2 <- length(unique(object$views2))
  Z2 <- matrix(NA, n, V2)
  for (v in 1:V2){
    Z2[,v] <- predict(object$trans[[v]], Z1[, object$views2 == v, drop=FALSE], s = cvlambda, type = metadat)
  }

  out <- predict(object$meta, Z2, s = cvlambda, type = predtype)
  return(out)

}

# TEST DATA
# n <- 1000
# set.seed(111220)
# test_X<- matrix(rnorm(8500), nrow=n, ncol=85)
# top_level <- c(rep(1,45), rep(2,20), rep(3,20))
# bottom_level <- c(rep(1:3, each=15), rep(4:5, each=10), rep(6:9, each=5))
# views <- cbind(bottom_level, top_level)
#
# beta <- c(rep(10, 55), rep(0, 30))
# eta <- test_X %*% beta
# p <- 1 /(1 + exp(-eta))
# test_y <- rbinom(n, 1, p)
