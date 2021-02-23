# This file is part of multiview: Methods for High-Dimensional Multi-View Learning
# Copyright (C) 2018-2021  Wouter van Loon
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#' Multi-View Stacking
#'
#' Fit a multi-view stacking model with two or more levels.
#' @param x input matrix of dimension nobs x nvars.
#' @param y outcome vector of length nobs.
#' @param views a matrix of dimension nvars x (levels - 1), where each entry is an integer describing to which view each feature corresponds.
#' @param type the type of MVS model to be fitted. Currently only type "StaPLR" is supported.
#' @param levels an integer >= 2, specifying the number of levels in the MVS procedure.
#' @param alphas a vector specifying the value of the alpha parameter to use at each level.
#' @param nnc a binary vector specifying whether to apply nonnegativity constraints or not (1/0) at each level.
#' @param parallel whether to use foreach to fit the learners and obtain the cross-validated predictions at each level in parallel. Executes sequentially unless a parallel back-end is registered beforehand.
#' @param seeds (optional) a vector specifying the seed to use at each level.
#' @param progress whether to show a progress bar (only supported when parallel = FALSE).
#' @param ... additional arguments to pass to the learning algorithm. See e.g. ?StaPLR. Note that these arguments are passed to the the learner at every level of the MVS procedure.
#' @return An object of S3 class "MVS".
#' @keywords TBA
#' @import foreach
#' @export
#' @author Wouter van Loon <w.s.van.loon@fsw.leidenuniv.nl>
#' @examples
#' set.seed(012)
#' n <- 1000
#' X <- matrix(rnorm(8500), nrow=n, ncol=85)
#' top_level <- c(rep(1,45), rep(2,20), rep(3,20))
#' bottom_level <- c(rep(1:3, each=15), rep(4:5, each=10), rep(6:9, each=5))
#' views <- cbind(bottom_level, top_level)
#' beta <- c(rep(10, 55), rep(0, 30)) * ((rbinom(85, 1, 0.5)*2)-1)
#' eta <- X %*% beta
#' p <- 1 /(1 + exp(-eta))
#' y <- rbinom(n, 1, p)
#'
#' fit <- MVS(x=X, y=y, views=views, type="StaPLR", levels=3, alphas=c(0,1,1), nnc=c(0,1,1))
#' coefficients <- coef(fit)
#'
#' new_X <- matrix(rnorm(2*85), nrow=2)
#' predict(fit, new_X)

MVS <- function(x, y, views, type="StaPLR", levels=2, alphas=c(0,1), nnc=c(0,1), parallel=FALSE, seeds=NULL, progress=TRUE, ...){

  pred_functions <- vector("list", length=ncol(views)+1)
  ll <- c(-Inf, 0)[nnc + 1]

  if(progress){
    message("Level 1 \n")
  }

  pred_functions[[1]] <- learn(X=x, y=y, views=views[,1], type=type, alpha1 = alphas[1], ll1=ll[1],
                               seed=seeds[1], progress=progress, parallel=parallel, ...)

  if(levels > 2){
    for(i in 2:ncol(views)){
      if(progress){
        message(paste("Level", i, "\n"))
      }
      pred_functions[[i]] <- learn(pred_functions[[i-1]]$CVs, y,
                                   views=condense(views, level=i), type=type,
                                   alpha1 = alphas[i], ll1=ll[i], seed=seeds[i],
                                   progress=progress, parallel=parallel, ...)
    }
  }

  if(progress){
    message(paste("Level", ncol(views)+1, "\n"))
  }

  pred_functions[[ncol(views)+1]] <- learn(pred_functions[[ncol(views)]]$CVs, y,
                                           views=rep(1,ncol(pred_functions[[ncol(views)]]$CVs)),
                                           type=type, alpha1 = alphas[ncol(views)+1], ll1=ll[ncol(views)+1],
                                           generate.CVs=FALSE, seed=seeds[ncol(views)+1],
                                           progress=progress, parallel=parallel, ...)

  for(i in 1:length(pred_functions)){
    pred_functions[[i]]$meta <- NULL
    names(pred_functions[[i]])[1] <- "models"
  }

  names(pred_functions) <- paste("Level", 1:(ncol(views)+1))
  attr(pred_functions, "type") <- type
  class(pred_functions) <- "MVS"

  return(pred_functions)

}


#' Make predictions from an "MVS" object.
#'
#' Make predictions from a "MVS" object.
#' @param object An object of class "MVS".
#' @param newx Matrix of new values for x at which predictions are to be made. Must be a matrix.
#' @param predtype The type of prediction returned by the meta-learner. Supported are types "response", "class" and "link".
#' @param cvlambda Values of the penalty parameters at which predictions are to be made. Defaults to the values giving minimum cross-validation error.
#' @return A matrix of predictions.
#' @keywords TBA
#' @export
#' @author Wouter van Loon <w.s.van.loon@fsw.leidenuniv.nl>
#' @examples
#' set.seed(012)
#' n <- 1000
#' X <- matrix(rnorm(8500), nrow=n, ncol=85)
#' top_level <- c(rep(1,45), rep(2,20), rep(3,20))
#' bottom_level <- c(rep(1:3, each=15), rep(4:5, each=10), rep(6:9, each=5))
#' views <- cbind(bottom_level, top_level)
#' beta <- c(rep(10, 55), rep(0, 30)) * ((rbinom(85, 1, 0.5)*2)-1)
#' eta <- X %*% beta
#' p <- 1 /(1 + exp(-eta))
#' y <- rbinom(n, 1, p)
#'
#' fit <- MVS(x=X, y=y, views=views, type="StaPLR", levels=3, alphas=c(0,1,1), nnc=c(0,1,1))
#' coefficients <- coef(fit)
#'
#' new_X <- matrix(rnorm(2*85), nrow=2)
#' predict(fit, new_X)

predict.MVS <- function(object, newx, predtype = "response", cvlambda = "lambda.min"){

  x <- newx

  for(i in 1:length(object)){

    Z <- matrix(NA, nrow=nrow(newx), ncol=length(object[[i]]$models))

    if(i < length(object)){
      pt <- object[[i]]$metadat
    }
    else pt <- predtype

    for(j in 1:ncol(Z)){
      Z[,j] <- predict(object[[i]]$models[[j]], x[, object[[i]]$view == j, drop=FALSE],
                       s = cvlambda, type = pt)
    }
    x <- Z
  }

  return(x)

}


#' Extract coefficients from an "MVS" object.
#'
#' Extract coefficients at each level from an "MVS" object at the CV-optimal values of the penalty parameters.
#' @param object An object of class "MVS".
#' @param cvlambda By default, the coefficients are extracted at the CV-optimal values of the penalty parameters. Choosing "lambda.1se" will extract them at the largest values within one standard error of the minima.
#' @return An object of S3 class "MVScoef".
#' @keywords TBA
#' @export
#' @author Wouter van Loon <w.s.van.loon@fsw.leidenuniv.nl>
#' @examples
#' set.seed(012)
#' n <- 1000
#' X <- matrix(rnorm(8500), nrow=n, ncol=85)
#' top_level <- c(rep(1,45), rep(2,20), rep(3,20))
#' bottom_level <- c(rep(1:3, each=15), rep(4:5, each=10), rep(6:9, each=5))
#' views <- cbind(bottom_level, top_level)
#' beta <- c(rep(10, 55), rep(0, 30)) * ((rbinom(85, 1, 0.5)*2)-1)
#' eta <- X %*% beta
#' p <- 1 /(1 + exp(-eta))
#' y <- rbinom(n, 1, p)
#'
#' fit <- MVS(x=X, y=y, views=views, type="StaPLR", levels=3, alphas=c(0,1,1), nnc=c(0,1,1))
#' coefficients <- coef(fit)
#'
#' new_X <- matrix(rnorm(2*85), nrow=2)
#' predict(fit, new_X)

coef.MVS <- function(object, cvlambda = "lambda.min"){

  out <- vector("list", length(object))

  for(i in 1:length(object)){
    out[[i]] <- vector("list", length(object[[i]]$models))
  }

  for(i in 1:length(object)){
    for(j in 1:length(object[[i]]$models)){
      out[[i]][[j]] <- coef(object[[i]]$models[[j]], s=cvlambda)
    }
  }
  names(out) <- paste("Level", 1:length(object))
  attr(out, "type") <- attr(object, "type")
  class(out) <- "MVScoef"

  return(out)
}
