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

MVS <- function(X, y, views, type="StaPLR", levels=2, alphas=c(0,1), progress=TRUE, seeds=NULL){

  pred_functions <- vector("list", length=ncol(views)+1)

  if(progress){
    message("Level 1 \n")
  }

  pred_functions[[1]] <- learn(X=X, y=y, views=views[,1], type=type, alpha1 = alphas[1],
                               seed=seeds[1], progress=progress)

  if(levels > 2){
    for(i in 2:ncol(views)){
      if(progress){
        message(paste("Level", i, "\n"))
      }
      pred_functions[[i]] <- learn(pred_functions[[i-1]]$CVs, y,
                                   views=condense(views, level=i), type=type,
                                   alpha1 = alphas[i], seed=seeds[i], progress=progress)
    }
  }

  if(progress){
    message(paste("Level", ncol(views)+1, "\n"))
  }

  pred_functions[[ncol(views)+1]] <- learn(pred_functions[[ncol(views)]]$CVs, y,
                                           views=rep(1,ncol(pred_functions[[ncol(views)]]$CVs)),
                                           type=type, alpha1 = alphas[ncol(views)+1], ll1=0,
                                           generate.CVs=FALSE, seed=seeds[ncol(views)+1],
                                           progress=progress)

  for(i in 1:length(pred_functions)){
    pred_functions[[i]]$meta <- NULL
    names(pred_functions[[i]])[1] <- "models"
  }

  names(pred_functions) <- paste("Level", 1:(ncol(views)+1))
  attr(pred_functions, "type") <- type
  class(pred_functions) <- "MVS"

  return(pred_functions)

}


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
  attr(out, "type") <- attr(object, "type")
  class(out) <- "MVScoef"

  return(out)
}

# predict.MVS <- function(object, newx, predtype = "response", cvlambda = "lambda.min"){
#
# }

# At the FIRST LEVEL, we should have:
# - A loop to learn the base-learner fv from each X[, views=v] THAT IS INCLUDED AT THIS LEVEL
# - A loop to obtain the cross-validated predictions for each of those fv, collected in matrix Z1

# Then at the next level, we should have:
# - A loop to learn the intermediate classifier from each Z1[, something_something]
# - A loop to obtain the cross-validated predictions for each of the intermediate classifiers, collected in Z2

# Then at the last level, we should have:
# - A loop to train the meta-learner on Z2 (or Z3, or Z4, whatever the highest level is)

# The learners differ on the basis of their INPUT and OUTPUT
# Base-learner:         INPUT: DATA
#                       OUTPUT: INPUT TO OTHER ALGORITHM
# Transitional learner: INPUT: OUTPUT OF OTHER ALGORITHM
#                       OUTPUT: INPUT FOR OTHER ALGORITHM
# Meta leaner:          INPUT: OUTPUT OF ANOTHER ALGORITHM
#                       OUTPUT: PREDICTION


# Transitional learner: A learner that takes as input the output of another learner, and outputs the input of another learner.
