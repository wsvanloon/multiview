MVS <- function(X, y, views, levels, etc){

  # pred_functions <- vector("list", length=ncol(views)+1)
  #
  # pred_functions[[1]] <- learn(X, y, views, etc)
  #
  # for(i in 2:ncol(views)){
  #   trans_learners[[i]] <- learn(pred_functions[[i-1]]$CVs, y, views, etc)
  # }
  #
  # pred_functions[[ncol(views)+1]] <- learn(pred_functions[[ncol(views)]]$CVs, y, views, etc, generate_CVs=FALSE)

}

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
