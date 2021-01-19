# Learn is a generic function to train a learner on multi-view input data,
# and produce learned functions and (optionally) the matrix of cross-validated
# predictions. It uses switch() to apply the correct learner, depending on
# argument 'type'.

learn <- function(X, y, views, type, generate.CVs=TRUE, ...){
  switch(type,
         StaPLR = StaPLR(X, y, view=views, skip.meta = TRUE, skip.cv = !generate.CVs, ...))
}
