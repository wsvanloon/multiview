kFolds <- function(y, k){
  out <- sample(rep(1:k, length=length(y)))
  return(out)
}
