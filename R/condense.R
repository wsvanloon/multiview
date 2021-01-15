condense <- function(views, level){

  tab <- xtabs(~views[, level - 1] + views[, level])
  condensed_views <- rep(NA, length(unique(views[, level])))
  for(i in 1:nrow(tab)){
    condensed_views [i] <- which(tab[i,] != 0)
  }

  return(condensed_views)
}
