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

condense <- function(views, level){

  tab <- xtabs(~views[, level - 1] + views[, level])
  condensed_views <- rep(NA, length(unique(views[, level])))
  for(i in 1:nrow(tab)){
    condensed_views [i] <- which(tab[i,] != 0)
  }

  return(condensed_views)
}
