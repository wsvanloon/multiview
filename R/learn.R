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

# Learn is a generic function to train a learner on multi-view input data,
# and produce learned functions and (optionally) the matrix of cross-validated
# predictions. It uses switch() to apply the correct learner, depending on
# argument 'type'.

learn <- function(X, y, views, type, generate.CVs=TRUE, ...){
  switch(type,
         StaPLR = StaPLR(X, y, view=views, skip.meta = TRUE, skip.cv = !generate.CVs, ...))
}
