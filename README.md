# R-package 'multiview'

Methods for high-dimensional multi-view learning. The multiview package currently provides functions to fit stacked penalized logistic regression (StaPLR) models, which are a special case of multi-view stacking (MVS).

For more information about the StaPLR and MVS methods, see https://doi.org/10.1016/j.inffus.2020.03.007.

## Installation

Using the 'devtools' package:

~~~ r
devtools::install_gitlab("wsvanloon/multiview")
~~~

## Using 'multiview'

The two main functions are:

- StaPLR() - fits penalized and stacked penalized logistic regression models models up to two levels. Can be used to generate cross-validated predictions.
- MVS() - fits StaPLR models with >= 2 levels.

Objects returned by either function have associated coef() and predict() methods. 