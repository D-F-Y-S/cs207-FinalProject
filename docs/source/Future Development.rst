Future Development
======================

1. Optimization
----------------

One of the shortcomings we notice of our current design is that, during the calculation, the derivate/value of an expression at a single point may be evaluated multiple times. When the Expression tree is shallow, this doesn't have much effect on the computation time. However, when the Expression tree is deep, the time spent on redundant work will grow exponentially, which can be a serious problem. We may want to add a cache to our library , so that when the same derivative/value is queried, it is fetched from the cache instead of  being computed again. This can largely accelerate our library in the case of complex Expressions.

2. Extensions
--------------

Since most machine learning problems can be formulated as optimization problems, and optimization routines can make use of automatic differentiations, we can actually develop a machine learning library on the top of our library. Other possible extensions include: more visualization tools, more optimization methods, even higher-order derivative than second-order, a neural network framework based on backward mode automatic differntiation.

3. Improvement
---------------

If user wish to add additional feature for the DFYS-autodiff package, please go to our GitHub_ repository, fork the repository, make the improvement, and submit pull request to us. 

.. _GitHub:: https://github.com/D-F-Y-S/cs207-FinalProject 