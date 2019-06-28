# Boosted-regression-forest
Own implementation of Regression Forest, tests, comparing to sklearn Regression Forest

Boosted Random Regression Forest - My own implementation of generating random regression forest. You can use it in two ways. 
boosted or not. Not boosted way: training observations are chosen randomly when we make next tree. Boosted way let you generate random forest where for next trees training observations are chosen with higher probability if already existing model made bigger mistakes on them. In experiments I used task: Wine Quality. Also repository contains tests in which I analyze different values of parameteres (number of trees in forest, minimum samples in leaf or number of observations which are used to train one tree). Tests were written for different splitting value(training set:testing set - 1:1, 7:3)  I compared my implementation with implementation which exists in sklearn library.

