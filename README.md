# German Credit Classifier

## Overview

This repository is a pattern recognition system that minimizes the risk faced by German banks when determining the creditworthiness of loan applicants. The original dataset includes 1000 data samples made up of a mixed feature set of 20 categorical and numerical features. In order to develop an optimal pattern recognition system, I started with an exploratory data analysis to visually identify data patterns and areas for data simplification. I applied preprocessing by encoding categorical data into numerical values, accounting for missing data, and developing a method for standardizing data. For the purpose of comparison, I designed a baseline classifier using only prior probabilities based on the creditworthy classifications from the original dataset. I then developed three more advanced classifiers: (1) Gaussian Naïve Bayes, (2) k-Nearest Neighbor, and (3) Support Vector Machine. After a series of rigorous 10-fold cross validation on the training data (80%), the Support Vector Classifier was selected and applied to the remaining test data (20%). Ultimately, the Support Vector Classifier achieved an accuracy of 80.0% on the test data with an F1 score of 0.86 for class 1 and 0.64 for class 2.

For more details about the project, see the report.pdf file. 

## Tech

This project has the following prerequisites: 

- [scikit-learn](http://scikit-learn.org/stable/) - Python machine learning library
- [NumPy](http://www.numpy.org/) - Python library for scientific computing
- [Pandas](http://pandas.pydata.org/pandas-docs/stable/) - Python-based data analysis toolkit
