ST558 Project 2
================
Yu Jiang
7/3/2020

# Introduction

## Describe the Data

Many articles have been published by Mashable in the past two years and
this original dataset can be found [this
website](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity).

There are 61 variables in total from the dataset above: 58 predictive
attributes, 2 non-predictive and 1 goal field. More details about used
data will be discussed later.

## the Purpose of Analysis

We are interested in predicting the popularity of a given article and
thus we are going to treat *shares*, an index of the article popularity,
as the response of models and some other variables as predictors.

After fitting the linear and non-linear models respectively, we can
choose the best one by comparing some numeric values, for example, AIC,
and then we can make a conclusion that which predictors contribute most
to the target, *shares*.

## Methods

After slicing data into two sets, a training set (70% of the data) and
the test set (30% of the data), we are going to use the threshold of
1400 shares to create two classes: if the number of shares is greater
than 1400, then the article is classified as popular; if the number of
shares is less than or equal to 1400, then the article is classified as
unpopular in order to formulate a classification problem.

Two types of models will be used to predict the shares. One is about an
ensemble model (bagged trees, random forests, or boosted trees) and the
other is about a linear regression model.

# Data Study

## Description of the Used Data

Since we are going to predict the popularity of an article, we have
chosen *shares* as the response. To start with, I am going to use all
the variables as the

## Data Split

``` r
# Load the library and set seed for reproducibility
library(tidyverse)
set.seed(2)

# Read data and select 'Monday'
news_pop <- read_csv('OnlineNewsPopularity.csv')[, -c(1, 2)]

# Convert 'shares' into factor: <= 1400: unpopular, > 1400: popular
news_pop$shares[news_pop$shares <= 1400] <- 0
news_pop$shares[news_pop$shares > 1400] <- 1
news_pop$shares <- factor(news_pop$shares, 
                          levels = c(0, 1), 
                          labels = c('unpopular', 'popular'))

# Just consider 'Monday' at first 
monday <- news_pop %>% select(-c("weekday_is_tuesday", "weekday_is_wednesday",
                             "weekday_is_thursday", "weekday_is_friday",
                             "weekday_is_saturday", "weekday_is_sunday"))
```

## Data Summarizations

# Modeling

## Ensemble Model Fit

## Linear Regression Fit

## Model Selection

# Conclusions
