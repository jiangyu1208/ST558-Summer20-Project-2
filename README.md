ST558 Project 2
================
Yu Jiang
7/3/2020

# Introduction

## Describe the Data

Many articles have been published by Mashable over two years from
seventh Jan, 2013 and this original dataset can be found [this
website](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity).

There are 61 variables in total from the dataset above: 58 predictive
attributes, 2 non-predictive and 1 goal field. More details about used
data will be discussed later.

## the Purpose of Analysis

We are interested in predicting the popularity of a given article and
thus we are going to treat *shares*, an index of the article popularity,
as the response of models and some other variables as predictors.

After fitting the linear and ensemble models respectively, we can choose
the best two different types of models by comparing some numeric values,
for example, AIC, and then we can make a conclusion that which
predictors contribute most to the target, *shares*.

## Methods

First, we can slice data into two sets, a training set (70% of the data)
and the test set (30% of the data) and then we are going to fit two
types of models to predict the shares.

Second, we are going to consider two types of models.

For the linear regression model, We will try to predict popularity in
two ways. The first target is as it is recorded, as the number of times
the articles is shared over the period. The second target is a binary
variable which discretizes the above. Following Fernandes et
al. (section 3.1), an article is considered popular if it exceeds 1400
shares. Therefore, we are going to use the threshold of 1400 shares to
create two classes: if the number of shares is greater than 1400, then
the article is classified as popular; if the number of shares is less
than or equal to 1400, then the article is classified as unpopular in
order to formulate a classification problem.

We are going to use multiple linear regression for the first task and
binary logistic regression for the second one.

For the ensemble models, we will also fit the two different
models(bagged trees and boosted trees) and choose the best one after
comparing the misclassification rate.

Finally, we will decide the best two different types of models for our
target.

# Data Study

## Description of the Used Data

Since we are going to predict the popularity of an article, we have
chosen *shares* as the response. To start with, for the linear
regression model, I am going to use all the variables as the predictors
and then gradually delete some unnecessary variables after fitting the
models.

Similarly, for the ensemble model, I am also going to start to fit these
models (bagged tree, random forest, boosted trees) with all predictors
from the train set. After comparing the misclassification rate, the best
model can be chosen.

## Data Split

``` r
# Load the all libraries and set seed for reproducibility
library(tidyverse)
library(ggplot2)
library(caret)
library(randomForest)
library(tree)
library(gbm)

set.seed(2)

# Read data and remove the first two columns
news_pop <- read_csv('OnlineNewsPopularity.csv')[, -c(1, 2)]

# Just consider 'Monday' at first 
monday <- news_pop %>% select(-c("weekday_is_tuesday", "weekday_is_wednesday",
                             "weekday_is_thursday", "weekday_is_friday",
                             "weekday_is_saturday", "weekday_is_sunday"))

# Check if the data has any missing values
sum(is.na(monday))
```

    ## [1] 0

``` r
# Split the 'Monday' data, 70% of the data for training and the rest for testing
train <- sample(1:nrow(monday), size = nrow(monday) * 0.7)
test <- dplyr::setdiff(1:nrow(monday), train)

mondayTrain <- monday[train, ]
mondayTest <- monday[test, ]
```

## Data Summarizations

### Response Variable

``` r
# Basic summary of the 'monday' data
summary(mondayTrain)
```

    ##  n_tokens_title n_tokens_content n_unique_tokens    n_non_stop_words  
    ##  Min.   : 3.0   Min.   :   0.0   Min.   :  0.0000   Min.   :   0.000  
    ##  1st Qu.: 9.0   1st Qu.: 245.0   1st Qu.:  0.4711   1st Qu.:   1.000  
    ##  Median :10.0   Median : 408.0   Median :  0.5401   Median :   1.000  
    ##  Mean   :10.4   Mean   : 544.2   Mean   :  0.5569   Mean   :   1.009  
    ##  3rd Qu.:12.0   3rd Qu.: 716.0   3rd Qu.:  0.6097   3rd Qu.:   1.000  
    ##  Max.   :20.0   Max.   :8474.0   Max.   :701.0000   Max.   :1042.000  
    ##  n_non_stop_unique_tokens   num_hrefs      num_self_hrefs       num_imgs      
    ##  Min.   :  0.0000         Min.   :  0.00   Min.   :  0.000   Min.   :  0.000  
    ##  1st Qu.:  0.6259         1st Qu.:  4.00   1st Qu.:  1.000   1st Qu.:  1.000  
    ##  Median :  0.6911         Median :  7.00   Median :  3.000   Median :  1.000  
    ##  Mean   :  0.6973         Mean   : 10.86   Mean   :  3.282   Mean   :  4.547  
    ##  3rd Qu.:  0.7553         3rd Qu.: 13.00   3rd Qu.:  4.000   3rd Qu.:  4.000  
    ##  Max.   :650.0000         Max.   :304.00   Max.   :116.000   Max.   :128.000  
    ##    num_videos    average_token_length  num_keywords   
    ##  Min.   : 0.00   Min.   :0.000        Min.   : 1.000  
    ##  1st Qu.: 0.00   1st Qu.:4.477        1st Qu.: 6.000  
    ##  Median : 0.00   Median :4.664        Median : 7.000  
    ##  Mean   : 1.25   Mean   :4.552        Mean   : 7.224  
    ##  3rd Qu.: 1.00   3rd Qu.:4.854        3rd Qu.: 9.000  
    ##  Max.   :91.00   Max.   :8.042        Max.   :10.000  
    ##  data_channel_is_lifestyle data_channel_is_entertainment data_channel_is_bus
    ##  Min.   :0.00000           Min.   :0.0000                Min.   :0.0000     
    ##  1st Qu.:0.00000           1st Qu.:0.0000                1st Qu.:0.0000     
    ##  Median :0.00000           Median :0.0000                Median :0.0000     
    ##  Mean   :0.05283           Mean   :0.1791                Mean   :0.1567     
    ##  3rd Qu.:0.00000           3rd Qu.:0.0000                3rd Qu.:0.0000     
    ##  Max.   :1.00000           Max.   :1.0000                Max.   :1.0000     
    ##  data_channel_is_socmed data_channel_is_tech data_channel_is_world
    ##  Min.   :0.0000         Min.   :0.0000       Min.   :0.0000       
    ##  1st Qu.:0.0000         1st Qu.:0.0000       1st Qu.:0.0000       
    ##  Median :0.0000         Median :0.0000       Median :0.0000       
    ##  Mean   :0.0582         Mean   :0.1855       Mean   :0.2117       
    ##  3rd Qu.:0.0000         3rd Qu.:0.0000       3rd Qu.:0.0000       
    ##  Max.   :1.0000         Max.   :1.0000       Max.   :1.0000       
    ##    kw_min_min       kw_max_min       kw_avg_min        kw_min_max    
    ##  Min.   : -1.00   Min.   :     0   Min.   :   -1.0   Min.   :     0  
    ##  1st Qu.: -1.00   1st Qu.:   445   1st Qu.:  141.3   1st Qu.:     0  
    ##  Median : -1.00   Median :   657   Median :  234.5   Median :  1400  
    ##  Mean   : 26.36   Mean   :  1158   Mean   :  313.5   Mean   : 13686  
    ##  3rd Qu.:  4.00   3rd Qu.:  1000   3rd Qu.:  356.4   3rd Qu.:  7900  
    ##  Max.   :377.00   Max.   :298400   Max.   :42827.9   Max.   :843300  
    ##    kw_max_max       kw_avg_max       kw_min_avg     kw_max_avg    
    ##  Min.   :     0   Min.   :     0   Min.   :  -1   Min.   :     0  
    ##  1st Qu.:843300   1st Qu.:172494   1st Qu.:   0   1st Qu.:  3560  
    ##  Median :843300   Median :243933   Median :1019   Median :  4355  
    ##  Mean   :751158   Mean   :258434   Mean   :1117   Mean   :  5653  
    ##  3rd Qu.:843300   3rd Qu.:329894   3rd Qu.:2061   3rd Qu.:  6015  
    ##  Max.   :843300   Max.   :843300   Max.   :3613   Max.   :298400  
    ##    kw_avg_avg    self_reference_min_shares self_reference_max_shares
    ##  Min.   :    0   Min.   :     0            Min.   :     0           
    ##  1st Qu.: 2380   1st Qu.:   642            1st Qu.:  1100           
    ##  Median : 2869   Median :  1200            Median :  2900           
    ##  Mean   : 3135   Mean   :  4057            Mean   : 10326           
    ##  3rd Qu.: 3600   3rd Qu.:  2700            3rd Qu.:  8100           
    ##  Max.   :43568   Max.   :843300            Max.   :843300           
    ##  self_reference_avg_sharess weekday_is_monday   is_weekend    
    ##  Min.   :     0.0           Min.   :0.0000    Min.   :0.0000  
    ##  1st Qu.:   985.4           1st Qu.:0.0000    1st Qu.:0.0000  
    ##  Median :  2200.0           Median :0.0000    Median :0.0000  
    ##  Mean   :  6451.3           Mean   :0.1663    Mean   :0.1312  
    ##  3rd Qu.:  5200.0           3rd Qu.:0.0000    3rd Qu.:0.0000  
    ##  Max.   :843300.0           Max.   :1.0000    Max.   :1.0000  
    ##      LDA_00            LDA_01            LDA_02            LDA_03       
    ##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
    ##  1st Qu.:0.02505   1st Qu.:0.02501   1st Qu.:0.02857   1st Qu.:0.02679  
    ##  Median :0.03339   Median :0.03335   Median :0.04000   Median :0.04000  
    ##  Mean   :0.18397   Mean   :0.14212   Mean   :0.21573   Mean   :0.22371  
    ##  3rd Qu.:0.24021   3rd Qu.:0.15160   3rd Qu.:0.33048   3rd Qu.:0.37476  
    ##  Max.   :0.92000   Max.   :0.92595   Max.   :0.92000   Max.   :0.92554  
    ##      LDA_04        global_subjectivity global_sentiment_polarity
    ##  Min.   :0.00000   Min.   :0.0000      Min.   :-0.37500         
    ##  1st Qu.:0.02857   1st Qu.:0.3961      1st Qu.: 0.05758         
    ##  Median :0.04073   Median :0.4538      Median : 0.11947         
    ##  Mean   :0.23443   Mean   :0.4438      Mean   : 0.11940         
    ##  3rd Qu.:0.40546   3rd Qu.:0.5086      3rd Qu.: 0.17801         
    ##  Max.   :0.92719   Max.   :1.0000      Max.   : 0.72784         
    ##  global_rate_positive_words global_rate_negative_words rate_positive_words
    ##  Min.   :0.00000            Min.   :0.000000           Min.   :0.0000     
    ##  1st Qu.:0.02841            1st Qu.:0.009579           1st Qu.:0.6000     
    ##  Median :0.03904            Median :0.015353           Median :0.7108     
    ##  Mean   :0.03965            Mean   :0.016630           Mean   :0.6828     
    ##  3rd Qu.:0.05028            3rd Qu.:0.021739           3rd Qu.:0.8000     
    ##  Max.   :0.15549            Max.   :0.162037           Max.   :1.0000     
    ##  rate_negative_words avg_positive_polarity min_positive_polarity
    ##  Min.   :0.0000      Min.   :0.0000        Min.   :0.00000      
    ##  1st Qu.:0.1857      1st Qu.:0.3062        1st Qu.:0.05000      
    ##  Median :0.2800      Median :0.3586        Median :0.10000      
    ##  Mean   :0.2882      Mean   :0.3543        Mean   :0.09583      
    ##  3rd Qu.:0.3846      3rd Qu.:0.4114        3rd Qu.:0.10000      
    ##  Max.   :1.0000      Max.   :1.0000        Max.   :1.00000      
    ##  max_positive_polarity avg_negative_polarity min_negative_polarity
    ##  Min.   :0.0000        Min.   :-1.0000       Min.   :-1.0000      
    ##  1st Qu.:0.6000        1st Qu.:-0.3294       1st Qu.:-0.7000      
    ##  Median :0.8000        Median :-0.2539       Median :-0.5000      
    ##  Mean   :0.7576        Mean   :-0.2602       Mean   :-0.5228      
    ##  3rd Qu.:1.0000        3rd Qu.:-0.1873       3rd Qu.:-0.3000      
    ##  Max.   :1.0000        Max.   : 0.0000       Max.   : 0.0000      
    ##  max_negative_polarity title_subjectivity title_sentiment_polarity
    ##  Min.   :-1.0000       Min.   :0.0000     Min.   :-1.00000        
    ##  1st Qu.:-0.1250       1st Qu.:0.0000     1st Qu.: 0.00000        
    ##  Median :-0.1000       Median :0.1500     Median : 0.00000        
    ##  Mean   :-0.1079       Mean   :0.2831     Mean   : 0.07075        
    ##  3rd Qu.:-0.0500       3rd Qu.:0.5000     3rd Qu.: 0.15000        
    ##  Max.   : 0.0000       Max.   :1.0000     Max.   : 1.00000        
    ##  abs_title_subjectivity abs_title_sentiment_polarity     shares      
    ##  Min.   :0.0000         Min.   :0.0000               Min.   :     1  
    ##  1st Qu.:0.1667         1st Qu.:0.0000               1st Qu.:   949  
    ##  Median :0.5000         Median :0.0000               Median :  1400  
    ##  Mean   :0.3411         Mean   :0.1565               Mean   :  3444  
    ##  3rd Qu.:0.5000         3rd Qu.:0.2500               3rd Qu.:  2800  
    ##  Max.   :0.5000         Max.   :1.0000               Max.   :690400

``` r
# Histogram of 'shares'
ggplot(data = mondayTrain, aes(x = shares)) +
  geom_histogram()
```

![](README_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

Since the histogram of share is highly skewed, we can consider to use
the log transformation to obtain a new histogram for shares.

``` r
x <- log(mondayTrain$shares)

ggplot(data = mondayTrain, aes(x)) +
  geom_histogram() + 
  xlab('Log(shares)')
```

![](README_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

After comparing these two histograms, we decide to use the log(shares)
as our response for the multiple linear regression model.

# Modeling

## Linear Regression Fit

### Multiple Linear Regression

First, we fit the multiple linear regression with all predictors.

``` r
# Multiple linear model 1
fit.1 <- lm(log(shares) ~ . , data = mondayTrain)
summary(fit.1)
```

    ## 
    ## Call:
    ## lm(formula = log(shares) ~ ., data = mondayTrain)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -8.1134 -0.5503 -0.1678  0.3899  5.6290 
    ## 
    ## Coefficients:
    ##                                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                    2.639e+02  5.397e+02   0.489 0.624871    
    ## n_tokens_title                 6.354e-03  2.602e-03   2.442 0.014599 *  
    ## n_tokens_content               6.513e-05  2.051e-05   3.175 0.001500 ** 
    ## n_unique_tokens                2.691e-01  1.747e-01   1.540 0.123561    
    ## n_non_stop_words              -2.753e-01  5.207e-01  -0.529 0.596960    
    ## n_non_stop_unique_tokens      -2.420e-01  1.483e-01  -1.632 0.102695    
    ## num_hrefs                      4.031e-03  6.033e-04   6.682 2.40e-11 ***
    ## num_self_hrefs                -8.713e-03  1.644e-03  -5.299 1.18e-07 ***
    ## num_imgs                       2.140e-03  8.119e-04   2.636 0.008387 ** 
    ## num_videos                     1.492e-03  1.424e-03   1.048 0.294571    
    ## average_token_length          -9.422e-02  2.208e-02  -4.268 1.98e-05 ***
    ## num_keywords                   1.541e-02  3.371e-03   4.572 4.85e-06 ***
    ## data_channel_is_lifestyle     -1.128e-01  3.602e-02  -3.131 0.001746 ** 
    ## data_channel_is_entertainment -1.860e-01  2.325e-02  -8.000 1.29e-15 ***
    ## data_channel_is_bus           -1.694e-01  3.506e-02  -4.832 1.36e-06 ***
    ## data_channel_is_socmed         1.727e-01  3.417e-02   5.055 4.33e-07 ***
    ## data_channel_is_tech           1.172e-01  3.391e-02   3.455 0.000552 ***
    ## data_channel_is_world         -4.871e-02  3.435e-02  -1.418 0.156146    
    ## kw_min_min                     8.707e-04  1.479e-04   5.888 3.96e-09 ***
    ## kw_max_min                     1.634e-05  4.296e-06   3.803 0.000143 ***
    ## kw_avg_min                    -1.250e-04  2.604e-05  -4.801 1.59e-06 ***
    ## kw_min_max                    -2.991e-07  1.062e-07  -2.816 0.004859 ** 
    ## kw_max_max                     1.926e-08  5.271e-08   0.365 0.714896    
    ## kw_avg_max                    -3.184e-07  7.592e-08  -4.194 2.74e-05 ***
    ## kw_min_avg                    -5.063e-05  6.921e-06  -7.315 2.64e-13 ***
    ## kw_max_avg                    -4.300e-05  2.400e-06 -17.917  < 2e-16 ***
    ## kw_avg_avg                     3.418e-04  1.335e-05  25.604  < 2e-16 ***
    ## self_reference_min_shares      7.314e-07  6.751e-07   1.083 0.278602    
    ## self_reference_max_shares      1.819e-07  3.689e-07   0.493 0.621965    
    ## self_reference_avg_sharess     1.030e-06  9.387e-07   1.097 0.272747    
    ## weekday_is_monday              4.473e-02  1.437e-02   3.113 0.001854 ** 
    ## is_weekend                     2.777e-01  1.597e-02  17.384  < 2e-16 ***
    ## LDA_00                        -2.571e+02  5.397e+02  -0.476 0.633868    
    ## LDA_01                        -2.574e+02  5.397e+02  -0.477 0.633364    
    ## LDA_02                        -2.575e+02  5.397e+02  -0.477 0.633258    
    ## LDA_03                        -2.574e+02  5.397e+02  -0.477 0.633394    
    ## LDA_04                        -2.573e+02  5.397e+02  -0.477 0.633545    
    ## global_subjectivity            4.023e-01  7.713e-02   5.216 1.84e-07 ***
    ## global_sentiment_polarity     -1.988e-01  1.523e-01  -1.305 0.191964    
    ## global_rate_positive_words    -9.773e-01  6.556e-01  -1.491 0.136057    
    ## global_rate_negative_words    -2.058e-01  1.265e+00  -0.163 0.870745    
    ## rate_positive_words            5.478e-01  5.082e-01   1.078 0.281057    
    ## rate_negative_words            4.781e-01  5.126e-01   0.933 0.350969    
    ## avg_positive_polarity          5.938e-02  1.247e-01   0.476 0.633812    
    ## min_positive_polarity         -4.186e-01  1.031e-01  -4.059 4.94e-05 ***
    ## max_positive_polarity         -1.845e-02  3.913e-02  -0.472 0.637276    
    ## avg_negative_polarity         -1.837e-01  1.141e-01  -1.610 0.107455    
    ## min_negative_polarity          3.137e-02  4.165e-02   0.753 0.451306    
    ## max_negative_polarity         -4.896e-02  9.506e-02  -0.515 0.606531    
    ## title_subjectivity             6.907e-02  2.494e-02   2.770 0.005607 ** 
    ## title_sentiment_polarity       8.616e-02  2.271e-02   3.793 0.000149 ***
    ## abs_title_subjectivity         1.308e-01  3.317e-02   3.944 8.03e-05 ***
    ## abs_title_sentiment_polarity  -3.531e-03  3.586e-02  -0.098 0.921579    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.8756 on 27697 degrees of freedom
    ## Multiple R-squared:  0.128,  Adjusted R-squared:  0.1264 
    ## F-statistic: 78.21 on 52 and 27697 DF,  p-value: < 2.2e-16

After deleting some predictors whose p-value is larger than 0.05, we can
obtain two fit models as follows,

``` r
# Multiple linear model 2 
fit.2 <- lm(log(shares) ~ . - n_unique_tokens - n_non_stop_words - 
              n_non_stop_unique_tokens - num_videos - 
              data_channel_is_world- kw_max_max - 
              self_reference_min_shares -  self_reference_max_shares - 
              self_reference_avg_sharess - 
              LDA_00 - LDA_01 - LDA_02 - LDA_03 - LDA_04 -
              global_sentiment_polarity - global_rate_positive_words -
              global_rate_negative_words - rate_positive_words - 
              rate_negative_words - avg_positive_polarity - 
              max_positive_polarity - avg_negative_polarity - 
              min_negative_polarity - max_negative_polarity - 
              abs_title_sentiment_polarity, data = mondayTrain)
summary(fit.2)
```

    ## 
    ## Call:
    ## lm(formula = log(shares) ~ . - n_unique_tokens - n_non_stop_words - 
    ##     n_non_stop_unique_tokens - num_videos - data_channel_is_world - 
    ##     kw_max_max - self_reference_min_shares - self_reference_max_shares - 
    ##     self_reference_avg_sharess - LDA_00 - LDA_01 - LDA_02 - LDA_03 - 
    ##     LDA_04 - global_sentiment_polarity - global_rate_positive_words - 
    ##     global_rate_negative_words - rate_positive_words - rate_negative_words - 
    ##     avg_positive_polarity - max_positive_polarity - avg_negative_polarity - 
    ##     min_negative_polarity - max_negative_polarity - abs_title_sentiment_polarity, 
    ##     data = mondayTrain)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -8.2589 -0.5553 -0.1708  0.3904  5.6290 
    ## 
    ## Coefficients:
    ##                                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                    6.362e+00  5.892e-02 107.974  < 2e-16 ***
    ## n_tokens_title                 6.313e-03  2.592e-03   2.436 0.014871 *  
    ## n_tokens_content               3.720e-05  1.392e-05   2.672 0.007534 ** 
    ## num_hrefs                      4.118e-03  5.747e-04   7.165 7.96e-13 ***
    ## num_self_hrefs                -7.653e-03  1.599e-03  -4.787 1.71e-06 ***
    ## num_imgs                       2.527e-03  7.208e-04   3.505 0.000457 ***
    ## average_token_length          -5.548e-02  8.575e-03  -6.470 9.98e-11 ***
    ## num_keywords                   1.595e-02  3.314e-03   4.814 1.49e-06 ***
    ## data_channel_is_lifestyle      2.779e-02  2.530e-02   1.098 0.272112    
    ## data_channel_is_entertainment -1.258e-01  1.587e-02  -7.927 2.32e-15 ***
    ## data_channel_is_bus            1.038e-01  1.689e-02   6.148 7.95e-10 ***
    ## data_channel_is_socmed         3.491e-01  2.450e-02  14.252  < 2e-16 ***
    ## data_channel_is_tech           2.523e-01  1.607e-02  15.697  < 2e-16 ***
    ## kw_min_min                     9.644e-04  9.423e-05  10.235  < 2e-16 ***
    ## kw_max_min                     1.668e-05  4.288e-06   3.889 0.000101 ***
    ## kw_avg_min                    -1.352e-04  2.596e-05  -5.208 1.92e-07 ***
    ## kw_min_max                    -3.469e-07  1.052e-07  -3.298 0.000975 ***
    ## kw_avg_max                    -2.418e-07  6.956e-08  -3.476 0.000509 ***
    ## kw_min_avg                    -5.934e-05  6.722e-06  -8.828  < 2e-16 ***
    ## kw_max_avg                    -4.584e-05  2.243e-06 -20.439  < 2e-16 ***
    ## kw_avg_avg                     3.741e-04  1.187e-05  31.523  < 2e-16 ***
    ## weekday_is_monday              4.608e-02  1.441e-02   3.196 0.001393 ** 
    ## is_weekend                     2.774e-01  1.600e-02  17.332  < 2e-16 ***
    ## global_subjectivity            4.868e-01  6.050e-02   8.046 8.88e-16 ***
    ## min_positive_polarity         -3.552e-01  8.071e-02  -4.401 1.08e-05 ***
    ## title_subjectivity             6.813e-02  1.927e-02   3.536 0.000407 ***
    ## title_sentiment_polarity       7.075e-02  2.076e-02   3.409 0.000654 ***
    ## abs_title_subjectivity         1.318e-01  3.283e-02   4.016 5.95e-05 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.8791 on 27722 degrees of freedom
    ## Multiple R-squared:  0.1202, Adjusted R-squared:  0.1194 
    ## F-statistic: 140.3 on 27 and 27722 DF,  p-value: < 2.2e-16

``` r
# Multiple linear model 3
fit.3 <- lm(log(shares) ~ . - n_unique_tokens - n_non_stop_words - 
              n_non_stop_unique_tokens - num_videos - 
              data_channel_is_lifestyle - 
              data_channel_is_world - kw_max_max - 
              self_reference_min_shares -  self_reference_max_shares - 
              self_reference_avg_sharess - 
              LDA_00 - LDA_01 - LDA_02 - LDA_03 - LDA_04 -
              global_sentiment_polarity - global_rate_positive_words -
              global_rate_negative_words - rate_positive_words - 
              rate_negative_words - avg_positive_polarity - 
              max_positive_polarity - avg_negative_polarity - 
              min_negative_polarity - max_negative_polarity - 
              abs_title_sentiment_polarity, data = mondayTrain)
summary(fit.3)
```

    ## 
    ## Call:
    ## lm(formula = log(shares) ~ . - n_unique_tokens - n_non_stop_words - 
    ##     n_non_stop_unique_tokens - num_videos - data_channel_is_lifestyle - 
    ##     data_channel_is_world - kw_max_max - self_reference_min_shares - 
    ##     self_reference_max_shares - self_reference_avg_sharess - 
    ##     LDA_00 - LDA_01 - LDA_02 - LDA_03 - LDA_04 - global_sentiment_polarity - 
    ##     global_rate_positive_words - global_rate_negative_words - 
    ##     rate_positive_words - rate_negative_words - avg_positive_polarity - 
    ##     max_positive_polarity - avg_negative_polarity - min_negative_polarity - 
    ##     max_negative_polarity - abs_title_sentiment_polarity, data = mondayTrain)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -8.2607 -0.5554 -0.1702  0.3908  5.6301 
    ## 
    ## Coefficients:
    ##                                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                    6.366e+00  5.878e-02 108.312  < 2e-16 ***
    ## n_tokens_title                 6.200e-03  2.590e-03   2.394 0.016675 *  
    ## n_tokens_content               3.806e-05  1.390e-05   2.738 0.006178 ** 
    ## num_hrefs                      4.115e-03  5.747e-04   7.160 8.24e-13 ***
    ## num_self_hrefs                -7.696e-03  1.598e-03  -4.815 1.48e-06 ***
    ## num_imgs                       2.494e-03  7.202e-04   3.464 0.000534 ***
    ## average_token_length          -5.581e-02  8.569e-03  -6.513 7.51e-11 ***
    ## num_keywords                   1.604e-02  3.313e-03   4.842 1.29e-06 ***
    ## data_channel_is_entertainment -1.295e-01  1.551e-02  -8.352  < 2e-16 ***
    ## data_channel_is_bus            1.011e-01  1.671e-02   6.053 1.44e-09 ***
    ## data_channel_is_socmed         3.449e-01  2.420e-02  14.255  < 2e-16 ***
    ## data_channel_is_tech           2.484e-01  1.568e-02  15.846  < 2e-16 ***
    ## kw_min_min                     9.606e-04  9.417e-05  10.201  < 2e-16 ***
    ## kw_max_min                     1.679e-05  4.287e-06   3.917 8.97e-05 ***
    ## kw_avg_min                    -1.358e-04  2.595e-05  -5.234 1.67e-07 ***
    ## kw_min_max                    -3.398e-07  1.050e-07  -3.236 0.001212 ** 
    ## kw_avg_max                    -2.544e-07  6.862e-08  -3.707 0.000210 ***
    ## kw_min_avg                    -5.950e-05  6.720e-06  -8.853  < 2e-16 ***
    ## kw_max_avg                    -4.603e-05  2.237e-06 -20.581  < 2e-16 ***
    ## kw_avg_avg                     3.753e-04  1.182e-05  31.762  < 2e-16 ***
    ## weekday_is_monday              4.613e-02  1.441e-02   3.200 0.001376 ** 
    ## is_weekend                     2.775e-01  1.600e-02  17.339  < 2e-16 ***
    ## global_subjectivity            4.915e-01  6.035e-02   8.145 3.96e-16 ***
    ## min_positive_polarity         -3.572e-01  8.069e-02  -4.427 9.59e-06 ***
    ## title_subjectivity             6.749e-02  1.926e-02   3.504 0.000459 ***
    ## title_sentiment_polarity       7.170e-02  2.074e-02   3.457 0.000546 ***
    ## abs_title_subjectivity         1.315e-01  3.283e-02   4.004 6.25e-05 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.8791 on 27723 degrees of freedom
    ## Multiple R-squared:  0.1202, Adjusted R-squared:  0.1194 
    ## F-statistic: 145.7 on 26 and 27723 DF,  p-value: < 2.2e-16

After deleting some unnecessary predictors twice, we can see that for
multiple linear model 3, all p-values are smaller than 0.05, which means
that all predictors are statistically significant.

### Binary Logistic Regression

Use *shares* to create the binary variable : diving the shares into two
groups (\<1400 and \>= 1400) and also start using all predictors and
delete some predictors whose p-value is larger than 0.05.

``` r
# Convert 'shares' into factor: <= 1400: unpopular, > 1400: popular
mondayTrain$shares[mondayTrain$shares <= 1400] <- 0
mondayTrain$shares[mondayTrain$shares > 1400] <- 1

mondayTrain$shares <- as.factor(mondayTrain$shares)
  
# GLM model 1
glmFit.1 <- glm(shares~. , data = mondayTrain, family = 'binomial')
summary(glmFit.1)
```

    ## 
    ## Call:
    ## glm(formula = shares ~ ., family = "binomial", data = mondayTrain)
    ## 
    ## Deviance Residuals: 
    ##    Min      1Q  Median      3Q     Max  
    ## -4.564  -1.034  -0.632   1.069   2.037  
    ## 
    ## Coefficients:
    ##                                 Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                    1.191e+04  1.142e+05   0.104 0.916933    
    ## n_tokens_title                -1.671e-03  6.344e-03  -0.263 0.792186    
    ## n_tokens_content               1.715e-04  5.264e-05   3.258 0.001121 ** 
    ## n_unique_tokens                2.651e-02  4.301e-01   0.062 0.950849    
    ## n_non_stop_words              -1.106e+01  1.096e+02  -0.101 0.919656    
    ## n_non_stop_unique_tokens      -6.147e-01  3.635e-01  -1.691 0.090782 .  
    ## num_hrefs                      8.299e-03  1.562e-03   5.312 1.08e-07 ***
    ## num_self_hrefs                -2.228e-02  4.132e-03  -5.392 6.98e-08 ***
    ## num_imgs                       8.175e-04  2.006e-03   0.407 0.683678    
    ## num_videos                    -4.141e-04  3.507e-03  -0.118 0.906013    
    ## average_token_length          -1.085e-01  5.387e-02  -2.013 0.044097 *  
    ## num_keywords                   4.750e-02  8.268e-03   5.745 9.19e-09 ***
    ## data_channel_is_lifestyle     -1.544e-01  8.851e-02  -1.745 0.081008 .  
    ## data_channel_is_entertainment -3.411e-01  5.629e-02  -6.059 1.37e-09 ***
    ## data_channel_is_bus           -3.125e-01  8.623e-02  -3.624 0.000290 ***
    ## data_channel_is_socmed         7.302e-01  8.601e-02   8.489  < 2e-16 ***
    ## data_channel_is_tech           4.722e-01  8.278e-02   5.704 1.17e-08 ***
    ## data_channel_is_world         -6.098e-02  8.366e-02  -0.729 0.466045    
    ## kw_min_min                     1.715e-03  3.651e-04   4.696 2.65e-06 ***
    ## kw_max_min                     3.806e-05  1.387e-05   2.744 0.006069 ** 
    ## kw_avg_min                    -2.556e-04  8.849e-05  -2.889 0.003868 ** 
    ## kw_min_max                    -7.484e-07  2.549e-07  -2.936 0.003324 ** 
    ## kw_max_max                    -4.108e-07  1.296e-07  -3.170 0.001525 ** 
    ## kw_avg_max                    -5.614e-07  1.868e-07  -3.005 0.002656 ** 
    ## kw_min_avg                    -7.874e-05  1.730e-05  -4.551 5.34e-06 ***
    ## kw_max_avg                    -9.524e-05  6.355e-06 -14.986  < 2e-16 ***
    ## kw_avg_avg                     7.247e-04  3.505e-05  20.677  < 2e-16 ***
    ## self_reference_min_shares      2.245e-06  2.031e-06   1.105 0.269060    
    ## self_reference_max_shares      6.878e-07  9.922e-07   0.693 0.488162    
    ## self_reference_avg_sharess     1.616e-06  2.531e-06   0.638 0.523219    
    ## weekday_is_monday              6.223e-02  3.480e-02   1.788 0.073719 .  
    ## is_weekend                     8.438e-01  4.055e-02  20.809  < 2e-16 ***
    ## LDA_00                        -1.191e+04  1.142e+05  -0.104 0.916927    
    ## LDA_01                        -1.191e+04  1.142e+05  -0.104 0.916919    
    ## LDA_02                        -1.191e+04  1.142e+05  -0.104 0.916919    
    ## LDA_03                        -1.191e+04  1.142e+05  -0.104 0.916919    
    ## LDA_04                        -1.191e+04  1.142e+05  -0.104 0.916923    
    ## global_subjectivity            8.871e-01  1.888e-01   4.699 2.62e-06 ***
    ## global_sentiment_polarity     -2.614e-01  3.718e-01  -0.703 0.482005    
    ## global_rate_positive_words    -2.644e+00  1.601e+00  -1.652 0.098552 .  
    ## global_rate_negative_words     3.742e+00  3.120e+00   1.199 0.230464    
    ## rate_positive_words            1.169e+01  1.096e+02   0.107 0.915090    
    ## rate_negative_words            1.130e+01  1.096e+02   0.103 0.917912    
    ## avg_positive_polarity         -1.986e-01  3.042e-01  -0.653 0.513752    
    ## min_positive_polarity         -5.427e-01  2.519e-01  -2.154 0.031208 *  
    ## max_positive_polarity         -1.044e-02  9.540e-02  -0.109 0.912816    
    ## avg_negative_polarity         -2.844e-02  2.797e-01  -0.102 0.919007    
    ## min_negative_polarity          6.286e-02  1.024e-01   0.614 0.539260    
    ## max_negative_polarity         -2.713e-01  2.325e-01  -1.167 0.243362    
    ## title_subjectivity             1.891e-01  6.110e-02   3.095 0.001966 ** 
    ## title_sentiment_polarity       1.909e-01  5.580e-02   3.421 0.000624 ***
    ## abs_title_subjectivity         2.331e-01  8.124e-02   2.870 0.004108 ** 
    ## abs_title_sentiment_polarity  -1.059e-01  8.800e-02  -1.203 0.228860    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 38467  on 27749  degrees of freedom
    ## Residual deviance: 34854  on 27697  degrees of freedom
    ## AIC: 34960
    ## 
    ## Number of Fisher Scoring iterations: 10

``` r
# GLM model 2
glmFit.2 <- glm(shares ~ . - n_tokens_title - n_unique_tokens - 
                  n_non_stop_words - n_non_stop_unique_tokens - 
                  num_imgs - num_videos - data_channel_is_lifestyle - 
                  data_channel_is_world - self_reference_min_shares - 
                  self_reference_max_shares - self_reference_avg_sharess -
                  weekday_is_monday - LDA_00 - LDA_01 - LDA_02 - LDA_03 -
                  LDA_04 - global_sentiment_polarity -
                  global_rate_positive_words - rate_positive_words - 
                  rate_negative_words - avg_positive_polarity - 
                  max_positive_polarity - avg_negative_polarity -
                  min_negative_polarity - max_negative_polarity - 
                  abs_title_sentiment_polarity, data = mondayTrain, 
                family = 'binomial')
summary(glmFit.2)
```

    ## 
    ## Call:
    ## glm(formula = shares ~ . - n_tokens_title - n_unique_tokens - 
    ##     n_non_stop_words - n_non_stop_unique_tokens - num_imgs - 
    ##     num_videos - data_channel_is_lifestyle - data_channel_is_world - 
    ##     self_reference_min_shares - self_reference_max_shares - self_reference_avg_sharess - 
    ##     weekday_is_monday - LDA_00 - LDA_01 - LDA_02 - LDA_03 - LDA_04 - 
    ##     global_sentiment_polarity - global_rate_positive_words - 
    ##     rate_positive_words - rate_negative_words - avg_positive_polarity - 
    ##     max_positive_polarity - avg_negative_polarity - min_negative_polarity - 
    ##     max_negative_polarity - abs_title_sentiment_polarity, family = "binomial", 
    ##     data = mondayTrain)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -4.8600  -1.0397  -0.6478   1.0766   2.1171  
    ## 
    ## Coefficients:
    ##                                 Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                   -2.086e+00  1.476e-01 -14.130  < 2e-16 ***
    ## n_tokens_content               2.065e-04  3.409e-05   6.058 1.38e-09 ***
    ## num_hrefs                      8.610e-03  1.472e-03   5.848 4.99e-09 ***
    ## num_self_hrefs                -2.025e-02  3.962e-03  -5.112 3.19e-07 ***
    ## average_token_length          -9.489e-02  2.095e-02  -4.530 5.89e-06 ***
    ## num_keywords                   4.942e-02  8.148e-03   6.066 1.32e-09 ***
    ## data_channel_is_entertainment -3.389e-01  3.753e-02  -9.030  < 2e-16 ***
    ## data_channel_is_bus            4.097e-01  4.025e-02  10.179  < 2e-16 ***
    ## data_channel_is_socmed         1.137e+00  6.235e-02  18.230  < 2e-16 ***
    ## data_channel_is_tech           7.958e-01  3.808e-02  20.899  < 2e-16 ***
    ## kw_min_min                     1.711e-03  3.639e-04   4.702 2.57e-06 ***
    ## kw_max_min                     3.514e-05  1.402e-05   2.506 0.012222 *  
    ## kw_avg_min                    -2.454e-04  8.810e-05  -2.786 0.005336 ** 
    ## kw_min_max                    -7.253e-07  2.523e-07  -2.874 0.004047 ** 
    ## kw_max_max                    -4.097e-07  1.259e-07  -3.255 0.001132 ** 
    ## kw_avg_max                    -6.431e-07  1.755e-07  -3.664 0.000248 ***
    ## kw_min_avg                    -9.067e-05  1.650e-05  -5.496 3.88e-08 ***
    ## kw_max_avg                    -9.988e-05  5.849e-06 -17.075  < 2e-16 ***
    ## kw_avg_avg                     7.762e-04  3.009e-05  25.799  < 2e-16 ***
    ## is_weekend                     8.225e-01  3.976e-02  20.685  < 2e-16 ***
    ## global_subjectivity            8.158e-01  1.483e-01   5.502 3.75e-08 ***
    ## global_rate_negative_words    -1.143e+00  1.286e+00  -0.889 0.373992    
    ## min_positive_polarity         -7.637e-01  1.962e-01  -3.893 9.91e-05 ***
    ## title_subjectivity             1.280e-01  4.702e-02   2.723 0.006471 ** 
    ## title_sentiment_polarity       1.601e-01  5.134e-02   3.118 0.001824 ** 
    ## abs_title_subjectivity         2.216e-01  7.933e-02   2.794 0.005214 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 38467  on 27749  degrees of freedom
    ## Residual deviance: 35105  on 27724  degrees of freedom
    ## AIC: 35157
    ## 
    ## Number of Fisher Scoring iterations: 4

``` r
# GLM model 3
glmFit.3 <- glm(shares ~ . - n_tokens_title - n_unique_tokens - 
                  n_non_stop_words - n_non_stop_unique_tokens - 
                  num_imgs - num_videos - data_channel_is_lifestyle - 
                  data_channel_is_world - self_reference_min_shares - 
                  self_reference_max_shares - self_reference_avg_sharess -
                  weekday_is_monday - LDA_00 - LDA_01 - LDA_02 - LDA_03 -
                  LDA_04 - global_sentiment_polarity -
                  global_rate_positive_words - global_rate_negative_words -
                  rate_positive_words - rate_negative_words -
                  avg_positive_polarity - max_positive_polarity -
                  avg_negative_polarity - min_negative_polarity -
                  max_negative_polarity - abs_title_sentiment_polarity, 
                data = mondayTrain, family = 'binomial')
summary(glmFit.3)
```

    ## 
    ## Call:
    ## glm(formula = shares ~ . - n_tokens_title - n_unique_tokens - 
    ##     n_non_stop_words - n_non_stop_unique_tokens - num_imgs - 
    ##     num_videos - data_channel_is_lifestyle - data_channel_is_world - 
    ##     self_reference_min_shares - self_reference_max_shares - self_reference_avg_sharess - 
    ##     weekday_is_monday - LDA_00 - LDA_01 - LDA_02 - LDA_03 - LDA_04 - 
    ##     global_sentiment_polarity - global_rate_positive_words - 
    ##     global_rate_negative_words - rate_positive_words - rate_negative_words - 
    ##     avg_positive_polarity - max_positive_polarity - avg_negative_polarity - 
    ##     min_negative_polarity - max_negative_polarity - abs_title_sentiment_polarity, 
    ##     family = "binomial", data = mondayTrain)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -4.8620  -1.0396  -0.6489   1.0773   2.1197  
    ## 
    ## Coefficients:
    ##                                 Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                   -2.088e+00  1.476e-01 -14.143  < 2e-16 ***
    ## n_tokens_content               2.024e-04  3.376e-05   5.997 2.02e-09 ***
    ## num_hrefs                      8.734e-03  1.467e-03   5.955 2.60e-09 ***
    ## num_self_hrefs                -2.024e-02  3.963e-03  -5.108 3.26e-07 ***
    ## average_token_length          -9.707e-02  2.080e-02  -4.667 3.06e-06 ***
    ## num_keywords                   4.963e-02  8.145e-03   6.093 1.10e-09 ***
    ## data_channel_is_entertainment -3.399e-01  3.751e-02  -9.061  < 2e-16 ***
    ## data_channel_is_bus            4.137e-01  4.000e-02  10.343  < 2e-16 ***
    ## data_channel_is_socmed         1.139e+00  6.229e-02  18.291  < 2e-16 ***
    ## data_channel_is_tech           7.995e-01  3.784e-02  21.129  < 2e-16 ***
    ## kw_min_min                     1.707e-03  3.639e-04   4.692 2.70e-06 ***
    ## kw_max_min                     3.531e-05  1.400e-05   2.522 0.011678 *  
    ## kw_avg_min                    -2.465e-04  8.796e-05  -2.802 0.005076 ** 
    ## kw_min_max                    -7.222e-07  2.522e-07  -2.863 0.004193 ** 
    ## kw_max_max                    -4.092e-07  1.259e-07  -3.252 0.001148 ** 
    ## kw_avg_max                    -6.475e-07  1.754e-07  -3.691 0.000223 ***
    ## kw_min_avg                    -9.060e-05  1.650e-05  -5.492 3.97e-08 ***
    ## kw_max_avg                    -9.980e-05  5.849e-06 -17.063  < 2e-16 ***
    ## kw_avg_avg                     7.756e-04  3.008e-05  25.786  < 2e-16 ***
    ## is_weekend                     8.224e-01  3.977e-02  20.681  < 2e-16 ***
    ## global_subjectivity            7.978e-01  1.469e-01   5.432 5.58e-08 ***
    ## min_positive_polarity         -7.649e-01  1.961e-01  -3.900 9.62e-05 ***
    ## title_subjectivity             1.252e-01  4.691e-02   2.668 0.007627 ** 
    ## title_sentiment_polarity       1.679e-01  5.058e-02   3.320 0.000902 ***
    ## abs_title_subjectivity         2.254e-01  7.922e-02   2.846 0.004431 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 38467  on 27749  degrees of freedom
    ## Residual deviance: 35106  on 27725  degrees of freedom
    ## AIC: 35156
    ## 
    ## Number of Fisher Scoring iterations: 4

After deleting some unnecessary predictors twice, we can see that for
binary logistic regression model 3, all p-values are smaller than 0.05,
which means that all predictors are statistically significant.

### Linear Regression Model Comparsion

We are going to choose a better linear model from the multiple linear
model 3 and the binary logistic regression model 3 with comparing the
values of AIC and BIC respectively.

``` r
AIC(fit.3, glmFit.3)
```

    ##          df      AIC
    ## fit.3    28 71628.88
    ## glmFit.3 25 35155.55

``` r
BIC(fit.3, glmFit.3)
```

    ##          df      BIC
    ## fit.3    28 71859.35
    ## glmFit.3 25 35361.33

Since the binary logistic regression model 3 has much smaller AIC and
BIC values, we can choose the binary logistic regression model 3.

## Ensemble model Fit

For the ensemble model fit, we are going to use the R machine learning
`caret` package.

### Bagged Tree

``` r
# Use ‘trainControl()‘ to control the computational nuances of the train method
trctrl <- trainControl(method = 'repeatedcv', number = 5, repeats = 2)

# Fit a bagged tree
baggedTree <- train(shares ~ ., data = mondayTrain, trControl=trctrl,
preProcess = c("center", "scale"), method = "treebag")

baggedTree
```

    ## Bagged CART 
    ## 
    ## 27750 samples
    ##    52 predictor
    ##     2 classes: '0', '1' 
    ## 
    ## Pre-processing: centered (52), scaled (52) 
    ## Resampling: Cross-Validated (5 fold, repeated 2 times) 
    ## Summary of sample sizes: 22200, 22201, 22199, 22200, 22200, 22200, ... 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.6414416  0.2827514

``` r
# Convert 'shares' in mondayTest into factors as well
mondayTest$shares[mondayTest$shares <= 1400] <- 0
mondayTest$shares[mondayTest$shares > 1400] <- 1
mondayTest$shares <- as.factor(mondayTest$shares)

# Predict classes for test dataset
test_pred_baggedTree <- predict(baggedTree, newdata = mondayTest)

# Accurary of the model
confusionMatrix(test_pred_baggedTree, mondayTest$shares)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 3944 2110
    ##          1 2116 3724
    ##                                          
    ##                Accuracy : 0.6447         
    ##                  95% CI : (0.636, 0.6533)
    ##     No Information Rate : 0.5095         
    ##     P-Value [Acc > NIR] : <2e-16         
    ##                                          
    ##                   Kappa : 0.2891         
    ##                                          
    ##  Mcnemar's Test P-Value : 0.9387         
    ##                                          
    ##             Sensitivity : 0.6508         
    ##             Specificity : 0.6383         
    ##          Pos Pred Value : 0.6515         
    ##          Neg Pred Value : 0.6377         
    ##              Prevalence : 0.5095         
    ##          Detection Rate : 0.3316         
    ##    Detection Prevalence : 0.5090         
    ##       Balanced Accuracy : 0.6446         
    ##                                          
    ##        'Positive' Class : 0              
    ## 

### Boosted Tree

``` r
# Fit a boosted tree
boostTree <- train(shares ~ ., data = mondayTrain, distribution = "bernoulli",
                   method = "gbm", verbose = FALSE, trControl=trctrl)

# Predict classes for test dataset
test_pred_boostTree <- predict(boostTree, newdata = mondayTest)

# Accurary of the model
confusionMatrix(test_pred_boostTree, mondayTest$shares)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 4034 1969
    ##          1 2026 3865
    ##                                           
    ##                Accuracy : 0.6641          
    ##                  95% CI : (0.6555, 0.6726)
    ##     No Information Rate : 0.5095          
    ##     P-Value [Acc > NIR] : <2e-16          
    ##                                           
    ##                   Kappa : 0.3281          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.3756          
    ##                                           
    ##             Sensitivity : 0.6657          
    ##             Specificity : 0.6625          
    ##          Pos Pred Value : 0.6720          
    ##          Neg Pred Value : 0.6561          
    ##              Prevalence : 0.5095          
    ##          Detection Rate : 0.3392          
    ##    Detection Prevalence : 0.5047          
    ##       Balanced Accuracy : 0.6641          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

### Ensemble model Model Comparsion

We are going to choose a better ensemble model after comparing the
misclassification rate.

``` r
baggedTbl <- table(data.frame(pred = predict(baggedTree, mondayTest), true = mondayTest$shares))

boostTbl <- table(data.frame(pred = predict(boostTree, mondayTest), true = mondayTest$shares))

#misclassificatoon rate
1-c(bag = sum(diag(baggedTbl)/sum(baggedTbl)),
    boost = sum(diag(boostTbl)/sum(boostTbl)))
```

    ##       bag     boost 
    ## 0.3553052 0.3358836

Since the boosted trees model with a smaller misclassification rate
0.336 and a larger accuracy rate 0.665 than the bagged trees model, we
can choose the boosted trees model.

# Conclusions

For the linear regression model, the binary logistic model 3 is the best
one after comparing AIC and BIC values while all variables are
statistically significant.

For the chosen model, the binary logistic model 3, we can see that some
variables whose coefficients are positive need to be increased, like
*n\_tokens\_content, num\_hrefs, num\_keywords, data\_channel\_is\_bus,
data\_channel\_is\_socmed, data\_channel\_is\_tech, kw\_min\_min,
kw\_max\_min, kw\_avg\_avg, is\_weekend, globabl\_subjectivity,
title\_subjectivity, title\_sentiment\_polarity and
abs\_title\_subjectivity* while the rest variables whose coefficients
are negative needs to be decreased, like *num\_self\_hrefs,
average\_token\_length, data\_channel\_is\_entertainment, kw\_avg\_min,
kw\_min\_max, kw\_max\_max, kw\_avg\_max, kw\_min\_avg, kw\_max\_avg,
and min\_positive\_polarity*.

For the ensemble model, the boosted tree model is the best one. This
gives highest accuracy of 66.5%. • The data set on a whole gives average
accuracy of 65.5% which shows that the dataset is inconsistent
indicating that irrelevant information has been used.

Therefore, this dataset is insufficient to predict the number of shares
with high levels of accuracy for a news article considering its
popularity and thus more data needs to be collected.
