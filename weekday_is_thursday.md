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
like AIC and BIC, and then we can make a conclusion about *shares*.

## Methods

First, we can slice data into two sets, a training set (70% of the data)
and the test set (30% of the data) and then we are going to start the
model fit.

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
regression model, I am going to fit the model with some selected
predictors.

Similarly, for the ensemble model, I am also going to start to fit two
models (bagged tree and boosted tree) with selected predictors from the
train set. After comparing the misclassification rate, the best model
can be chosen.

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

# Read data and remove the first two columns since they are non-predictive.

news_pop <- read_csv('OnlineNewsPopularity.csv')[, -c(1, 2)]

params$weekday
```

    ## [1] "weekday_is_thursday"

``` r
# Just consider 'Monday' at first 
dat<- news_pop %>% select(!starts_with('weekday_is_'), params$weekday)

# Check if the data has any missing values
sum(is.na(dat))
```

    ## [1] 0

Since there is no missing data for this dataset, we can slice the data
into the train and the test sets, respectively.

``` r
# Split the 'Monday' data, 70% of the data for training and the rest for testing
train <- sample(1:nrow(dat), size = nrow(dat) * 0.7)
test <- dplyr::setdiff(1:nrow(dat), train)

TrainDat <- dat[train, ]
TestDat <- dat[test, ]
```

## Data Summarizations

After splicing the data into two sets, we can explore the response
variable, *shares* and predictors.

### Response Variable

``` r
# Histogram of 'shares'
ggplot(data = TrainDat, aes(x = shares)) +
  geom_histogram()
```

![](weekday_is_thursday_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

Since the histogram of share is highly skewed, we can consider to use
the log transformation to obtain a new histogram for shares.

``` r
x <- log(TrainDat$shares)

ggplot(data = TrainDat, aes(x)) +
  geom_histogram() + 
  xlab('Log(shares)')
```

![](weekday_is_thursday_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

After comparing these two histograms, we decide to use the log(shares)
as our response for the multiple linear regression model.

### Predictor Variables

``` r
summary(TrainDat)
```

    ##  n_tokens_title n_tokens_content n_unique_tokens n_non_stop_words
    ##  Min.   : 3.0   Min.   :   0     Min.   :  0     Min.   :   0    
    ##  1st Qu.: 9.0   1st Qu.: 245     1st Qu.:  0     1st Qu.:   1    
    ##  Median :10.0   Median : 408     Median :  1     Median :   1    
    ##  Mean   :10.4   Mean   : 544     Mean   :  1     Mean   :   1    
    ##  3rd Qu.:12.0   3rd Qu.: 716     3rd Qu.:  1     3rd Qu.:   1    
    ##  Max.   :20.0   Max.   :8474     Max.   :701     Max.   :1042    
    ##  n_non_stop_unique_tokens   num_hrefs   num_self_hrefs    num_imgs  
    ##  Min.   :  0              Min.   :  0   Min.   :  0    Min.   :  0  
    ##  1st Qu.:  1              1st Qu.:  4   1st Qu.:  1    1st Qu.:  1  
    ##  Median :  1              Median :  7   Median :  3    Median :  1  
    ##  Mean   :  1              Mean   : 11   Mean   :  3    Mean   :  5  
    ##  3rd Qu.:  1              3rd Qu.: 13   3rd Qu.:  4    3rd Qu.:  4  
    ##  Max.   :650              Max.   :304   Max.   :116    Max.   :128  
    ##    num_videos average_token_length  num_keywords  data_channel_is_lifestyle
    ##  Min.   : 0   Min.   :0.0          Min.   : 1.0   Min.   :0.00             
    ##  1st Qu.: 0   1st Qu.:4.5          1st Qu.: 6.0   1st Qu.:0.00             
    ##  Median : 0   Median :4.7          Median : 7.0   Median :0.00             
    ##  Mean   : 1   Mean   :4.6          Mean   : 7.2   Mean   :0.05             
    ##  3rd Qu.: 1   3rd Qu.:4.9          3rd Qu.: 9.0   3rd Qu.:0.00             
    ##  Max.   :91   Max.   :8.0          Max.   :10.0   Max.   :1.00             
    ##  data_channel_is_entertainment data_channel_is_bus data_channel_is_socmed
    ##  Min.   :0.00                  Min.   :0.00        Min.   :0.00          
    ##  1st Qu.:0.00                  1st Qu.:0.00        1st Qu.:0.00          
    ##  Median :0.00                  Median :0.00        Median :0.00          
    ##  Mean   :0.18                  Mean   :0.16        Mean   :0.06          
    ##  3rd Qu.:0.00                  3rd Qu.:0.00        3rd Qu.:0.00          
    ##  Max.   :1.00                  Max.   :1.00        Max.   :1.00          
    ##  data_channel_is_tech data_channel_is_world   kw_min_min    kw_max_min    
    ##  Min.   :0.00         Min.   :0.00          Min.   : -1   Min.   :     0  
    ##  1st Qu.:0.00         1st Qu.:0.00          1st Qu.: -1   1st Qu.:   445  
    ##  Median :0.00         Median :0.00          Median : -1   Median :   657  
    ##  Mean   :0.19         Mean   :0.21          Mean   : 26   Mean   :  1158  
    ##  3rd Qu.:0.00         3rd Qu.:0.00          3rd Qu.:  4   3rd Qu.:  1000  
    ##  Max.   :1.00         Max.   :1.00          Max.   :377   Max.   :298400  
    ##    kw_avg_min      kw_min_max       kw_max_max       kw_avg_max    
    ##  Min.   :   -1   Min.   :     0   Min.   :     0   Min.   :     0  
    ##  1st Qu.:  141   1st Qu.:     0   1st Qu.:843300   1st Qu.:172494  
    ##  Median :  234   Median :  1400   Median :843300   Median :243933  
    ##  Mean   :  313   Mean   : 13686   Mean   :751158   Mean   :258434  
    ##  3rd Qu.:  356   3rd Qu.:  7900   3rd Qu.:843300   3rd Qu.:329894  
    ##  Max.   :42828   Max.   :843300   Max.   :843300   Max.   :843300  
    ##    kw_min_avg     kw_max_avg       kw_avg_avg    self_reference_min_shares
    ##  Min.   :  -1   Min.   :     0   Min.   :    0   Min.   :     0           
    ##  1st Qu.:   0   1st Qu.:  3560   1st Qu.: 2380   1st Qu.:   642           
    ##  Median :1019   Median :  4355   Median : 2869   Median :  1200           
    ##  Mean   :1117   Mean   :  5653   Mean   : 3135   Mean   :  4057           
    ##  3rd Qu.:2061   3rd Qu.:  6015   3rd Qu.: 3600   3rd Qu.:  2700           
    ##  Max.   :3613   Max.   :298400   Max.   :43568   Max.   :843300           
    ##  self_reference_max_shares self_reference_avg_sharess   is_weekend  
    ##  Min.   :     0            Min.   :     0             Min.   :0.00  
    ##  1st Qu.:  1100            1st Qu.:   985             1st Qu.:0.00  
    ##  Median :  2900            Median :  2200             Median :0.00  
    ##  Mean   : 10326            Mean   :  6451             Mean   :0.13  
    ##  3rd Qu.:  8100            3rd Qu.:  5200             3rd Qu.:0.00  
    ##  Max.   :843300            Max.   :843300             Max.   :1.00  
    ##      LDA_00         LDA_01         LDA_02         LDA_03         LDA_04    
    ##  Min.   :0.00   Min.   :0.00   Min.   :0.00   Min.   :0.00   Min.   :0.00  
    ##  1st Qu.:0.03   1st Qu.:0.03   1st Qu.:0.03   1st Qu.:0.03   1st Qu.:0.03  
    ##  Median :0.03   Median :0.03   Median :0.04   Median :0.04   Median :0.04  
    ##  Mean   :0.18   Mean   :0.14   Mean   :0.22   Mean   :0.22   Mean   :0.23  
    ##  3rd Qu.:0.24   3rd Qu.:0.15   3rd Qu.:0.33   3rd Qu.:0.37   3rd Qu.:0.41  
    ##  Max.   :0.92   Max.   :0.93   Max.   :0.92   Max.   :0.93   Max.   :0.93  
    ##  global_subjectivity global_sentiment_polarity global_rate_positive_words
    ##  Min.   :0.00        Min.   :-0.38             Min.   :0.000             
    ##  1st Qu.:0.40        1st Qu.: 0.06             1st Qu.:0.028             
    ##  Median :0.45        Median : 0.12             Median :0.039             
    ##  Mean   :0.44        Mean   : 0.12             Mean   :0.040             
    ##  3rd Qu.:0.51        3rd Qu.: 0.18             3rd Qu.:0.050             
    ##  Max.   :1.00        Max.   : 0.73             Max.   :0.155             
    ##  global_rate_negative_words rate_positive_words rate_negative_words
    ##  Min.   :0.000              Min.   :0.00        Min.   :0.00       
    ##  1st Qu.:0.010              1st Qu.:0.60        1st Qu.:0.19       
    ##  Median :0.015              Median :0.71        Median :0.28       
    ##  Mean   :0.017              Mean   :0.68        Mean   :0.29       
    ##  3rd Qu.:0.022              3rd Qu.:0.80        3rd Qu.:0.38       
    ##  Max.   :0.162              Max.   :1.00        Max.   :1.00       
    ##  avg_positive_polarity min_positive_polarity max_positive_polarity
    ##  Min.   :0.00          Min.   :0.00          Min.   :0.00         
    ##  1st Qu.:0.31          1st Qu.:0.05          1st Qu.:0.60         
    ##  Median :0.36          Median :0.10          Median :0.80         
    ##  Mean   :0.35          Mean   :0.10          Mean   :0.76         
    ##  3rd Qu.:0.41          3rd Qu.:0.10          3rd Qu.:1.00         
    ##  Max.   :1.00          Max.   :1.00          Max.   :1.00         
    ##  avg_negative_polarity min_negative_polarity max_negative_polarity
    ##  Min.   :-1.00         Min.   :-1.00         Min.   :-1.00        
    ##  1st Qu.:-0.33         1st Qu.:-0.70         1st Qu.:-0.12        
    ##  Median :-0.25         Median :-0.50         Median :-0.10        
    ##  Mean   :-0.26         Mean   :-0.52         Mean   :-0.11        
    ##  3rd Qu.:-0.19         3rd Qu.:-0.30         3rd Qu.:-0.05        
    ##  Max.   : 0.00         Max.   : 0.00         Max.   : 0.00        
    ##  title_subjectivity title_sentiment_polarity abs_title_subjectivity
    ##  Min.   :0.00       Min.   :-1.00            Min.   :0.00          
    ##  1st Qu.:0.00       1st Qu.: 0.00            1st Qu.:0.17          
    ##  Median :0.15       Median : 0.00            Median :0.50          
    ##  Mean   :0.28       Mean   : 0.07            Mean   :0.34          
    ##  3rd Qu.:0.50       3rd Qu.: 0.15            3rd Qu.:0.50          
    ##  Max.   :1.00       Max.   : 1.00            Max.   :0.50          
    ##  abs_title_sentiment_polarity     shares       weekday_is_thursday
    ##  Min.   :0.00                 Min.   :     1   Min.   :0.00       
    ##  1st Qu.:0.00                 1st Qu.:   949   1st Qu.:0.00       
    ##  Median :0.00                 Median :  1400   Median :0.00       
    ##  Mean   :0.16                 Mean   :  3444   Mean   :0.18       
    ##  3rd Qu.:0.25                 3rd Qu.:  2800   3rd Qu.:0.00       
    ##  Max.   :1.00                 Max.   :690400   Max.   :1.00

From the output above, we can remove the variable *is\_weekend* since it
seems to be duplicating days of week; five *LDA* variables due to
meaningless values and *kw\_min\_min, kw\_avg\_min, kw\_min\_avg*
because of negative values. We can obtain a new train and a test set
below.

``` r
# A new train and a test set
TrainDat <- TrainDat %>% select(!starts_with('LDA_'), -c(is_weekend, kw_min_min, kw_avg_min, kw_min_avg))

TestDat <- TestDat %>% select(!starts_with('LDA_'), -c(is_weekend, kw_min_min, kw_avg_min, kw_min_avg))

head(TrainDat)
```

    ## # A tibble: 6 x 44
    ##   n_tokens_title n_tokens_content n_unique_tokens n_non_stop_words
    ##            <dbl>            <dbl>           <dbl>            <dbl>
    ## 1              7              802           0.502             1.00
    ## 2             14             1193           0.411             1.00
    ## 3             11              147           0.760             1.00
    ## 4              9              206           0.629             1.00
    ## 5              8               60           0.814             1.00
    ## 6              8               79           0.861             1.00
    ## # ... with 40 more variables: n_non_stop_unique_tokens <dbl>, num_hrefs <dbl>,
    ## #   num_self_hrefs <dbl>, num_imgs <dbl>, num_videos <dbl>,
    ## #   average_token_length <dbl>, num_keywords <dbl>,
    ## #   data_channel_is_lifestyle <dbl>, data_channel_is_entertainment <dbl>,
    ## #   data_channel_is_bus <dbl>, data_channel_is_socmed <dbl>,
    ## #   data_channel_is_tech <dbl>, data_channel_is_world <dbl>, kw_max_min <dbl>,
    ## #   kw_min_max <dbl>, kw_max_max <dbl>, kw_avg_max <dbl>, kw_max_avg <dbl>,
    ## #   kw_avg_avg <dbl>, self_reference_min_shares <dbl>,
    ## #   self_reference_max_shares <dbl>, self_reference_avg_sharess <dbl>,
    ## #   global_subjectivity <dbl>, global_sentiment_polarity <dbl>,
    ## #   global_rate_positive_words <dbl>, global_rate_negative_words <dbl>,
    ## #   rate_positive_words <dbl>, rate_negative_words <dbl>,
    ## #   avg_positive_polarity <dbl>, min_positive_polarity <dbl>,
    ## #   max_positive_polarity <dbl>, avg_negative_polarity <dbl>,
    ## #   min_negative_polarity <dbl>, max_negative_polarity <dbl>,
    ## #   title_subjectivity <dbl>, title_sentiment_polarity <dbl>,
    ## #   abs_title_subjectivity <dbl>, abs_title_sentiment_polarity <dbl>,
    ## #   shares <dbl>, weekday_is_thursday <dbl>

``` r
head(TestDat)
```

    ## # A tibble: 6 x 44
    ##   n_tokens_title n_tokens_content n_unique_tokens n_non_stop_words
    ##            <dbl>            <dbl>           <dbl>            <dbl>
    ## 1              9              531           0.504             1.00
    ## 2              8              960           0.418             1.00
    ## 3             12              989           0.434             1.00
    ## 4             11               97           0.670             1.00
    ## 5              8             1118           0.512             1.00
    ## 6              8              397           0.625             1.00
    ## # ... with 40 more variables: n_non_stop_unique_tokens <dbl>, num_hrefs <dbl>,
    ## #   num_self_hrefs <dbl>, num_imgs <dbl>, num_videos <dbl>,
    ## #   average_token_length <dbl>, num_keywords <dbl>,
    ## #   data_channel_is_lifestyle <dbl>, data_channel_is_entertainment <dbl>,
    ## #   data_channel_is_bus <dbl>, data_channel_is_socmed <dbl>,
    ## #   data_channel_is_tech <dbl>, data_channel_is_world <dbl>, kw_max_min <dbl>,
    ## #   kw_min_max <dbl>, kw_max_max <dbl>, kw_avg_max <dbl>, kw_max_avg <dbl>,
    ## #   kw_avg_avg <dbl>, self_reference_min_shares <dbl>,
    ## #   self_reference_max_shares <dbl>, self_reference_avg_sharess <dbl>,
    ## #   global_subjectivity <dbl>, global_sentiment_polarity <dbl>,
    ## #   global_rate_positive_words <dbl>, global_rate_negative_words <dbl>,
    ## #   rate_positive_words <dbl>, rate_negative_words <dbl>,
    ## #   avg_positive_polarity <dbl>, min_positive_polarity <dbl>,
    ## #   max_positive_polarity <dbl>, avg_negative_polarity <dbl>,
    ## #   min_negative_polarity <dbl>, max_negative_polarity <dbl>,
    ## #   title_subjectivity <dbl>, title_sentiment_polarity <dbl>,
    ## #   abs_title_subjectivity <dbl>, abs_title_sentiment_polarity <dbl>,
    ## #   shares <dbl>, weekday_is_thursday <dbl>

# Modeling

## Linear Regression Fit

### Multiple Linear Regression

We are going to fit the multiple linear regression with selected
predictors.

``` r
# Multiple linear model
fit <- lm(log(shares) ~ . , data = TrainDat)
summary(fit)
```

    ## 
    ## Call:
    ## lm(formula = log(shares) ~ ., data = TrainDat)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -8.117 -0.558 -0.165  0.401  5.642 
    ## 
    ## Coefficients:
    ##                                Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                    6.76e+00   6.14e-02  110.04  < 2e-16 ***
    ## n_tokens_title                 5.61e-03   2.62e-03    2.14  0.03251 *  
    ## n_tokens_content               6.49e-05   2.07e-05    3.14  0.00170 ** 
    ## n_unique_tokens                2.23e-01   1.75e-01    1.27  0.20274    
    ## n_non_stop_words              -1.56e-02   5.92e-02   -0.26  0.79190    
    ## n_non_stop_unique_tokens      -2.13e-01   1.49e-01   -1.43  0.15203    
    ## num_hrefs                      4.33e-03   6.06e-04    7.14  9.4e-13 ***
    ## num_self_hrefs                -9.01e-03   1.65e-03   -5.46  4.9e-08 ***
    ## num_imgs                       1.95e-03   8.14e-04    2.40  0.01645 *  
    ## num_videos                     1.11e-03   1.42e-03    0.78  0.43471    
    ## average_token_length          -9.40e-02   2.17e-02   -4.33  1.5e-05 ***
    ## num_keywords                   2.39e-02   3.29e-03    7.25  4.2e-13 ***
    ## data_channel_is_lifestyle     -4.47e-02   2.96e-02   -1.51  0.13115    
    ## data_channel_is_entertainment -2.26e-01   2.20e-02  -10.27  < 2e-16 ***
    ## data_channel_is_bus            1.25e-03   2.37e-02    0.05  0.95784    
    ## data_channel_is_socmed         2.48e-01   2.92e-02    8.50  < 2e-16 ***
    ## data_channel_is_tech           1.39e-01   2.35e-02    5.91  3.5e-09 ***
    ## data_channel_is_world         -1.46e-01   2.49e-02   -5.85  5.0e-09 ***
    ## kw_max_min                    -3.37e-06   1.70e-06   -1.98  0.04774 *  
    ## kw_min_max                    -3.92e-07   1.05e-07   -3.73  0.00019 ***
    ## kw_max_max                    -1.70e-07   3.47e-08   -4.90  9.8e-07 ***
    ## kw_avg_max                    -3.13e-07   7.33e-08   -4.26  2.0e-05 ***
    ## kw_max_avg                    -3.43e-05   2.00e-06  -17.10  < 2e-16 ***
    ## kw_avg_avg                     2.87e-04   9.92e-06   28.90  < 2e-16 ***
    ## self_reference_min_shares      1.11e-06   6.80e-07    1.64  0.10141    
    ## self_reference_max_shares      2.88e-07   3.72e-07    0.77  0.43874    
    ## self_reference_avg_sharess     6.22e-07   9.46e-07    0.66  0.51099    
    ## global_subjectivity            4.07e-01   7.75e-02    5.24  1.6e-07 ***
    ## global_sentiment_polarity     -2.39e-01   1.54e-01   -1.56  0.11917    
    ## global_rate_positive_words    -4.45e-01   6.61e-01   -0.67  0.50051    
    ## global_rate_negative_words    -6.78e-01   1.28e+00   -0.53  0.59507    
    ## rate_positive_words            2.65e-01   1.29e-01    2.05  0.04013 *  
    ## rate_negative_words            2.06e-01   1.47e-01    1.41  0.15942    
    ## avg_positive_polarity          9.67e-02   1.26e-01    0.77  0.44143    
    ## min_positive_polarity         -4.52e-01   1.04e-01   -4.35  1.3e-05 ***
    ## max_positive_polarity         -1.84e-02   3.95e-02   -0.47  0.64025    
    ## avg_negative_polarity         -2.10e-01   1.15e-01   -1.82  0.06815 .  
    ## min_negative_polarity          3.62e-02   4.20e-02    0.86  0.38820    
    ## max_negative_polarity         -3.56e-02   9.59e-02   -0.37  0.71037    
    ## title_subjectivity             5.85e-02   2.51e-02    2.33  0.01996 *  
    ## title_sentiment_polarity       9.13e-02   2.29e-02    3.99  6.8e-05 ***
    ## abs_title_subjectivity         1.18e-01   3.34e-02    3.52  0.00043 ***
    ## abs_title_sentiment_polarity   5.82e-03   3.62e-02    0.16  0.87214    
    ## weekday_is_thursday           -6.05e-02   1.37e-02   -4.41  1.1e-05 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.88 on 27706 degrees of freedom
    ## Multiple R-squared:  0.112,  Adjusted R-squared:  0.111 
    ## F-statistic: 81.6 on 43 and 27706 DF,  p-value: <2e-16

### Binary Logistic Regression

Use *shares* to create the binary variable : diving the shares into two
groups (\<1400 and \>= 1400) and fit the model with the selected
predictors as well.

``` r
# Convert 'shares' into factor: <= 1400: unpopular, > 1400: popular
TrainDat$shares[TrainDat$shares <= 1400] <- 0
TrainDat$shares[TrainDat$shares > 1400] <- 1

TrainDat$shares <- as.factor(TrainDat$shares)
  
# GLM model
glmFit <- glm(shares~. , data = TrainDat, family = 'binomial')
summary(glmFit)
```

    ## 
    ## Call:
    ## glm(formula = shares ~ ., family = "binomial", data = TrainDat)
    ## 
    ## Deviance Residuals: 
    ##    Min      1Q  Median      3Q     Max  
    ## -4.386  -1.047  -0.672   1.078   1.944  
    ## 
    ## Coefficients:
    ##                                Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                   -1.40e+00   1.48e-01   -9.49  < 2e-16 ***
    ## n_tokens_title                -3.41e-03   6.27e-03   -0.54  0.58605    
    ## n_tokens_content               1.83e-04   5.20e-05    3.51  0.00044 ***
    ## n_unique_tokens               -8.43e-02   4.23e-01   -0.20  0.84200    
    ## n_non_stop_words               4.06e-01   1.45e-01    2.81  0.00498 ** 
    ## n_non_stop_unique_tokens      -5.49e-01   3.57e-01   -1.54  0.12451    
    ## num_hrefs                      9.03e-03   1.56e-03    5.80  6.5e-09 ***
    ## num_self_hrefs                -2.16e-02   3.98e-03   -5.43  5.5e-08 ***
    ## num_imgs                      -1.21e-04   1.98e-03   -0.06  0.95102    
    ## num_videos                    -2.17e-03   3.43e-03   -0.63  0.52651    
    ## average_token_length          -1.05e-01   5.18e-02   -2.02  0.04347 *  
    ## num_keywords                   6.29e-02   7.90e-03    7.95  1.8e-15 ***
    ## data_channel_is_lifestyle      1.27e-01   7.06e-02    1.80  0.07116 .  
    ## data_channel_is_entertainment -4.15e-01   5.24e-02   -7.92  2.4e-15 ***
    ## data_channel_is_bus            2.75e-01   5.66e-02    4.87  1.1e-06 ***
    ## data_channel_is_socmed         1.01e+00   7.33e-02   13.75  < 2e-16 ***
    ## data_channel_is_tech           6.44e-01   5.63e-02   11.44  < 2e-16 ***
    ## data_channel_is_world         -1.72e-01   5.96e-02   -2.89  0.00384 ** 
    ## kw_max_min                    -1.46e-06   5.09e-06   -0.29  0.77427    
    ## kw_min_max                    -8.16e-07   2.48e-07   -3.29  0.00100 ** 
    ## kw_max_max                    -7.44e-07   8.29e-08   -8.97  < 2e-16 ***
    ## kw_avg_max                    -6.24e-07   1.76e-07   -3.54  0.00040 ***
    ## kw_max_avg                    -8.24e-05   5.32e-06  -15.48  < 2e-16 ***
    ## kw_avg_avg                     6.41e-04   2.58e-05   24.89  < 2e-16 ***
    ## self_reference_min_shares      3.21e-06   2.02e-06    1.59  0.11153    
    ## self_reference_max_shares      8.96e-07   9.75e-07    0.92  0.35847    
    ## self_reference_avg_sharess     6.75e-07   2.50e-06    0.27  0.78692    
    ## global_subjectivity            8.29e-01   1.86e-01    4.46  8.3e-06 ***
    ## global_sentiment_polarity     -3.67e-01   3.68e-01   -1.00  0.31838    
    ## global_rate_positive_words    -9.45e-01   1.58e+00   -0.60  0.54964    
    ## global_rate_negative_words     2.21e+00   3.08e+00    0.72  0.47331    
    ## rate_positive_words            1.81e-01   3.09e-01    0.58  0.55932    
    ## rate_negative_words           -1.59e-01   3.52e-01   -0.45  0.65167    
    ## avg_positive_polarity         -7.51e-02   3.01e-01   -0.25  0.80305    
    ## min_positive_polarity         -6.45e-01   2.49e-01   -2.59  0.00958 ** 
    ## max_positive_polarity         -1.92e-02   9.43e-02   -0.20  0.83853    
    ## avg_negative_polarity         -9.15e-02   2.76e-01   -0.33  0.74042    
    ## min_negative_polarity          8.02e-02   1.01e-01    0.79  0.42720    
    ## max_negative_polarity         -2.41e-01   2.30e-01   -1.05  0.29430    
    ## title_subjectivity             1.53e-01   6.04e-02    2.54  0.01124 *  
    ## title_sentiment_polarity       2.01e-01   5.52e-02    3.63  0.00028 ***
    ## abs_title_subjectivity         1.87e-01   8.02e-02    2.33  0.01979 *  
    ## abs_title_sentiment_polarity  -7.75e-02   8.70e-02   -0.89  0.37338    
    ## weekday_is_thursday           -1.45e-01   3.29e-02   -4.41  1.0e-05 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 38467  on 27749  degrees of freedom
    ## Residual deviance: 35494  on 27706  degrees of freedom
    ## AIC: 35582
    ## 
    ## Number of Fisher Scoring iterations: 6

### Linear Regression Model Comparsion

We are going to choose a better linear model from the multiple linear
model and the binary logistic regression model with comparing the values
of AIC and BIC respectively.

``` r
AIC(fit, glmFit)
```

    ##        df   AIC
    ## fit    45 71907
    ## glmFit 44 35582

``` r
BIC(fit, glmFit)
```

    ##        df   BIC
    ## fit    45 72277
    ## glmFit 44 35944

Since the binary logistic regression model has much smaller AIC and BIC
values, we can choose the binary logistic regression model.

## Ensemble model Fit

For the ensemble model fit, we are going to use the R machine learning
`caret` package.

### Bagged Tree

``` r
# Use ‘trainControl()‘ to control the computational nuances of the train method
trctrl <- trainControl(method = 'repeatedcv', number = 5, repeats = 2)

# Fit a bagged tree
baggedTree <- train(shares ~ ., data = TrainDat, trControl=trctrl,
preProcess = c("center", "scale"), method = "treebag")

baggedTree
```

    ## Bagged CART 
    ## 
    ## 27750 samples
    ##    43 predictor
    ##     2 classes: '0', '1' 
    ## 
    ## Pre-processing: centered (43), scaled (43) 
    ## Resampling: Cross-Validated (5 fold, repeated 2 times) 
    ## Summary of sample sizes: 22200, 22201, 22199, 22200, 22200, 22200, ... 
    ## Resampling results:
    ## 
    ##   Accuracy  Kappa
    ##   0.63      0.27

``` r
# Convert 'shares' in mondayTest into factors as well
TestDat$shares[TestDat$shares <= 1400] <- 0
TestDat$shares[TestDat$shares > 1400] <- 1
TestDat$shares <- as.factor(TestDat$shares)

# Predict classes for test dataset
test_pred_baggedTree <- predict(baggedTree, newdata = TestDat)

# Accurary of the model
confusionMatrix(test_pred_baggedTree, TestDat$shares)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 3833 2155
    ##          1 2227 3679
    ##                                        
    ##                Accuracy : 0.632        
    ##                  95% CI : (0.623, 0.64)
    ##     No Information Rate : 0.51         
    ##     P-Value [Acc > NIR] : <2e-16       
    ##                                        
    ##                   Kappa : 0.263        
    ##                                        
    ##  Mcnemar's Test P-Value : 0.283        
    ##                                        
    ##             Sensitivity : 0.633        
    ##             Specificity : 0.631        
    ##          Pos Pred Value : 0.640        
    ##          Neg Pred Value : 0.623        
    ##              Prevalence : 0.510        
    ##          Detection Rate : 0.322        
    ##    Detection Prevalence : 0.503        
    ##       Balanced Accuracy : 0.632        
    ##                                        
    ##        'Positive' Class : 0            
    ## 

### Boosted Tree

``` r
# Fit a boosted tree
boostTree <- train(shares ~ ., data = TrainDat, distribution = "bernoulli",
                   method = "gbm", verbose = FALSE, trControl=trctrl)

# Predict classes for test dataset
test_pred_boostTree <- predict(boostTree, newdata = TestDat)

# Accurary of the model
confusionMatrix(test_pred_boostTree, TestDat$shares)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 3952 1988
    ##          1 2108 3846
    ##                                         
    ##                Accuracy : 0.656         
    ##                  95% CI : (0.647, 0.664)
    ##     No Information Rate : 0.51          
    ##     P-Value [Acc > NIR] : <2e-16        
    ##                                         
    ##                   Kappa : 0.311         
    ##                                         
    ##  Mcnemar's Test P-Value : 0.063         
    ##                                         
    ##             Sensitivity : 0.652         
    ##             Specificity : 0.659         
    ##          Pos Pred Value : 0.665         
    ##          Neg Pred Value : 0.646         
    ##              Prevalence : 0.510         
    ##          Detection Rate : 0.332         
    ##    Detection Prevalence : 0.499         
    ##       Balanced Accuracy : 0.656         
    ##                                         
    ##        'Positive' Class : 0             
    ## 

### Ensemble model Model Comparsion

We are going to choose a better ensemble model after comparing the
misclassification rate.

``` r
baggedTbl <- table(data.frame(pred = predict(baggedTree, TestDat), true = TestDat$shares))

boostTbl <- table(data.frame(pred = predict(boostTree, TestDat), true = TestDat$shares))

#misclassificatoon rate
1-c(bag = sum(diag(baggedTbl)/sum(baggedTbl)),
    boost = sum(diag(boostTbl)/sum(boostTbl)))
```

    ##   bag boost 
    ##  0.37  0.34

Since the boosted trees model with a smaller misclassification rate
0.336 and a larger accuracy rate 0.665 than the bagged trees model, we
can choose the boosted trees model.

# Conclusions

For the linear regression model, the binary logistic model is better
than the multiple linear regression model after comparing AIC and BIC
values.

For the chosen model, the binary logistic model, we can see that some
variables whose coefficients are positive need to be increased while the
rest variables whose coefficients are negative needs to be decreased.

For the ensemble model, the boosted tree model is the best one. This
gives highest accuracy around 65%. The data set on a whole gives average
accuracy around 64% which shows that the dataset is inconsistent
indicating that irrelevant information has been used.

Therefore, this dataset is insufficient to predict the number of shares
with high levels of accuracy for a news article considering its
popularity and thus more data needs to be collected.
