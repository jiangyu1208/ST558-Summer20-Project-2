# Load the libraries

library(tidyverse)
library(haven)
library(knitr)

# Read data and remove the first two columns since they are non-predictive.
news_pop <- read_csv('OnlineNewsPopularity.csv')[, -c(1, 2)]
news_pop


data.frame(output_file = "MondayAnalysis.md", params = list(weekday = "weekday_is_monday"))

#get unique weekdays
weekdays <- c("weekday_is_monday", "weekday_is_tuesday", "weekday_is_wednesday",
              "weekday_is_thursday", "weekday_is_friday", "weekday_is_saturday",
              "weekday_is_sunday")

#create filenames
output_file <- paste0(weekdays, ".md")

#create a list for each day with just the weekday parameter
params = lapply(weekdays, FUN = function(x){list(weekday = x)})

#put into a data frame 
reports <- tibble(output_file, params)
reports


library(rmarkdown)
apply(reports, MARGIN = 1,
      FUN = function(x){
        render(input = "README.Rmd", output_file = x[[1]], params = x[[2]])
      })
## 

