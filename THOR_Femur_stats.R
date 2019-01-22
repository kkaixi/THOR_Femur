setwd('C:/Users/tangk/Python/')
source("read_data.R")
source("helper.R")
library(jsonlite)

directory <- 'P:/Data Analysis/Projects/THOR Femur'

features <- read.csv(file.path(directory,'features.csv'))

model_1d <- lm(left_femur_load ~ min_distance_from_b, data=features)
summary(model_1d)

model_2d <- lm(left_femur_load ~ min_distance_from_b + veh_cg_veh_right, data=features)
summary(model_2d)

# plot the predicted vs. actual
plot(1:dim(features)[1], features$left_femur_load, col='red')
points(as.numeric(names(model_2d$fitted.values)),model_2d$fitted.values, col='green')
