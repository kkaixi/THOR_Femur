setwd('C:/Users/tangk/pmg-projects/THOR_Femur')
source("C:/Users/tangk/AppData/Local/Continuum/anaconda3/PMG/COM/read_data.R")
source("C:/Users/tangk/AppData/Local/Continuum/anaconda3/PMG/COM/helper.R")
library(jsonlite)

directory <- 'P:/Data Analysis/Projects/THOR Femur'

features <- read.csv(file.path(directory,'features.csv'))
rownames(features) <- features$TC

model_1d <- lm(left_femur_load ~ min_distance_from_b, data=features)

summary(model_1d)

model_2d <- lm(left_femur_load ~ min_distance_from_b + left_foot_z, data=features)
summary(model_2d)

# plot the predicted vs. actual
plot(1:dim(features)[1], features$left_femur_load, col='red')
points(as.numeric(names(model_2d$fitted.values)),model_2d$fitted.values, col='green')

model_res <- lm(left_femur_load_plus_x1 ~ left_foot_z, data=features)
summary(model_res)
