setwd('C:/Users/tangk/pmg-projects/THOR_Femur')
source("C:/Users/tangk/AppData/Local/Continuum/anaconda3/PMG/COM/read_data.R")
source("C:/Users/tangk/AppData/Local/Continuum/anaconda3/PMG/COM/helper.R")
library(jsonlite)

directory <- 'P:/Data Analysis/Projects/THOR Femur'
table <- read_table(directory, query=c('DUMMY==\'THOR\'', 'KAB==\'NO\'', 'SPEED==48'))
rownames(table) <- table$TC

features <- read.csv(file.path(directory,'features.csv'))
rownames(features) <- features$X
features.trunc <- features[rownames(table),]

#model_1d <- lm(Min_11FEMRLE00THFOZB ~ min_distance_from_b, data=features.trunc)
#summary(model_1d)

model_2d <- lm(Min_11FEMRLE00THFOZB ~ Max_11ACTBLE00THFOYB + Max_11TIBILEMITHACYA, data=features.trunc)
summary(model_2d)

# plot the predicted vs. actual
plot(1:dim(features)[1], features$left_femur_load, col='red')
points(as.numeric(names(model_2d$fitted.values)),model_2d$fitted.values, col='green')

model_res <- lm(left_femur_load_plus_x1 ~ left_foot_z, data=features)
summary(model_res)
