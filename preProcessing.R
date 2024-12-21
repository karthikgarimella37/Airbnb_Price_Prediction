# Libraries

library(readr)
library(dplyr)
library(knitr)
library(kableExtra)
library(tidyr)
library(caret)
library(corrplot)
library(e1071)
library(AppliedPredictiveModeling)
library(tidyverse)
library(Hmisc)
library(mlbench)
library(e1071)
library(caret)
library(lubridate)
library(ggplot2)

# Folder Path
folder_path <- "C:\\Users\\karth\\OneDrive\\Desktop\\Class Notes\\MA 5790\\Project\\Project Data"

gz_files <- list.files(folder_path, pattern = "\\.gz$", full.names = TRUE)

city_from_folder <- basename(folder_path) # Extracts the last part of the folder path (which should be the city name)

data_list <- list()

for (file in gz_files) {
  file_name <- basename(file)
  city_from_file <- gsub("\\.gz$", "", file_name)
  city_state_split <- strsplit(city_from_file, ", ")[[1]]
  
  print(city_state_split[2])
  
  city_from_file <- city_state_split[1]
  state_from_file <- ifelse(length(city_from_file) > 1, city_from_file[2], NA)
  
  data <- readr::read_csv(file)
  
  data <- data %>%
    separate(host_location, into = c("city", "state"), sep = ", ", remove = FALSE)
  
  data$city <- city_from_file[1]
  data$state <- city_state_split[2]
  
  # sampled_data <- data %>% slice_sample(n = 100, replace = TRUE)
  
  data_list[[file]] <- data
  
}

# add city, state
# add columns for date, month and year
# 
cleaned_data_list <- list()

for (file in names(data_list)) {
  data_list[[file]] <- data_list[[file]] %>%
    dplyr::mutate(neighbourhood_cleansed = as.character(neighbourhood_cleansed))

  cleaned_data_list[[file]] <- data_list[[file]]
}


final_df <- dplyr::bind_rows(cleaned_data_list)


# Dataframe format
airbnb_df <- data.frame(final_df)
airbnb_df <- airbnb_df %>% filter(state == "Washington")

# Displaying the data beautifully
kable(head(airbnb_df), format = "html", table.attr = "class='table table-bordered'") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"))


# Dimension of the complete dataset
dim(airbnb_df)


unique(airbnb_df$city)

# 
# image(is.na(airbnb_df), main = "Missing Values",
#       xlab = "Observation", ylab = "Variable",
#       xaxt = "n", yaxt = "n", bty = "n",
#       col = c("yellow", "maroon"))
# axis(1, seq(0, 1, length.out = nrow(airbnb_df)), 1:nrow(airbnb_df), col = "white")
# 
# 





# Response and Predictor variables
response <- airbnb_df$price
predictors0 <- subset(airbnb_df, select = -price)
predictors <- subset(airbnb_df, select = -price)

# hist(predictors[1:9,])


response <- gsub("[$,]", "", response)

predictors <- predictors[which(!is.na(response)), ]

airbnb_df <- airbnb_df[which(!is.na(response)), ]
airbnb_df$price <- as.numeric(gsub("[$,]", "", airbnb_df$price))

response <- as.numeric(response[!is.na(response)])
response <- data.frame(response)
responseProcess <- preProcess(response, method = c("YeoJohnson"))

response <- predict(responseProcess, response)

hist(response, breaks = 100, col = 'grey', main = "Distribution of Airbnb Homestay Price")

boxplot(log(response), main = "Boxplot of Airbnb Homestay Price",
        xlab = "Price", ylab = "Price (Log Scale)")
axis(1, seq(0, 1, length.out = length(response)), 1:length(response), col = "white")


# Column names
names(predictors)

# Unique column types
unique(sapply(predictors, class) == 'logical')

# Cont or Cat columns
continuous_columns <- names(predictors)[sapply(predictors, is.numeric)]
categorical_columns <- names(predictors)[sapply(predictors, function(col) is.factor(col) | is.character(col ))]
logical_columns <- names(predictors)[sapply(predictors, function(col) is.logical(col))]
date_columns <- names(predictors)[sapply(predictors, function(col) inherits(col, "Date"))]

length(continuous_columns)
length(categorical_columns)

length(logical_columns)
length(date_columns)

length(predictors)


barplot(table(predictors$host_response_time	))

barplot(table(predictors$room_type))


barplot(table(predictors$state), horiz = TRUE)




# Create a data frame of counts for each state
state_counts <- predictors %>%
  count(state)

# Create the horizontal bar plot using ggplot
ggplot(state_counts, aes(x = reorder(state, n), y = n)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +  # Flip the coordinates to make it horizontal
  labs(x = "State", y = "Count", title = "Count of States") +
  theme_minimal()


# List of specific column names
columns_to_plot <- c("host_response_time", "host_response_rate",
                     "host_acceptance_rate","host_total_listings_count", "accommodates",
                     "bathrooms","bedrooms", "beds",
                     "number_of_reviews","review_scores_rating", "reviews_per_month", 
                     "host_response_time")

par(mfrow=c(3,4))
# Loop through each column name to plot the histogram

for (col in columns_to_plot) {
  # Check if the column exists in the data
  if (col %in% names(predictors)) {
    barplot(table(predictors[[col]]), 
         main = paste( col), 
         xlab = col, 
         col = "lightblue", 
         border = "black")
  }
}




# skew_df

kable(pred_skewness, caption = "Skewness Analysis Table", format = "markdown")


# Predictors with description that can be removed as it cannot be 1-hot-encoded

predictors <- subset(predictors, select = -c(id, listing_url, scrape_id, last_scraped, source,
                                             name, description,neighborhood_overview,picture_url,
                                             host_id, host_location, host_verifications,
                                             neighbourhood, neighbourhood_cleansed,
                                             neighbourhood_group_cleansed, calendar_updated,
                                             bathrooms_text, calendar_last_scraped,
                                             host_url, host_name, host_about, license,
                                             host_thumbnail_url, host_picture_url, 
                                             host_neighbourhood,property_type,
                                             amenities, host_listings_count,
                                             minimum_minimum_nights,maximum_minimum_nights,
                                             minimum_maximum_nights,maximum_maximum_nights,
                                             minimum_nights_avg_ntm,maximum_nights_avg_ntm,
                                             availability_60,availability_90,availability_365,
                                             number_of_reviews_ltm, number_of_reviews_l30d,
                                             review_scores_rating,review_scores_accuracy,
                                             review_scores_cleanliness, review_scores_checkin,
                                             review_scores_communication,review_scores_location,
                                             calculated_host_listings_count_entire_homes,
                                             calculated_host_listings_count_private_rooms,
                                             calculated_host_listings_count_shared_rooms
                                             ))


kable(head(predictors), format = "html", table.attr = "class='table table-bordered'") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"))

dim(predictors)

# Extracting date into multiple columns
# host_since
# first_review
# last_review

## Converting the columns into Date format


predictors$host_since <- as.Date(predictors$host_since)
predictors$first_review <- as.Date(predictors$first_review)
predictors$last_review <- as.Date(predictors$last_review)

# creating is_weekend_or_not column using date/month/year
# predictors$weekend <- wday(predictors$date, label = TRUE) %in% c("Sat", "Sun")
# not needed as host_since, first_review and last_review does 
# not make sense to have is_weekend or not

# Creating columns for day, month and year
predictors$host_since_day <- day(predictors$host_since)
predictors$host_since_month <- month(predictors$host_since)
predictors$host_since_year <- year(predictors$host_since)

predictors$first_review_day <- day(predictors$first_review)
predictors$first_review_month <- month(predictors$first_review)
predictors$first_review_year <- year(predictors$first_review)

predictors$last_review_day <- day(predictors$last_review)
predictors$last_review_month <- month(predictors$last_review)
predictors$last_review_year <- year(predictors$last_review)

# Removing date format columns
predictors <- subset(predictors, select = -c(host_since,first_review, last_review))

kable(head(predictors), format = "html", table.attr = "class='table table-bordered'") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"))


# Changing True/False variables to 1/0

# host_is_superhost,host_has_profile_pic	host_identity_verified,has_availability,instant_bookable


logical_to_num <- function(col){
  ifelse(col, 1, 0)
}

predictors$host_is_superhost <- logical_to_num(predictors$host_is_superhost)
predictors$host_has_profile_pic <- logical_to_num(predictors$host_has_profile_pic)
predictors$host_identity_verified <- logical_to_num(predictors$host_identity_verified)
predictors$has_availability <- logical_to_num(predictors$has_availability)
predictors$instant_bookable <- logical_to_num(predictors$instant_bookable)

kable(head(predictors), format = "html", table.attr = "class='table table-bordered'") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"))


# Fill N/A string values with NA


predictors <- predictors %>%
  mutate(across(where(is.character), ~ na_if(., "N/A")))


# Remove percentage in columns
predictors$host_response_rate <- as.numeric(gsub("%", "", predictors$host_response_rate))
predictors$host_acceptance_rate <- as.numeric(gsub("%", "", predictors$host_acceptance_rate))
response <- as.numeric(gsub("%", "", response))



#  Filling NA values in categorical columns

# Function to calculate mode
# get_mode <- function(x) {
#   uniq_x <- unique(x)
#   uniq_x[which.max(tabulate(match(x, uniq_x)))]
# }
# 
# 
# predictors <- predictors %>%
#   mutate(host_response_time = ifelse(is.na(host_response_time),
#                                      get_mode(host_response_time),
#                                      host_response_time))
# 
# predictors <- predictors %>%
#   mutate(v = ifelse(is.na(room_type), get_mode(room_type), room_type))
# 
# sum(is.na(predictors$room_type))
# 
# 
# predictors <- subset(predictors, select = -v)
# 
# kable(head(predictors), format = "html", table.attr = "class='table table-bordered'") %>%
#   kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"))
# 
# 
# 
# is.na(predictors_im)
# any(is.na(predictors_im))
# 
# 
# 
# kable(head(predictors), format = "html", table.attr = "class='table table-bordered'") %>%
#   kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"))
# 

predictors$bedrooms <- as.factor(predictors$bedrooms)
predictors$host_since_month <- as.factor(predictors$host_since_month)
predictors$first_review_month <- as.factor(predictors$first_review_month)
predictors$last_review_month <- as.factor(predictors$last_review_month)

# Dummy variables for categorical data

predictors <- subset(predictors, select = -city)

# dummy_var <- dummyVars("~host_response_time+room_type+state", data=predictors,
                       # fullRank=TRUE)

dummy_var <- dummyVars("~host_response_time+room_type+bedrooms+host_since_month+first_review_month+last_review_month",
                       data=predictors, fullRank=TRUE)

add_dummy <- data.frame(predict(dummy_var, newdata=predictors))


predictors <- cbind(predictors, add_dummy)


predictors <- subset(predictors, select = -c(
state, host_response_time,room_type,bedrooms,
host_since_month, first_review_month,last_review_month))

# predictors <- subset(predictors, select = -city)

# predictors <- subset(predictors, select = -bedrooms)

dim(predictors)

# knnImpute for predictor variables
# 
# 
# Im <- preProcess(predictors,method=c("knnImpute"))
# ## Apply imputation
# predictors_im <- predict(Im,predictors)

# parallel processing

library(doParallel)
num_cores <- parallel::detectCores()

# Create a cluster
cl <- makeCluster(num_cores - 2)  # Use one less core to avoid freezing your system

# Register the parallel backend
registerDoParallel(cl)


# Does center and scaling with knnimpute (preProcess function)
system.time({
  Im <- preProcess(predictors, method = c("knnImpute"))
  predictors_im <- predict(Im, predictors)
})

write.csv(predictors_im, "C:\\Users\\karth\\OneDrive\\Desktop\\Class Notes\\MA 5790\\Project\\Project Part 1 PreProcessing\\Predictors_KnnImputed_WA.csv", row.names = FALSE)
par(mfrow = c(1, 1))
image(is.na(predictors_im), main = "Missing Values",
      xlab = "Observation", ylab = "Variable",
      xaxt = "n", yaxt = "n", bty = "n",
      col = c("yellow", "maroon"))
axis(1, seq(0, 1, length.out = nrow(airbnb_df)), 1:nrow(airbnb_df), col = "white")

View(predictors_im)


kable(head(predictors_im), format = "html", table.attr = "class='table table-bordered'") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"))

dim(predictors_im)

predictors_im <- subset(predictors_im, select = -c(
  # state, host_response_time,room_type,
  host_since_month, first_review_month,last_review_month))
predictors_im <- subset(predictors_im, select = -city)

predictors_im <- subset(predictors_im, select = -bedrooms)



# knn imputed csv dataframe


predictors_im <- read.csv("C:/Users/karth/OneDrive/Desktop/Class Notes/MA 5790/Project/Project Part 1 PreProcessing/predictors_im.csv",
                           sep = ",")

(predictors_im)

dim(predictors_im)


predictors_im <- subset(predictors_im, select = -c( host_listings_count,
                                             minimum_minimum_nights,maximum_minimum_nights,
                                             minimum_maximum_nights,maximum_maximum_nights,
                                             minimum_nights_avg_ntm,maximum_nights_avg_ntm,
                                             availability_60,availability_90,availability_365,
                                             number_of_reviews_ltm, number_of_reviews_l30d,
                                             review_scores_rating,review_scores_accuracy,
                                             review_scores_cleanliness, review_scores_checkin,
                                             review_scores_communication,review_scores_location,
                                             calculated_host_listings_count_entire_homes,
                                             calculated_host_listings_count_private_rooms,
                                             calculated_host_listings_count_shared_rooms))

# Skewness
skewValues <- apply(predictors_im, 2, skewness)

skew_or_not <- function(x) {
  if (x > 1 || x < -1){
    return('Highly Skewed')
  }
  else if ((x > 0.5 && x < 1) || (x < -0.5 && x > -1)){
    return('Moderately Skewed')
  }
  else{
    return('Approx. Symmetric')
  }
}

skewValues <- data.frame(skewValues)



skewValues$Skewness <- sapply(!is.na(skewValues$skewValues), skew_or_not)

# skew_df

kable(skewValues, caption = "Skewness Analysis Table", format = "markdown")

kable(head(skewValues), format = "html", table.attr = "class='table table-bordered'") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"))


# Near zero variance predictors
zero_var <- nearZeroVar(predictors_im)
predictors_im <- predictors_im[, -zero_var]

dim(predictors_im)
# Deleting predictors
pred_corr <- cor(predictors_im)

corrplot(pred_corr, order = 'hclust')

library(corrplot)

# par(mar = c(1, 1, 1, 1))
# options(repr.plot.width=5, repr.plot.height=5)
# 
# # Assuming pred_corr is your correlation matrix
# corrplot(pred_corr, order = 'hclust',
#          col = NULL,bg = "white",
#          tl.col = "transparent",
#          addgrid.col = "grey")


par(mfrow=c(1, 1), mar=c(2, 2, 2, 2))  # Modify margins as needed
windows(width = 12, height = 10)
# Create the correlation plot
corrplot(pred_corr, order = 'hclust',
         col = NULL, bg = "white",
         tl.col = "transparent",
         addgrid.col = "grey")


dim(pred_corr)

highCorr <- findCorrelation(pred_corr, cutoff = .8)

filter_pred <- predictors_im[, -highCorr]

dim(filter_pred)


kable(head(filter_pred), format = "html", table.attr = "class='table table-bordered'") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"))



hist(filter_pred)
boxplot(filter_pred)
# BoxCox

pred_boxcox <- preProcess(filter_pred, method = c("YeoJohnson"))

pred_trans <- predict(pred_boxcox, filter_pred)

pred_boxcoxnzv <- preProcess(predictors_im, method = c("YeoJohnson"))

pred_transnzv <- predict(pred_boxcoxnzv, predictors_im)


# SpatialSign


predictors_spatial_sign <- spatialSign(pred_trans)

predictors_spatial_signnzv <- spatialSign(pred_transnzv)



# Training and Testin Data Split


set.seed(37)

dataPartition <- createDataPartition(response[, 1], p = 0.8, list = FALSE)

trainx <- predictors_spatial_sign[dataPartition, ]
trainy <- response[dataPartition]

testx <- predictors_spatial_sign[-dataPartition, ]
testy <- response[-dataPartition]

trainx_nzv <- predictors_spatial_signnzv[dataPartition]
trainy_nzv <- response[dataPartition]

testx_nzv <- predictors_spatial_signnzv[-dataPartition, ]
testy_nzv <- response[-dataPartition]


ctrl <- trainControl(method = "cv", number = 10,
                     verboseIter = TRUE,
                     allowParallel = TRUE)

# Linear Models

# Ordinary Linear Regression

trainOLR <- train(trainx, trainy,
                   method = "lm",
                  # preProcess = c("center", "scale"),
                  metric = "RMSE",
                  trControl = ctrl)

trainOLR


testOLR <- predict(trainOLR, testx)

defaultSummary(data.frame(obs = testy, pred = testOLR))


# Ridge Regression


ridgeGrid <- data.frame(.lambda = seq(0, 1, length = 15))

trainRidge <- train(trainx, trainy, method = "ridge", tuneGrid = ridgeGrid,
                     # preProcess = c("center", "scale", "BoxCox", "spatialSign"),
                    metric = "RMSE",
                     trControl = ctrl)

trainRidge
plot(trainRidge)

testRidge <- predict(trainRidge, testx)

defaultSummary(data.frame(obs = testy, pred = testRidge))


# Lasso Model

lassoGrid <- expand.grid(fraction = seq(0.01, 1, length = 10))
trainLasso <- train(trainx, trainy, method = "lasso", tuneGrid = lassoGrid,
                     # preProcess = c("center", "scale", "BoxCox", "spatialSign") ,
                    metric = "RMSE",
                    trControl = ctrl)


trainLasso
plot(trainLasso)

testLasso <- predict(trainLasso, testx)

defaultSummary(data.frame(obs = testy, pred = testLasso))

# Elastic Net Model

enetGrid <- expand.grid(.lambda = c(0, 0.1, .1), .fraction = seq(.05, 1, length = 20))
trainEnet <- train(trainx, trainy, method = "enet", tuneGrid = enetGrid,
                    # preProcess = c("center", "scale", "BoxCox", "spatialSign") ,
                   metric = "RMSE",
                    trControl = ctrl)

trainEnet
plot(trainEnet)


testEnet <- predict(trainEnet, testx)

defaultSummary(data.frame(obs = testy, pred = testEnet))

# PLS


trainPLS <- train(trainx, trainy, method = "pls", tuneLength = 20, 
                 trControl = ctrl, metric = "RMSE")
trainPLS

plot(trainPLS)

testPLS <- predict(trainPLS, testx)

defaultSummary(data.frame(obs = testy, pred = testPLS))




# Non-Linear Models


# KNN
system.time({
trainKNN <- train(x = trainx, y = trainy,
                  method = "knn",
                  # preProcess = c("center", "scale", "BoxCox", "spatialSign"),
                  metric = "RMSE",
                  tuneLength = 20)
})
trainKNN

plot(trainKNN)

testKNN <- predict(trainKNN, newdata = testx)

postResample(pred = testy, obs = testKNN)

# MARS
system.time({
marsGrid <- expand.grid(.degree = 1:3, .nprune = 1:40)

trainMARS <- train(trainx, trainy,
                   method = "earth",
                   tuneGrid = marsGrid,
                   # preProcess = c("center", "scaling", "BoxCox", "spatialSign"),
                   trControl = ctrl,
                   metric = "RMSE")
})
trainMARS
plot(trainMARS)

testMARS <- predict(trainMARS, newdata = testx)

postResample(pred = testMARS, obs = testy)

# Neural Network
library(doParallel)
num_cores <- parallel::detectCores()

# Create a cluster
cl <- makeCluster(num_cores - 2)  # Use one less core to avoid freezing your system

# Register the parallel backend
registerDoParallel(cl)

nnetGrid <- expand.grid(.decay = c(0, .1, 1, 2),
                        .size = c(1:10), 
                        .bag = FALSE)
system.time({
trainNN <- train(x = trainx, y = trainy,
                  method = "avNNet",
                  tuneGrid = nnetGrid,
                  trControl = ctrl,
                  metric = "RMSE",
                  preProc = c("center", "scale"),
                  linout = TRUE,
                  trace = FALSE,
                  MaxNWts = 10 * (ncol(trainx) + 1) + 10 + 1,
                  maxit = 200)
})
trainNN

plot(trainNN)

testNN <- predict(trainNN, newdata = testx)

postResample(pred = testNN, obs = testy)

# SVM
# Define a grid of hyperparameters for SVM
library(kernlab)
sigmaRangeReduced <- sigest(as.matrix(trainx))
svmGrid <- expand.grid(
  .sigma = seq(0, 1, 0.1),
  .C = 2^(seq(-4, 6))
)

# parallel processing
num_cores <- parallel::detectCores()

# Create a cluster
cl <- makeCluster(num_cores - 2)  # Use one less core to avoid freezing your system

# Register the parallel backend
registerDoParallel(cl)

# Train the SVM model
system.time({
trainSVM <- train(
  x = trainx, 
  y = trainy,
  method = "svmRadial",             # SVM with RBF kernel
  metric = "RMSE",
  tuneGrid = svmGrid,               # Candidate parameter grid
  # preProcess = c("center", "scale", "BoxCox", "spatialSign"),
  tuneLength = 14,
  trControl = ctrl                  # Control parameters (e.g., cross-validation)
)
})
# Summarize the results
summary(trainSVM)

# Plot the tuning results
plot(trainSVM)

# Make predictions on the test set
testSVM <- predict(trainSVM, newdata = testx)

# Evaluate model performance
postResample(pred = testSVM, obs = testy)






# Table Summary



results <- data.frame(
  Model = c("OLR", "Ridge", "Lasso", "Enet", "PLS", "KNN", "MARS", "NN", "SVM"),
  Best_Tuning_Parameter = c(
    paste("Intercept = ", trainOLR$bestTune$intercept), # OLR
    paste("Lambda = ", trainRidge$bestTune$lambda),  # Ridge
    paste("Fraction = ", trainLasso$bestTune$fraction),  # Lasso
    paste("Fraction = ", trainEnet$bestTune$fraction, ", Lambda = ", trainEnet$bestTune$lambda),  # Enet
    paste("ncomp = ", trainPLS$bestTune$ncomp), # PLS
    paste("k = ", trainKNN$bestTune$k), # KNN
    paste("nPrune = ", trainMARS$bestTune$nprune, ", Degree = ", trainMARS$bestTune$degree), # MARS
    paste("size = ", trainNN$bestTune$size, ", decay = ", trainNN$bestTune$decay, 
          ", bag = ", trainNN$bestTune$bag), # NN
    paste("Sigma = ", trainSVM$bestTune$sigma, ", C = ", trainSVM$bestTune$C) # SVM
  ),
  Training_RMSE = c(
    trainOLR$results$RMSE[which.max(trainOLR$results$RMSE)],
    trainRidge$results$RMSE[which.max(trainRidge$results$RMSE)],
    trainLasso$results$RMSE[which.max(trainLasso$results$RMSE)],
    trainEnet$results$RMSE[which.max(trainEnet$results$RMSE)],
    trainPLS$results$RMSE[which.max(trainPLS$results$RMSE)],
    trainKNN$results$RMSE[which.max(trainKNN$results$RMSE)],
    trainMARS$results$RMSE[which.max(trainMARS$results$RMSE)],
    trainNN$results$RMSE[which.max(trainNN$results$RMSE)],
    trainSVM$results$RMSE[which.max(trainSVM$results$RMSE)]
  ),
  Testing_RMSE = c(
    unname(postResample(pred = testOLR, obs = testy)[1]),
    unname(postResample(pred = testRidge, obs = testy)[1]), 
    unname(postResample(pred = testLasso, obs = testy)[1]), 
    unname(postResample(pred = testEnet, obs = testy)[1]),
    unname(postResample(pred = testPLS, obs = testy)[1]),
    unname(postResample(pred = testKNN, obs = testy)[1]),
    unname(postResample(pred = testMARS, obs = testy)[1]),
    unname(postResample(pred = testNN, obs = testy)[1]),
    unname(postResample(pred = testSVM, obs = testy)[1])
  ),
    Training_R2 = c(
    trainOLR$results$Rsquared[which.max(trainOLR$results$Rsquared)],
    trainRidge$results$Rsquared[which.max(trainRidge$results$Rsquared)],
    trainLasso$results$Rsquared[which.max(trainLasso$results$Rsquared)],
    trainEnet$results$Rsquared[which.max(trainEnet$results$Rsquared)],
    trainPLS$results$Rsquared[which.max(trainPLS$results$Rsquared)],
    trainKNN$results$Rsquared[which.max(trainKNN$results$Rsquared)],
    trainMARS$results$Rsquared[which.max(trainMARS$results$Rsquared)],
    trainNN$results$Rsquared[which.max(trainNN$results$Rsquared)],
    trainSVM$results$Rsquared[which.max(trainSVM$results$RMSE)]
  ),
  Testing_R2 = c(
    unname(postResample(pred = testOLR, obs = testy)[2]),
    unname(postResample(pred = testRidge, obs = testy)[2]), 
    unname(postResample(pred = testLasso, obs = testy)[2]), 
    unname(postResample(pred = testEnet, obs = testy)[2]),
    unname(postResample(pred = testPLS, obs = testy)[2]),
    unname(postResample(pred = testKNN, obs = testy)[2]),
    unname(postResample(pred = testMARS, obs = testy)[2]),
    unname(postResample(pred = testNN, obs = testy)[2]),
    unname(postResample(pred = testSVM, obs = testy)[2])
  )
)



kable(results, 
      col.names = c("Model", "Best Tuning Parameter", "Training RMSE", "Testing RMSE", "Training R2", "Testing R2"), 
      caption = "Model Performance Summary") %>%
  kable_styling(full_width = FALSE, position = "left") %>%
  row_spec(0, bold = TRUE, background = "#f2f2f2") %>%  # Header row
  column_spec(1, bold = TRUE)

varImp(trainNN)
plot(varImp(trainNN), top = 10)
########## END!!!!!!!!!!!!!! #############################################


# BoxCox and spatialSign Transformation
# # 
# par(mfrow=c(3,3))
# for(col in zero_var[1:9]){
#   barplot(table(filter_pred[col]), names = colnames(filter_pred[col]))
# }
# 
# 
# filter_pred <- filter_pred[, -zero_var]
# {plot.new(); dev.off()}
# 
# par(mfrow=c(1, 1), mar=c(2, 2, 2, 2))  # Modify margins as needed
# windows(width = 12, height = 10)
# corrplot(cor(filter_pred), order = 'hclust',
#          col = NULL,bg = "white",
#          tl.col = "transparent",
#          addgrid.col = "grey")
# 
# 
# {plot.new(); dev.off()}
# par(mfrow=c(3,3))
# for (i in c(5,16,29,39,42,44,45,46,47)) {
#   hist(predictors_im[,i], main=names(predictors_im[i]), col = 'black')
#   
# }
# 
# 
# kable(head(filter_pred), format = "html", table.attr = "class='table table-bordered'") %>%
#   kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"))
# 
# dim(filter_pred)

# BoxCox Transformation


pred_boxcox <- preProcess(filter_pred, method = c("BoxCox", "center", "scale"))

pred_trans <- predict(pred_boxcox, filter_pred)

kable(head(predictors_spatial_sign), format = "html", table.attr = "class='table table-bordered'") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"))

table(predictors_im[["latitude"]])


columns_to_plot <- c("bathrooms", "bedrooms", "host_acceptance_rate",
                     "number_of_reviews", "host_response_rate",
                     "host_total_listings_count", "maximum_nights",
                     "review_scores_cleanliness")

existing_columns <- columns_to_plot %in% colnames(pred_trans)

existing_columns

par(mfrow=c(3,3))

for (col in columns_to_plot) {
  
  if (col %in% names(filter_pred)) {
    
    data_vector <- filter_pred[[col]]
    
    
    if (length(filter_pred) > 0 && all(!is.na(filter_pred))) {
      
      hist(filter_pred[[col]], 
              main = col, 
              col = 'skyblue',
              xaxt = "n")  
    }
  }}

par(mfrow=c(3,3))
for (col in columns_to_plot) {
  
  if (col %in% names(pred_trans)) {
    
    data_vector <- pred_trans[[col]]
    
    
    if (length(pred_trans) > 0 && all(!is.na(pred_trans))) {
      
      hist(pred_trans[[col]], 
           main = col, 
           col = 'skyblue',
           xaxt = "n")  
    }
  }}

# Before Spatial Sign Transformation

par(mfrow=c(3,3))
for (col in columns_to_plot) {
  boxplot(pred_trans[col], main=names(pred_trans[col]),
          type="l", col = 'skyblue')
  
}
names(pred_trans)

predictors_spatial_sign <- spatialSign(pred_trans)


any(is.infinite(predictors_spatial_sign))

predictors_spatial_sign <- data.frame(predictors_spatial_sign)

# After Spatial Sign Transformation
par(mfrow=c(3,3))
for (col in columns_to_plot) {
  boxplot(predictors_spatial_sign[col], main=names(predictors_spatial_sign[col]),
          type="l", col = 'skyblue')
  
}
for (col in columns_to_plot) {
  
  if (col %in% names(predictors_spatial_sign)) {
    
    data_vector <- predictors_spatial_sign[[col]]
    
    
    if (length(data_vector) > 0 && all(!is.na(data_vector))) {
      
      boxplot(data_vector, 
              main = col, 
              col = 'skyblue', 
              type = 'l',
              ylab = col)  
    }
}}


dim(predictors_spatial_sign)
dim(predictors0)




transformation_skewness <- apply(predictors_spatial_sign, 2, skewness)

skew_or_not <- function(x) {
  if (x > 1 || x < -1){
    return('Highly Skewed')
  }
  else if ((x > 0.5 && x < 1) || (x < -0.5 && x > -1)){
    return('Moderately Skewed')
  }
  else{
    return('Approx. Symmetric')
  }
}

transformation_skewness <- data.frame(transformation_skewness)



transformation_skewness$Skewness <- sapply(transformation_skewness$transformation_skewness, skew_or_not)

# skew_df

kable(transformation_skewness, caption = "Skewness Analysis Table", format = "markdown")

kable(head(transformation_skewness), format = "html", table.attr = "class='table table-bordered'") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"))




# PCA

dim(predictors_im)

pca_pred <- preProcess(predictors_im, method = c("pca"))
pca_pred
# PCA needed 52 components to capture 95 percent of the variance

str(pca_predicted_pred)


pca_predicted_pred <- predict(pca_pred, predictors_im)
dim(predictors_im)
dim(pca_predicted_pred)

hist(pca_predicted_pred)

# PCA Plotting
num_components <- ncol(pca_predicted_pred)

# We can retrieve the PCA model matrix to calculate the variances
pca_model_matrix <- as.matrix(pca_predicted_pred)

# Calculate the variance for each principal component
variance_explained <- apply(pca_model_matrix, 2, var)

# Normalize to get the proportion of variance explained
explained_variance <- variance_explained / sum(variance_explained)


par(mfrow=c(1,1))
# Plot cumulative explained variance
plot(cumsum(explained_variance), type = "o",
     xlab = "Principal Component", 
     ylab = "Cumulative Proportion of Variance Explained",
     main = "Cumulative Explained Variance by Principal Components")

# Only for PCA Plotting without matrix multiplication
pca_result <- prcomp(predictors_im, center = TRUE, scale. = TRUE)
summary(pca_result)  # Check variance explained
sdev <- pca_result$sdev  # Get standard deviations

par(mfrow = c(1,1))

plot(cumsum(pca_result$sdev^2 / sum(pca_result[1:10]$sdev^2)), type="o",
     xlab = "PCA", ylab = "Variance", main = 'Variance explained by PCAs')


pca_result <- prcomp(predictors_im, center = TRUE, scale. = TRUE)
summary(pca_result)  # Check variance explained
sdev <- pca_result$sdev  # Get standard deviations



# Plot for PCA scatter for first 10 PCAs
plot(pca_predicted_pred[1:5])

# Set up the plotting area to have 2 rows and 3 columns (for 6 plots)
par(mfrow = c(5,5))

# Loop to plot the first 6 plots
for (i in 1:25) {
  plot(pca_predicted_pred[[i]], main = paste("Plot", i))
}













# EX:



values <- sapply(predictors, function(x) sum(length(which(!is.na(x)))))
nans <- sapply(predictors, function(x) sum(length(which(is.na(x)))))

percent.missing <- (nans/(values + nans))*100

missings <- data.frame(nans, percent.missing)
missings
missings[order(missings$nans, decreasing = TRUE),]