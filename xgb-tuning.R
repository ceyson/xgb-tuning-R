# Based on Ben Hamner script from Springleaf
# https://www.kaggle.com/benhamner/springleaf-marketing-response/random-forest-example

# Inits
libs <- c("readr","xgboost","lubridate","pROC","caret")
lapply(libs, require, character.only=TRUE)

## Seed
set.seed(1718)

## Working dir
setwd("C:/Users/Antares/Documents/Kaggle/Competitions/Homesite/")

cat("reading the train and test data\n")
train <- read_csv("./data/train.csv")
test  <- read_csv("./data/test.csv")

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- 0
test[is.na(test)]     <- 0

# Seperating out the elements of the date column for the train set
train$month <- as.integer(month(train$Original_Quote_Date))
train$year <- as.integer(year(train$Original_Quote_Date))
train$dayOfMonth <- as.integer(day(train$Original_Quote_Date))
train$wday <- as.integer(wday(train$Original_Quote_Date))
train$weekSegment <- ifelse(train$wday %in% 2:6, "week", "weekend")
train$weekSegment2[train$wday == 2] <- "bweek"
train$weekSegment2[train$wday == 3] <- "bweek"
train$weekSegment2[train$wday == 4] <- "bweek"
train$weekSegment2[train$wday == 5] <- "eweek"
train$weekSegment2[train$wday == 6] <- "eweek"
train$weekSegment2[train$wday == 1] <- "weeke"
train$weekSegment2[train$wday == 7] <- "weeke"

# Removing the date column
train <- train[,-c(2)]

# Seperating out the elements of the date column for the test set
test$month <- as.integer(month(test$Original_Quote_Date))
test$year <- as.integer(year(test$Original_Quote_Date))
test$dayOfMonth <- as.integer(day(test$Original_Quote_Date))
test$wday <- as.integer(wday(test$Original_Quote_Date))
test$weekSegment <- ifelse(test$wday %in% 2:6, "week", "weekend")
test$weekSegment2[test$wday == 2] <- "bweek"
test$weekSegment2[test$wday == 3] <- "bweek"
test$weekSegment2[test$wday == 4] <- "bweek"
test$weekSegment2[test$wday == 5] <- "eweek"
test$weekSegment2[test$wday == 6] <- "eweek"
test$weekSegment2[test$wday == 1] <- "weeke"
test$weekSegment2[test$wday == 7] <- "weeke"

# Removing the date column
test <- test[,-c(2)]

## Numericize all predictors
feature.names <- names(train)[c(3:304)]
cat("Feature Names\n")
feature.names

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

################################################################################################
#
# Parameter Tuning -- https://github.com/msegala/Kaggle-Springleaf/blob/master/XGBoost_Step1.R
#
# Problem: Running XgBoost itself results in a situation where the AUC maximizes at 1.
#
# Solution: Split the train (labeled) data into a 70% holdout set and 30% training set.
#           For a grid of tuning parameters, develop models that are applied to the holdout
#           data from which an AUC is computed. This AUC should theoretically approximate
#           the performance of the unseen test Homesite data. 
#
#           Parameters and AUC are stored to dataframe for examination.
#
################################################################################################

max_depth_        = c(6,7,8)
eta_              = c(0.018,0.019,0.020,0.021,0.022,0.023)
nround_           = c(1800,1900,2000)
# gamma_            = c(0)
# min_child_weight_ = c(1)
subsample_        = c(0.81,0.82,0.83)
colsample_bytree_ = c(0.75,0.76,0.77,0.78)

dfTune = data.frame(i                = numeric(),
                    max_depth        = numeric(),
                    eta              = numeric(),
                    nround           = numeric(),
                    # gamma            = numeric(),
                    # min_child_weight = numeric(),
                    subsample        = numeric(),
                    colsample        = numeric(),
                    # best_train       = numeric(),
                    # best_test        = numeric(),
                    auc              = numeric())

## Seed
set.seed(1221)

## Split data
index <- createDataPartition(train$QuoteConversion_Flag, p = .70, list = FALSE) 
holdout <- train[-index,]
validation <- train[index,]

## Sample data
tra <- holdout[,feature.names]
h <- sample(nrow(holdout),2000)

## XgBoost
xgval <- xgb.DMatrix(data=data.matrix(tra[h,]),label=holdout$QuoteConversion_Flag[h])
xgtrain <- xgb.DMatrix(data=data.matrix(tra),label=holdout$QuoteConversion_Flag)

gc()
watchlist<-list(val=xgval,train=xgtrain)

i = 1
for (m in max_depth_){
  for (e in eta_){
    for (n in nround_){
      ## for (g in gamma_){
        ## for (mi in min_child_weight_){
          for (s in subsample_){
            for (c in colsample_bytree_){
              
              param <- list(max.depth        = m,
                            eta              = e,
                            # gamma            = g,
                            # min_child_weight = mi,
                            subsample        = s,
                            colsample_bytree = c,                            
                            silent           = 1, 
                            objective        = "binary:logistic",
                            booster          = "gbtree",
                            eval_metric      = "auc",
                            print.every.n    = 5)
              
              clf <- xgb.train(params            = param,  
                               data              = xgtrain, 
                               nrounds           = n,      
                               verbose           = 1,   
                               # early.stop.round  = 10,
                               watchlist         = watchlist, 
                               maximize          = FALSE)

              ## Generate predictions (probs) on holdout
              predictions <- predict(clf, data.matrix(validation[,feature.names]))

              ## AUC
              reality <- validation[,c("QuoteNumber" ,"QuoteConversion_Flag")]
              predicted <- as.data.frame(cbind(validation$QuoteNumber,predictions))
              colnames(predicted) <- c("QuoteNumber","Prediction")

              validate <- merge(reality,predicted)

              auc <- auc(validate$QuoteConversion_Flag, validate$Prediction)
              print(auc)
              
              # best_train = clf$bestScore
              # best_test  = clf$bestScore
              
              cat("iteration = ", i,": Max_Depth, Eta, NRound,                          Subsample, ColSample = ",m,e,n,       s,c,"AUC = ",auc, "\n")
                                                              # Gamma, Min_Child_Weight,                               # g,mi,
              
              dfTune[i,] <- c(i,m,e,n,       s,c,                    auc)
                                  # g,mi,        best_train,best_test
              i = i + 1              
              
              print(dfTune)
              
            }
          }
        ## }
      ## }
    }
  }
}

## Best model specification
dfTune[which(dfTune$auc == max(dfTune$auc)), ]
dfOpt <- dfTune[which(dfTune$auc == max(dfTune$auc)), ]

## Set optimal parameters
max_depth_opt          <- dfOpt[,"max_depth"]
eta_opt                <- dfOpt[,"eta"]
nround_opt             <- dfOpt[,"nround"]
# gamma_opt             <- dfOpt[,"gamma_opt"]
# min_child_weight_opt  <- dfOpt[,"min_child_weight_opt"]
subsample_opt          <- dfOpt[,"subsample"]
colsample_bytree_opt   <- dfOpt[,"colsample"]

## Set seed
set.seed(1976)

## Sample data
tra <- train[,feature.names]
h <- sample(nrow(holdout),2000)

## Optimized(?) XgBoost Specification
dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=train$QuoteConversion_Flag[h])
dtrain<-xgb.DMatrix(data=data.matrix(tra),label=train$QuoteConversion_Flag)

watchlist<-list(val=dval,train=dtrain)
param <- list(objective        = "binary:logistic", 
              booster          = "gbtree",
              eval_metric      = "auc",
              eta              = eta_opt,
              max_depth        = max_depth_opt,
              subsample        = subsample_opt,
              colsample_bytree = colsample_bytree_opt
              # num_parallel_tree  = 2
              # alpha = 0.0001, 
              # lambda = 1
)

clf <- xgb.train(params           = param, 
                 data             = dtrain, 
                 nrounds          = nround_opt, 
                 verbose          = 1,  #1
                 #early.stop.round = 150,
                 watchlist        = watchlist,
                 maximize         = FALSE
)

## Predictions and submission
pred1 <- predict(clf, data.matrix(test[,feature.names]))
submission <- data.frame(QuoteNumber=test$QuoteNumber, QuoteConversion_Flag=pred1)
cat("saving the submission file\n")
write_csv(submission, paste("./results/xgboost/benchmark/optimal-xgboost_eta-",eta_opt,"_depth-",max_depth_opt,"_sub-",subsample_opt,"_col-",colsample_bytree_opt,"_nround-",nround_opt,".csv", sep=""))


