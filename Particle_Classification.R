############################
## PARTICLE CLASSIFICATION##
############################

set.seed(101)

###############
## LIBRARIES ##
###############

library(tidymodels) 
library(tidyverse)
library(vroom) 
library(randomForest)
library(doParallel)
library(lightgbm)
library(themis)
library(bonsai)
library(bestNormalize)
library(embed)
library(pROC)
conflicted::conflicts_prefer(yardstick::rmse)
conflicted::conflicts_prefer(yardstick::accuracy)
conflicted::conflicts_prefer(yardstick::spec)

##########################
## PARRALELL PROCESSING ##
##########################

all_cores <- parallel::detectCores(logical = FALSE)
registerDoParallel(cores = all_cores)

##########
## DATA ##
##########

data <- vroom("Results.csv")
split1<- sample(c(rep(0, 0.7 * nrow(data)), rep(1, 0.3 * nrow(data))))

train <- data[split1 == 0,]
test <- data[split1 == 1,]

train <- mutate(train,id = ...1)
train <- train[,-1]

test <- mutate(test,id = ...1)
test <- test[,-1]

# Add Particle Classifications to data sets

train$type <- ifelse(train$Area > 0.000005, "dust", "artifact")
train$type <- as.factor(train$type)
test_actual <- test
test_actual$type <- ifelse(test_actual$Area > 0.000005, "dust", "artifact")
test_actual$type <- as.factor(test_actual$type)


#######
##EDA##
#######

# DataExplorer::plot_missing(train)
# DataExplorer::plot_histogram(train)
# DataExplorer::plot_bar(train)

#####################
## TRANSFORMATIONS ##
#####################

choose_best_normalization <- function(train, response_var, exclude_cols = NULL) {
  # Remove the response variable and excluded columns from the feature columns
  feature_cols <- setdiff(colnames(train), c(response_var, exclude_cols))

  # Initialize an empty list to store the chosen normalization functions
  normalization_functions <- list()

  # Get the total number of feature columns
  num_features <- length(feature_cols)

  # Iterate over each feature column using index i
  for (i in 1:num_features) {
    col_name <- feature_cols[i]

    # Check the data type of the column
    col_type <- typeof(train[[col_name]])

    if (col_type == "list") {
      # Extract numeric values from the list column
      train[[col_name]] <- sapply(train[[col_name]], function(x) x[[1]])
    }

    # Convert the column to numeric
    train[[col_name]] <- as.numeric(train[[col_name]])

    # Find the best normalization function for the column
    best_norm <- bestNormalize(train[[col_name]], allow_orderNorm = TRUE, allow_exp = TRUE)

    # Store the chosen normalization function in the list
    normalization_functions[[col_name]] <- best_norm$chosen_transform
  }

  # Return the list of chosen normalization functions
  return(normalization_functions)
}


normalization_funcs <- choose_best_normalization(train, response_var = "type", exclude_cols = c("id"))
# 
# normalization_funcs
# 
# ## Results
# Area = Boxcox(Area)
# XM = orderNorm(XM)
# YM = orderNorm(YM)
# Circ. = orderNorm(Circ.)
# AR = orderNorm(AR)
# Round = orderNorm(Round)
# Solidarity = yeojohnson(Solidarity)

#####################
## MODEL SELECTION ##
#####################

#Quick iteration subset
split2<- sample(c(rep(0, 0.05 * nrow(train)),rep(1, 0.95 * nrow(train)),1))
split3<- sample(c(rep(1, 0.05 * nrow(test)),rep(0, 0.95 * nrow(test)),1))
qi_train <- train[split2 == 0,]
qi_test <- test[split3 == 1,]

qi_test_actual <- qi_test
qi_test_actual$type <- ifelse(qi_test_actual$Area > 0.000005, "dust", "artifact")
qi_test_actual$type <- as.factor(qi_test_actual$type)


############
## RECIPE ##
############

my_recipe <- recipe(type~., data=qi_train) %>%
  update_role(id, new_role="id") %>% 
  step_rm(Area) %>%
#  step_mutate(Area = predict(boxcox(Area))) %>% # Uses a Box Cox transformation to Normalize Area Data
  step_mutate(XM = predict(orderNorm(XM))) %>%  # Uses a Order Normal transformation to Normalize XM Data
  step_mutate(YM = predict(orderNorm(YM))) %>%  # Uses a Order Normal transformation to Normalize YM Data
  step_mutate(Circ. = predict(orderNorm(Circ.))) %>%
  step_mutate(AR = predict(orderNorm(AR))) %>%
  step_mutate(Round = predict(orderNorm(Round))) %>%
  step_mutate(Solidity = predict(yeojohnson(Solidity))) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(type)) %>% 
  # step_normalize(all_numeric_predictors()) %>% 
  step_smote(all_outcomes(), neighbors=7)

prepped_recipe <- prep(my_recipe, verbose = T)
bake_1 <- bake(prepped_recipe, new_data = NULL)

###################
## BOOSTED MODEL ##
###################

boost_model <- boost_tree(tree_depth=tune(),
                          trees= tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>% # used lightgbm because it is faster
  set_mode("classification")

boost_workflow <- workflow() %>% #Creates a workflow
  add_recipe(my_recipe) %>% #Adds in my recipe
  add_model(boost_model) 

tuning_grid_boost <- grid_regular(tree_depth(),
                                  learn_rate(),
                                  trees(),
                                  levels = 5)
folds_boost <- vfold_cv(train, v = 10, repeats=1)

CV_results_boost <- boost_workflow %>%
  tune_grid(resamples=folds_boost,
            grid=tuning_grid_boost,
            metrics=metric_set(roc_auc,f_meas, sens, recall, accuracy))

bestTune_boost <- CV_results_boost %>%
  select_best("roc_auc")

final_wf_boost <- boost_workflow %>% 
  finalize_workflow(bestTune_boost) %>% 
  fit(data = qi_train)


boost_prediction <- final_wf_boost %>% 
  predict(new_data = qi_test, type="prob")


boost_predictions <- bind_cols(qi_test$id,boost_prediction$.pred_class,qi_test_actual$type)
colnames(boost_predictions) <- c("Id","Prediction","Actual")

roc(boost_predictions$Actual,boost_predictions$Prediction)

#TRIAL 1 Roc

vroom_write(boost_predictions,"boost_predictions2.csv",',')
