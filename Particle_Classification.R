############################
## PARTICLE CLASSIFICATION##
############################

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

# Add Particle Classifications to data sets

train$type <- ifelse(train$Area > 0.000005, "dust", "artifact")
test_actual <- test
test_actual$type <- ifelse(test_actual$Area > 0.000005, "dust", "artifact")

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

# Assuming your training data is stored in a variable called 'train'
normalization_funcs <- choose_best_normalization(train, response_var = "type", exclude_cols = c("...1"))
