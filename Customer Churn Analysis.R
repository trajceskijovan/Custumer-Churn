# -----------------------------------------------------------------
# Customer Churn Analysis
#
# Jovan Trajceski
# -----------------------------------------------------------------

# Clear up data in global environment
rm(list=ls())

# Data Loading and Pre-processing
library(tidyverse)
library(patchwork)
library(caret)
library(vcd)
library(gridExtra)
library(knitr)
library(corrplot)
library(scales)
library(lme4)
library(DMwR2)
library(DMwR)
library(InformationValue)
library(ROCR)
library(rpart)
library(randomForest)
library(xgboost)
library(xgboostExplainer)
library(MASS)
library(ggmosaic)
library(e1071)
library(ranger)
library(penalized)
library(rpart.plot)
library(ggcorrplot)
library(caTools)
library(snow)
library(doParallel)
doFuture::registerDoFuture()
future::plan("multisession")
library(naniar)
suppressPackageStartupMessages(c(library(caret),library(corrplot),library(smotefamily)))
library(ROSE)
library(kableExtra)
library(knitr)



setwd("C:/Users/Jovan Trajceski/Downloads/Churn")

bankChurn <- read_csv('Churn_Modelling.csv')

glimpse(bankChurn)


# Data Cleaning
bankChurn <- bankChurn %>% 
    dplyr::select(-RowNumber, -CustomerId, -Surname) %>% #remove unwanted column 
    mutate(Geography = as.factor(Geography),
           Gender = as.factor(Gender),
           HasCrCard = as.factor(HasCrCard),
           IsActiveMember = as.factor(IsActiveMember),
           Exited = as.factor(Exited),
           Tenure = as.factor(Tenure),
           NumOfProducts = as.factor(NumOfProducts))

# Check NAs
sapply(bankChurn, function(x) sum(is.na(x)))
vis_miss(bankChurn)

# Data Overview
summary(bankChurn)

# Target
ggplot(bankChurn, aes(Exited, fill = Exited)) +
    geom_bar() +
    theme(legend.position = 'none')

table(bankChurn$Exited)

round(prop.table(table(bankChurn$Exited)),3)

# Continuous Variable Distribution
bankChurn %>%
    keep(is.numeric) %>%
    gather() %>%
    ggplot() +
    geom_histogram(mapping = aes(x=value,fill=key), color="black") +
    facet_wrap(~ key, scales = "free") +
    theme_minimal() +
    theme(legend.position = 'none')

# Correlation Matrix
numericVarName <- names(which(sapply(bankChurn, is.numeric)))
corr <- cor(bankChurn[,numericVarName], use = 'pairwise.complete.obs')
ggcorrplot(corr, lab = TRUE)

# Categorical Variable Distribution
bankChurn %>%
    dplyr::select(-Exited) %>% 
    keep(is.factor) %>%
    gather() %>%
    group_by(key, value) %>% 
    summarize(n = n()) %>% 
    ggplot() +
    geom_bar(mapping=aes(x = value, y = n, fill=key), color="black", stat='identity') + 
    coord_flip() +
    facet_wrap(~ key, scales = "free") +
    theme_minimal() +
    theme(legend.position = 'none')

# Age
age_hist <- ggplot(bankChurn, aes(x = Age, fill = Exited)) +
    geom_histogram(binwidth = 5) +
    theme_minimal() +
    scale_x_continuous(breaks = seq(0,100,by=10), labels = comma)

age_boxplot <- ggplot(bankChurn, aes(x = Exited, y = Age, fill = Exited)) +
    geom_boxplot() + 
    theme_minimal() +
    theme(legend.position = 'none')

age_hist | age_boxplot


# Balance
balance_hist <- ggplot(bankChurn, aes(x = Balance, fill = Exited)) +
    geom_histogram() +
    theme_minimal() +
    scale_x_continuous(breaks = seq(0,255000,by=30000), labels = comma) +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5))

balance_box <- ggplot(bankChurn, aes(x = Exited, y = Balance, fill = Exited)) +
    geom_boxplot() + 
    theme_minimal() +
    theme(legend.position = 'none')

balance_hist | balance_box

# Credit Score
credit_hist <- ggplot(bankChurn, aes(x = CreditScore, fill = Exited)) +
    geom_histogram() +
    theme_minimal() +
    #scale_x_continuous(breaks = seq(0,255000,by=30000), labels = comma) +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5))

credit_box <- ggplot(bankChurn, aes(x = Exited, y = CreditScore, fill = Exited)) +
    geom_boxplot() + 
    theme_minimal() +
    theme(legend.position = 'none')

credit_hist | credit_box

# Estimated Salary
estimated_hist <- ggplot(bankChurn, aes(x = EstimatedSalary, fill = Exited)) +
    geom_histogram() +
    theme_minimal() +
    #scale_x_continuous(breaks = seq(0,255000,by=30000), labels = comma) +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5))

estimated_box <- ggplot(bankChurn, aes(x = Exited, y = EstimatedSalary, fill = Exited)) +
    geom_boxplot() + 
    theme_minimal() +
    theme(legend.position = 'none')

estimated_hist | estimated_box

# Categorical Variables Exploration
gender_graph <- bankChurn %>%
    dplyr::select(Gender, Exited) %>% 
    table(.) %>% 
    as.data.frame() %>% 
    ggplot(.) +
    ggmosaic::geom_mosaic(aes(weight = Freq, x = product(Gender), fill = Exited)) +
    ggthemes::theme_tufte() +
    scale_fill_brewer(type = "qual") +
    labs(x = 'Gender')

geography_graph <- bankChurn %>%
    dplyr::select(Geography, Exited) %>% 
    table(.) %>% 
    as.data.frame() %>% 
    ggplot(.) +
    ggmosaic::geom_mosaic(aes(weight = Freq, x = product(Geography), fill = Exited)) +
    ggthemes::theme_tufte() +
    scale_fill_brewer(type = "qual") +
    labs(x = 'Geography')

tenure_graph <- bankChurn %>%
    dplyr::select(Tenure, Exited) %>% 
    table(.) %>% 
    as.data.frame() %>% 
    ggplot(.) +
    ggmosaic::geom_mosaic(aes(weight = Freq, x = product(Tenure), fill = Exited)) +
    ggthemes::theme_tufte() +
    scale_fill_brewer(type = "qual") +
    labs(x = 'Tenure')

HasCrCard_graph <- bankChurn %>%
    dplyr::select(HasCrCard, Exited) %>% 
    table(.) %>% 
    as.data.frame() %>% 
    ggplot(.) +
    ggmosaic::geom_mosaic(aes(weight = Freq, x = product(HasCrCard), fill = Exited)) +
    ggthemes::theme_tufte() +
    scale_fill_brewer(type = "qual") +
    labs(x = 'HasCrCard')

IsActiveMember_graph <- bankChurn %>%
    dplyr::select(IsActiveMember, Exited) %>% 
    table(.) %>% 
    as.data.frame() %>% 
    ggplot(.) +
    ggmosaic::geom_mosaic(aes(weight = Freq, x = product(IsActiveMember), fill = Exited)) +
    ggthemes::theme_tufte() +
    scale_fill_brewer(type = "qual") +
    labs(x = 'IsActiveMember')

NumOfProducts_graph <- bankChurn %>%
    dplyr::select(NumOfProducts, Exited) %>% 
    table(.) %>% 
    as.data.frame() %>% 
    ggplot(.) +
    ggmosaic::geom_mosaic(aes(weight = Freq, x = product(NumOfProducts), fill = Exited)) +
    ggthemes::theme_tufte() +
    scale_fill_brewer(type = "qual") +
    labs(x = 'NumOfProducts')

(gender_graph | geography_graph) / (IsActiveMember_graph | HasCrCard_graph ) / (tenure_graph | NumOfProducts_graph)

# Feature selection with chi-square test
chi.square <- vector()
p.value <- vector()
cateVar <- bankChurn %>% 
    dplyr::select(-Exited) %>% 
    keep(is.factor)

for (i in 1:length(cateVar)) {
    p.value[i] <- chisq.test(bankChurn$Exited, unname(unlist(cateVar[i])), correct = FALSE)[3]$p.value
    chi.square[i] <- unname(chisq.test(bankChurn$Exited, unname(unlist(cateVar[i])), correct = FALSE)[1]$statistic)
}

chi_sqaure_test <- tibble(variable = names(cateVar)) %>% 
    add_column(chi.square = chi.square) %>% 
    add_column(p.value = p.value)

knitr::kable(chi_sqaure_test)

# Formatted table
chi_sqaure_test %>%
    kbl(caption = "Feature selection with chi-square test") %>%
    kable_classic(full_width = T, html_font = "Cambria") %>%
    row_spec(3:3, bold = T, color = "white", background = "red") %>%
    row_spec(5:5, bold = T, color = "white", background = "red") %>%
    footnote(general = "The chi-square for Tenure and HasCrCard are pretty small, at the same time, their p-values are greater than 0.05, so it confirms our hypothesis that these two features will not provide useful information on the reponse (target) variable. Thus I decided to drop these two variables.",
    )


# Drop 2 variables with pval > 0.05
bankChurn <- bankChurn %>% 
    dplyr::select(-Tenure, -HasCrCard)

# Build Predictive Models
# Data Partition: split the data using a stratified sampling approach.
set.seed(1234)
sample_set <- bankChurn %>%
    pull(.) %>% 
    sample.split(SplitRatio = .7)

bankTrain <- subset(bankChurn, sample_set == TRUE)
bankTest <- subset(bankChurn, sample_set == FALSE)


# Class Balancing: Let’s look at the class distribution again.
round(prop.table(table(bankChurn$Exited)),3)

round(prop.table(table(bankTrain$Exited)),3)

round(prop.table(table(bankTest$Exited)),3)

# Balance train set

# check table
table(bankTrain$Exited)

#check classes distribution
prop.table(table(bankTrain$Exited))

#over sampling
data_balanced_over <- ovun.sample(Exited ~ ., data = bankTrain, method = "over",N = 11148)$data
table(data_balanced_over$Exited)
# method over instructs the algorithm to perform over sampling. 
# N refers to number of observations in the resulting balanced set. 

# Undersampling
data_balanced_under <- ovun.sample(Exited ~ ., data = bankTrain, method = "under", N = 2852, seed = 1)$data
table(data_balanced_under$Exited)

# Now the data set is balanced. But, you see that we’ve lost significant information from the sample. 
# Let’s do both undersampling and oversampling on this imbalanced data. 
# This can be achieved using method = “both“. 
# In this case, the minority class is oversampled with replacement and majority class is undersampled without replacement.
data_balanced_both <- ovun.sample(Exited ~ ., data = bankTrain, method = "both", p=0.5, N=7000, seed = 1)$data
table(data_balanced_both$Exited)


# Logistic Regression
## Train the model
logit.mod <- glm(Exited ~., family = binomial(link = 'logit'), data = data_balanced_both)

## Look at the result
summary(logit.mod)

## Predict the outcomes against our test data
logit.pred.prob <- predict(logit.mod, bankTest, type = 'response')
logit.pred <- as.factor(ifelse(logit.pred.prob > 0.5, 1, 0))

head(bankTest,10)
head(logit.pred.prob,10)


#View the confusion matrix of logistic regression.
caret::confusionMatrix(logit.pred, bankTest$Exited, positive = "1")

# Decision Tree
ctrl <-
    trainControl(method = "cv", #cross-validation
                 number = 10, #10-fold
                 selectionFunction = "best")

grid <- 
    expand.grid(
        .cp = seq(from=0.0001, to=0.005, by=0.0001)
    )
set.seed(1234)
tree.mod <-
    train(
        Exited ~.,
        data = data_balanced_both,
        method = "rpart",
        metric = "Kappa",
        trControl = ctrl,
        tuneGrid = grid
    )

tree.mod

# Make predictions based on our candidate model
tree.pred.prob <- predict(tree.mod, bankTest, type = "prob")
tree.pred <- predict(tree.mod, bankTest, type = "raw")

# View the confusion Matrix of decision tree.
caret::confusionMatrix(tree.pred, bankTest$Exited, positive = "1")

# Random Forest

## Create a control object.
ctrl <- trainControl(method = "cv",
                     number = 10,
                     selectionFunction = "best")

## Create a grid search based on the available parameters.
grid <- expand.grid(.mtry = c(1:8))

## Build the random forest model
rf.mod <- 
    train(Exited ~.,
          data = data_balanced_both,
          method = 'rf',
          metric = 'Kappa',
          trControl = ctrl,
          tuneGrid = grid)

rf.mod

## Make the predictions
rf.pred <- predict(rf.mod, bankTest, type = "raw")
rf.pred.prob <- predict(rf.mod, bankTest, type = "prob")

# View the confusion matrix of random forest.
caret::confusionMatrix(rf.pred, bankTest$Exited, positive = "1")


# eXtreme Gradient Boosting (XgBoost)
## Create a control object
ctrl <-
    trainControl(method = "cv",
                 number = 10,
                 selectionFunction = "best")

modelLookup("xgbTree")

## Grid Search
grid <- expand.grid(
    nrounds = 40,
    max_depth = c(4,5,6,7,8),
    eta =  c(0.1,0.2,0.3,0.4,0.5),
    gamma = 0.01,
    colsample_bytree = 1,
    min_child_weight = 1,
    subsample = c(0.5, 1)
)


## Build XGBoost
set.seed(1234)
xgb.mod <-
    train(
        Exited ~ .,
        data = data_balanced_both,
        method = "xgbTree",
        metric = "Kappa",
        trControl = ctrl,
        tuneGrid = grid
    )

xgb.mod


## Make the prediction
xgb.pred <- predict(xgb.mod, bankTest, type = "raw")
xgb.pred.prob <- predict(xgb.mod, bankTest, type = "prob")

#View the confusion matrix of XGBoost.
caret::confusionMatrix(xgb.pred, bankTest$Exited, positive = "1")

    
# Compare Models’ Performance

## Logistic Regression
test <- bankTest$Exited
pred <- logit.pred
prob <- logit.pred.prob

# Logistic Regression ROC curve
roc.pred <- prediction(predictions = prob, labels = test)
roc.perf <- performance(roc.pred, measure = "tpr", x.measure = "fpr")
plot(roc.perf, main = "ROC Curve for Customer Churn Prediction Approaches", col = 2, lwd = 2)
abline(a = 0, b = 1, lwd = 3, lty = 2, col = 1)

## Logistic Regression Performance Metrics
accuracy <- mean(test == pred)
precision <- posPredValue(pred, test, positive = "1")
recall <- caret::sensitivity(pred, test, positive = "1")
fmeasure <- (2 * precision * recall)/(precision + recall)
confmat <- caret::confusionMatrix(pred, test, positive = "1")
kappa <- as.numeric(confmat$overall["Kappa"])
auc <- as.numeric(performance(roc.pred, measure = "auc")@y.values)
comparisons <- tibble(approach="Logistic Regression", accuracy = accuracy, fmeasure = fmeasure,kappa = kappa, auc = auc)

## Classification Tree
test <- bankTest$Exited
pred <- tree.pred
prob <- tree.pred.prob[,2]

## Classification Tree ROC Curve
roc.pred <- prediction(predictions = prob, labels = test)
roc.perf <- performance(roc.pred, measure = "tpr", x.measure = "fpr")
plot(roc.perf, col=3, lwd = 2, add=TRUE)

## Classification Tree Performance Metrics
accuracy <- mean(test == pred)
precision <- posPredValue(pred, test, positive = "1")
recall <- caret::sensitivity(pred, test, positive = "1")
fmeasure <- (2 * precision * recall)/(precision + recall)
confmat <- caret::confusionMatrix(pred, test, positive = "1")
kappa <- as.numeric(confmat$overall["Kappa"])
auc <- as.numeric(performance(roc.pred, measure = "auc")@y.values)
comparisons <- comparisons %>%
    add_row(approach="Classification Tree", accuracy = accuracy, fmeasure = fmeasure, kappa = kappa, auc = auc) 

## Random Forest
test <- bankTest$Exited
pred <- rf.pred
prob <- rf.pred.prob[,2]

## Random Forest ROC Curve
roc.pred <- prediction(predictions = prob, labels = test)
roc.perf <- performance(roc.pred, measure = "tpr", x.measure = "fpr")
plot(roc.perf, col=4, lwd = 2, add=TRUE)

## Random Forest Performance Metrics
accuracy <- mean(test == pred)
precision <- posPredValue(pred, test, positive = "1")
recall <- caret::sensitivity(pred, test, positive = "1")
fmeasure <- (2 * precision * recall)/(precision + recall)
confmat <- caret::confusionMatrix(pred, test, positive = "1")
kappa <- as.numeric(confmat$overall["Kappa"])
auc <- as.numeric(performance(roc.pred, measure = "auc")@y.values)
comparisons <- comparisons %>%
    add_row(approach="Random Forest", accuracy = accuracy, fmeasure = fmeasure, kappa = kappa, auc = auc) 

## XGBoost
test <- bankTest$Exited
pred <- xgb.pred
prob <- xgb.pred.prob[,2]

# Plot ROC Curve.
roc.pred <- prediction(predictions = prob, labels = test)
roc.perf <- performance(roc.pred, measure = "tpr", x.measure = "fpr")
plot(roc.perf, col=5, lwd = 2, add=TRUE)

# Get performance metrics.
accuracy <- mean(test == pred)
precision <- posPredValue(pred, test, positive = "1")
recall <- caret::sensitivity(pred, test, positive = "1")
fmeasure <- (2 * precision * recall)/(precision + recall)
confmat <- caret::confusionMatrix(pred, test, positive = "1")
kappa <- as.numeric(confmat$overall["Kappa"])
auc <- as.numeric(performance(roc.pred, measure = "auc")@y.values)
comparisons <- comparisons %>%
    add_row(approach="eXtreme Gradient Boosting", accuracy = accuracy, fmeasure = fmeasure, kappa = kappa, auc = auc) 

# Draw ROC legend.
legend(0.6, 0.6, c('Logistic Regression', 'Classification Tree', 'Random Forest', 'eXtreme Gradient Boosting'), 2:5)

knitr::kable(comparisons)


# Formatted table
comparisons %>%
    kbl(caption = "Comparison Table") %>%
    kable_classic(full_width = T, html_font = "Cambria") %>%
    row_spec(3:3, bold = T, color = "white", background = "green") %>%
    footnote(general = "We already know that the response variable is quite imbalanced, so I will not use prediction accuracy as our only metric, instead, I will use several metrics here to select the best model. From the ROC curve and the comparison table, Random Forest achieves a better performance. I’ll go with Random Forest as our final model. ",
    )


# Feature Importance

# First is the feature importance plot of Random Forest model.
vip::vip(rf.mod)

# Second is the feature importance plot of XGBoost model.
vip::vip(xgb.mod)

# Combined
c1<-vip::vip(rf.mod, aesthetics = list(colour="darkgrey", fill="grey"))+ ggtitle("Random Forest")
c2<-vip::vip(xgb.mod, aesthetics = list(colour="black", fill="black"))+ ggtitle("XGBoost")
grid.arrange(c1, c2, ncol = 2)

