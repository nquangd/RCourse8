---
title: "Predicting Weight Lifting Exercise using Data collected by Activity Bands"
author: "Duy Nguyen"
date: "12/8/2018"
output: 
  html_document: 
    keep_md: yes
---



## Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement ??? a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).


This analysis explore the provided dataset and apply machine learning agorithm to predict the type of exercises. The analysis is organised with the following sections:  
(1) Cleaning the dataset  
(2) Feature selection based on different approaches  
(3) Model selection based on cross validation  
(4) Out of sample error estimation  
(5) Predict the provided testset  

## Cleaning the dataset

The dataset is tidy up by the following steps:  
(1) Remove "nearzerovar" variables  
(2) Remove variables that contains only NAs  
(3) Remove time/date related variables  
(4) Remove "window num" variable which was used to calculate the values of variables  
(5) Split the training set to cross training and cross test set to be used for cross validation  



```r
set.seed(1000)

url1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

if (!file.exists("pml-training.csv")) {download.file(url1,destfile = "pml-training.csv", method = "curl")}
if (!file.exists("pml-testing.csv")) {download.file(url2,destfile = "pml-testing.csv", method = "curl")}

rawtrain <- read.csv("pml-training.csv")
rawtest <- read.csv("pml-testing.csv")
# Train set

nearzero <- nearZeroVar(rawtrain, saveMetrics = F)
train1 <- rawtrain[,-nearzero]
# Remove all columns that is all NAs
naindex <- apply(train1, 2, function(x) sum(is.na(x)) )
train2 <- train1[,naindex==0]

# Test set

nearzero <- nearZeroVar(rawtest, saveMetrics = F)
test1 <- rawtest[,-nearzero]
# Remove all columns that is all NAs
naindex <- apply(test1, 2, function(x) sum(is.na(x)) )
test2 <- test1[,naindex==0]
# Check if the variables name of the test and training are equal
nn <- which((names(train2)==names(test2))==FALSE)
names(train2)[nn]
```

```
## [1] "classe"
```

```r
names(test2)[nn]
```

```
## [1] "problem_id"
```

```r
# Remove the id and username variables. From the paper, the window is for features calculation to generate the dataset, it is not relevant for the current analysis
train2 <- train2[,-c(1:6)]
test2 <- test2[,-c(1:6)]
# Split the train set to train and validation for cross validation test

subindex <- createDataPartition(train2$classe, p = 0.7, list = F) 
crosstrain <- train2[subindex,]
crosstest <- train2[-subindex,]
dim(crosstrain)
```

```
## [1] 13737    53
```

## Feature selection
Since the author is not familiar with this particular field, it is not possible to choose the most important features based on scientific backgrounds of the physics of body movement. Therefore, features are choosen based on different machine learning algorithms. In particular, three method are considered:  
(1) Using Best-First-Search with backtracks algorithm, this is based on Correlation based feature selection approach.  
(2) Using decision tree model to evaluate the relative importance of features  
(3) Using all available features for building the model  


```r
## Use Correlation based feature selection with backwards
set.seed(1000)
evaluator <- function(subset) {
    #k-fold cross validation
    k <- 5
    splits <- runif(nrow(crosstrain))
    results = sapply(1:k, function(i) {
      test.idx <- (splits >= (i - 1) / k) & (splits < i / k)
      train.idx <- !test.idx
      test <- crosstrain[test.idx, , drop=FALSE]
      train <- crosstrain[train.idx, , drop=FALSE]
      tree <- rpart(as.simple.formula(subset, "classe"), train)
      error.rate = sum(test$classe != predict(tree, test, type="c")) / nrow(test)
      return(1 - error.rate)
    })
    #print(subset)
    #print(mean(results))
    return(mean(results))
}

subset <- best.first.search(names(crosstrain)[-53], evaluator)
fs1 <- as.simple.formula(subset, "classe")
print(fs1)
```

```
## classe ~ pitch_belt + yaw_belt + yaw_arm + gyros_arm_x + gyros_dumbbell_z + 
##     accel_dumbbell_z + magnet_dumbbell_y + magnet_dumbbell_z + 
##     roll_forearm + pitch_forearm + gyros_forearm_x
## <environment: 0x7fb433067888>
```

```r
var1 <- as.data.frame(matrix(c("Method 1", paste(subset, collapse = ",")),nrow = 1, ncol = 2))
### Use decision tree to see which variables are importance

    #k-fold cross validation
    k <- 5
    splits <- runif(nrow(crosstrain))
    results <-  sapply(1:k, function(i) {
      test.idx <- (splits >= (i - 1) / k) & (splits < i / k)
      train.idx <- !test.idx
      test <- crosstrain[test.idx, , drop=FALSE]
      train <- crosstrain[train.idx, , drop=FALSE]
      tree <- rpart(classe~., data = train)
      v <- tree$variable.importance
      return(v[1:20])
    })
 fs2 <- as.simple.formula(rownames(results),"classe") 
 print(fs2)
```

```
## classe ~ roll_belt + pitch_belt + accel_belt_z + pitch_forearm + 
##     total_accel_belt + yaw_belt + accel_dumbbell_y + magnet_dumbbell_z + 
##     total_accel_dumbbell + magnet_dumbbell_y + roll_dumbbell + 
##     magnet_belt_z + magnet_dumbbell_x + accel_dumbbell_z + accel_dumbbell_x + 
##     magnet_belt_x + accel_forearm_x + accel_belt_x + magnet_belt_y + 
##     accel_belt_y
## <environment: 0x7fb431dd32d0>
```

```r
var2 <- as.data.frame(matrix(c("Method 2", paste(rownames(results), collapse = ",")),nrow = 1, ncol = 2))
#kable_styling(rbind(var1,var2))
```

The selected features for the first approaches contain **11** features. In the second approach, the first **20** features ranked according to importance were choosen. The selected features can be seen on the printed formula. They will further be used to evaluate different models based on cross validation


## Model selection based on cross validation
The generalised linear regression (multinominal) migh not be a good choice for multilevel categorical outcome as in this particular problem. Therefore, decision tree and random forest are evaluated with three different subsets of features derived from the previous section.   

### Decision tree

The decision tree model with three different sets of features were fitted and tested on the the cross-validation test set. The accuracy of the predictions with three subsets of predictors is shown in the following table. It can be seen that the Accuracy is fairly the same for all three models despite the great difference in the number of predictors. The approach 1, which is based on correlation between features, contains the smallest number of predictors, i.e. **11**, gives better accuracy than the approach 2 where 20 predictors were choosen. When all predictors are used, the model could predict a little bit better.   
Since the approach 1 with **11** features gives the accuracy at the same level as the full features (52) model, it is choosen to go to further step.



```r
tree1 <- rpart(fs1, data = crosstrain)
tree2 <- rpart(fs2, data = crosstrain)
tree3 <- rpart(classe~., data = crosstrain)
cfm1 <- confusionMatrix(predict(tree1, crosstest, type = "c"),crosstest$classe)
cfm2 <- confusionMatrix(predict(tree2, crosstest, type = "c"),crosstest$classe)
cfm3 <- confusionMatrix(predict(tree3, crosstest, type = "c"),crosstest$classe)
Acc <- c(cfm1$overall["Accuracy"],cfm2$overall["Accuracy"],cfm3$overall["Accuracy"])
names(Acc) <- c("Feature 1", "Feature 2", "All features")
kable_styling(kable(Acc,col.names=c("Accuracy")), full_width = F)
```

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> Accuracy </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> Feature 1 </td>
   <td style="text-align:right;"> 0.7566695 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Feature 2 </td>
   <td style="text-align:right;"> 0.7440952 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> All features </td>
   <td style="text-align:right;"> 0.7653356 </td>
  </tr>
</tbody>
</table>


### Random forest

The random forest model with three different sets of features were fitted and tested on the the cross-validation test set. The accuracy of the predictions with three subsets of predictors is shown in the following table. The accuracy table shows the same trend as that was observed in the decision tree model. That is, the approach 1, based on correlation, gives the accuracy at the same level while requiring only **11** predictors. 

Comparing the two models, i.e. decision tree and random forest, it is obvious that the random forest gives much higher accuracy on the same setting. Therefore, the random forest will be choosen to predict the actual test set



```r
rf1 <- randomForest(fs1, data = crosstrain)
rf2 <- randomForest(fs2, data = crosstrain)
rf3 <- randomForest(classe~., data = crosstrain)
cfm1 <- confusionMatrix(predict(rf1, crosstest, type = "c"),crosstest$classe)
cfm2 <- confusionMatrix(predict(rf2, crosstest, type = "c"),crosstest$classe)
cfm3 <- confusionMatrix(predict(rf3, crosstest, type = "c"),crosstest$classe)
Acc <- c(cfm1$overall["Accuracy"],cfm2$overall["Accuracy"],cfm3$overall["Accuracy"])
names(Acc) <- c("Feature 1", "Feature 2", "All features")
kable_styling(kable(Acc,col.names=c("Accuracy")), full_width = F)
```

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> Accuracy </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> Feature 1 </td>
   <td style="text-align:right;"> 0.9887850 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Feature 2 </td>
   <td style="text-align:right;"> 0.9828377 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> All features </td>
   <td style="text-align:right;"> 0.9949023 </td>
  </tr>
</tbody>
</table>

## Out of samples error

The out of samples error was determined by (1-Accuracy) where the accuracy was estimated from the cross validation test. The error coressponds to the ratio of the number of misclassified case to the total number of observations. In this analysis, the out of samples error for the choosen model, i.e. random forest with 11 features, is **0.011**  


## Predict the test set

The predicted value for the testset using the **random forest with 11 selected predictors** is printed as following, the result will be check via the quiz.



```r
finalpredict <- predict(rf1, test2, type = "c")
print(finalpredict)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  B  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
