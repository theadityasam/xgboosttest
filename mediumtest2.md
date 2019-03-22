# Medium Test 2  

std::cout was added at line 317 of the file regression_obj.cu  
## Screenshot
![img](https://github.com/theadityasam/xgboosttest/blob/master/images/mediumtest2.png)
  
The logging was tested on ovarian cancer dataset:
```
rm(list = ls())
library("xgboost")
library(data.table)
library(Matrix)
library(survival)
library(glmnet)
data("ovarian")
X <- cbind(ovarian$ecog.ps, ovarian$rx)
Y <- ifelse(ovarian$fustat == 1, ovarian$futime, ovarian$futime*-1) #Assigning negative values if deceased
param <- list(  objective   = "survival:cox", #Setting the objective parameter to count:poisson
                eta         = 1,
                max.depth   = 3,
                nthread     = 2
)
bstSparse <- xgb.cv(data = X, label = Y, nrounds = 7, params = param, nfold = 5, showsd = F, prediction = T)
pred <- round(bstSparse$pred)
mae <- (sum(abs(pred - ovarian$fustat)))/length(ovarian$fustat)
mae
```
## Console Output
```
(Many iteration outputs)
.
.
.
Iteration:23    Prediction:-0.474999    True Label:-1227
Iteration:1    Prediction:-0.236025    True Label:59
Iteration:2    Prediction:-0.236025    True Label:115
Iteration:3    Prediction:-0.380314    True Label:156
Iteration:4    Prediction:-0.380314    True Label:268
Iteration:5    Prediction:-0.302555    True Label:353
Iteration:6    Prediction:-2.30749    True Label:-377
Iteration:7    Prediction:-2.30749    True Label:-421
Iteration:8    Prediction:-0.236025    True Label:431
Iteration:9    Prediction:-0.380314    True Label:-448
Iteration:10    Prediction:-0.302555    True Label:464
Iteration:11    Prediction:-0.302555    True Label:475
Iteration:12    Prediction:-0.236025    True Label:-477
Iteration:13    Prediction:-0.302555    True Label:563
Iteration:14    Prediction:-0.380314    True Label:638
Iteration:15    Prediction:-2.30749    True Label:-744
Iteration:16    Prediction:-0.302555    True Label:-769
Iteration:17    Prediction:-2.30749    True Label:-770
Iteration:18    Prediction:-0.236025    True Label:-803
Iteration:19    Prediction:-0.380314    True Label:-855
Iteration:20    Prediction:-0.380314    True Label:-1040
Iteration:21    Prediction:-0.236025    True Label:-1106
Iteration:22    Prediction:-2.30749    True Label:-1206
Iteration:23    Prediction:-0.302555    True Label:-1227
> pred <- round(bstSparse$pred)
> mae <- (sum(abs(pred - ovarian$fustat)))/length(ovarian$fustat)
> mae
[1] 0.3846154 
```
