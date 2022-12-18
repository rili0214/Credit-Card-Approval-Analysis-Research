rm(list = ls())

library(dplyr)
library(caret)
library("ggplot2")
library(tidyverse)

#read the data
rare_data <- read.table("4280data.data", fileEncoding = "UTF-8", sep = ",")

#checking the missing data, replace all "?" with NA 
df <- rare_data %>% 
  mutate_all(na_if, "?")
sapply(df, function(x) sum(is.na(x)))

#handle the missing data, replace NA in Age with mean
df$V2 = as.numeric(df$V2)
df$V2 = ifelse(is.na(df$V2),ave(df$V2, FUN = function (x)mean(x, na.rm = TRUE)),
        df$V2)

#remove the rest of the missing values
df<-na.omit(df)

#obtain each attribute from the set
gender <- df$V1
age <- df$V2;
debt <- df$V3;
marital <- df$V4;
bankcus <- df$V5;
edu <- df$V6;
ethnicity <- df$V7;
y_emp <- df$V8;
prior <- df$V9;
emp <- df$V10;
credit <- df$V11;
driver <- df$V12;
citizen <- df$V13;
zip <- df$V14;
income <- df$V15;
result = df$V16

#normalize data
process <- preProcess(as.data.frame(age), method=c("range"))
df$V2 <- predict(process, as.data.frame(age))
process <- preProcess(as.data.frame(debt), method=c("range"))
df$V3 <- predict(process, as.data.frame(debt))
process <- preProcess(as.data.frame(y_emp), method=c("range"))
df$V8 <- predict(process, as.data.frame(y_emp))
process <- preProcess(as.data.frame(credit), method=c("range"))
df$V11 <- predict(process, as.data.frame(credit))
process <- preProcess(as.data.frame(income), method=c("range"))
df$V15 <- predict(process, as.data.frame(income))

#data visualization for continuous variables
frequency <- function(x) {
  factor(x, levels = names(table(x)))
}
ggplot(df, aes(x = frequency(`V2`))) + geom_bar()
ggplot(df, aes(x = frequency(`V3`))) + geom_bar()
ggplot(df, aes(x = frequency(`V8`))) + geom_bar()
ggplot(df, aes(x = frequency(`V11`))) + geom_bar()
ggplot(df, aes(x = frequency(`V15`))) + geom_bar()

#data visualization for discrete variables
df$V1<-ifelse(df$V1=="a",1,0)
df$V9<-ifelse(df$V9=="t",1,0)
df$V10<-ifelse(df$V10=="t",1,0)
df$V12<-ifelse(df$V12=="t",1,0)

df$V4 <- with(df, ifelse(V4 == "u", 1, ifelse(V4 == "y", 2, ifelse(V4 == "l", 3, ifelse(V4 == "6", 4, V4)))))
df$V5 <- with(df, ifelse(V5 == "g", 1, ifelse(V5 == "p", 2, ifelse(V5 == "gg", 3, V5))))
df$V6 <- with(df, ifelse(V6 == "c", 1, ifelse(V6 == "d", 2, ifelse(V6 == "cc", 3, ifelse(V6 == "i", 4, ifelse(V6 == "j", 5, ifelse(V6 == "k", 6, ifelse(V6 == "m", 7, V6))))))))
df$V6 <- with(df, ifelse(V6 == "r", 8, ifelse(V6 == "q", 9, ifelse(V6 == "w", 10, ifelse(V6 == "x", 11, ifelse(V6 == "e", 12, ifelse(V6 == "aa", 13, ifelse(V6 == "ff", 14, V6))))))))
df$V7 <- with(df, ifelse(V7 == "v", 1, ifelse(V7 == "h", 2, ifelse(V7 == "bb", 3, ifelse(V7 == "j", 4, ifelse(V7 == "n", 5, ifelse(V7 == "z", 6, ifelse(V7 == "dd", 7, ifelse(V7 == "ff", 8, ifelse(V7 == "o", 9, V7))))))))))
df$V13 <- with(df, ifelse(V13 == "g", 1, ifelse(V13 == "p", 2, ifelse(V13 == "s", 3, V13))))
df$V16 <- with(df, ifelse(V16 == "+", 1, ifelse(V16 == "-", 0, V16)))

df$V4 <- as.numeric(df$V4)
df$V5 <- as.numeric(df$V5)
df$V6 <- as.numeric(df$V6)
df$V7 <- as.numeric(df$V7)
df$V13 <- as.numeric(df$V13)
df$V14 <- as.numeric(df$V14)
df$V16 <- as.numeric(df$V16)
str(df)
pairs(df)

#split the data into train and test set
sample <- sample(c(TRUE, FALSE), nrow(df), replace=TRUE, prob=c(0.7,0.3))
train <- df[sample, ]
test <- df[!sample, ]  

#logistic
model <- glm(V16 ~ ., family="binomial"(link="logit"), data=train)
library(caret)
pdata <- predict(model, newdata = train, type = "response")
confusionMatrix(data = as.numeric(pdata>0.5), reference = train$V16)
summary(logitMod)
library(pscl)
pscl::pR2(logitMod)["McFadden"]
caret::varImp(logitMod)
library(pROC)
roc_score=roc(train$V16, pdata) #AUC score
plot(roc_score ,main ="ROC curve -- Logistic Regression ")

#LDA
set.seed(1)
row.number = sample(1:nrow(df), 0.6*nrow(df))
train = df[row.number,]
test = df[-row.number,]
dim(train)
dim(test)

library(MASS)
lda.model = lda (V16~., data=train)
lda.model
predmodel.train.lda = predict(lda.model, data=train)
confusionMatrix(data = as.factor(predmodel.train.lda$class), reference = as.factor(train$V16))
par(mar = c(1, 1, 1, 1))
ldahist(predmodel.train.lda$x[,1], g= predmodel.train.lda$class)

predmodel.test.lda = predict(lda.model, newdata=test)
confusionMatrix(data = as.factor(predmodel.test.lda$class), reference = as.factor(test$V16))
par(mfrow=c(1,1))
plot(predmodel.test.lda$x[,1], predmodel.test.lda$class, col=test$V16+10)

#QDA
qda.model = qda (V16~ V2 + V3 + V8 + V11 + V15, data=train)
qda.model
predmodel.train.qda = predict(qda.model, data=train)
confusionMatrix(data = as.factor(predmodel.train.qda$class), reference = as.factor(train$V16))

predmodel.test.qda = predict(qda.model, newdata=test)
confusionMatrix(data = as.factor(predmodel.test.qda$class), reference = as.factor(test$V16))
par(mfrow=c(1,1))
plot(predmodel.test.qda$posterior[,2], predmodel.test.qda$class, col=test$V16+10)

#KNN, K = 1~30
p <- c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13", "V14","V15")
y <- "V16"
library(class)
r <- data.frame(array(NA, dim = c(0, 2), dimnames = list(NULL, c("k","accuracy"))))
for (k in 1:30) {
  set.seed(60402) 
  predictions <- knn(train = train[,p],
                     test = test[,p],
                     cl = train[,y],
                     k = k)
  t <- table(pred = predictions, ref = test[,y])
  a <- sum(diag(t)) / sum(t)
  r <- rbind(r, data.frame(k = k, accuracy = a))
}
# find best k
r[which.max(r$accuracy),]
(k.best <- r[which.max(r$accuracy),"k"])
# plot
with(r, plot(k, accuracy, type = "l", xlab="X Label", ylab="Y Label"))
abline(v = k.best, lty = 2)

#ridge regression
y <- df$V16
x <- data.matrix(df[, c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13", "V14","V15")])
library(glmnet)
model <- glmnet(x, y, alpha = 0)
summary(model)

cv_model <- cv.glmnet(x, y, alpha = 0)
best_lambda <- cv_model$lambda.min#find optimal lambda value that minimizes test MSE
best_lambda
plot(cv_model) 

best_model <- glmnet(x, y, alpha = 0, lambda = best_lambda)
coef(best_model)

plot(model, xvar = "lambda")#produce Ridge trace plot

#find R-Squared
y_predicted <- predict(model, s = best_lambda, newx = x)
sst <- sum((y - mean(y))^2)
sse <- sum((y_predicted - y)^2)
rsq <- 1 - sse/sst
rsq

#Lasso model
y <- df$V16
x <- data.matrix(df[, c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13", "V14","V15")])
library(glmnet)
cv_model <- cv.glmnet(x, y, alpha = 1)
best_lambda <- cv_model$lambda.min#find optimal lambda value that minimizes test MSE
best_lambda

plot(cv_model) #produce plot of test MSE by lambda value

best_model <- glmnet(x, y, alpha = 1, lambda = best_lambda)#find coefficients of best model
coef(best_model)

y_predicted <- predict(best_model, s = best_lambda, newx = x)#find R-Squared
sst <- sum((y - mean(y))^2)
sse <- sum((y_predicted - y)^2)
rsq <- 1 - sse/sst
rsq

#PCR
library(pls)
set.seed(1)
model <- pcr(V16~., data=df, scale=TRUE, validation="CV")#fit PCR model
summary(model)

par(mfrow=c(1,3))
validationplot(model)#visualize cross-validation plots
validationplot(model, val.type="MSEP")
validationplot(model, val.type="R2")

pcr_pred <- predict(model, test, ncomp=15)
sqrt(mean((pcr_pred - test$V16)^2))#calculate RMSE

#PLS model
set.seed(1)
model <- plsr(V16~., data=df, scale=TRUE, validation="CV")
summary(model)

par(mfrow=c(1,3))
validationplot(model)
validationplot(model, val.type="MSEP")
validationplot(model, val.type="R2")

pls_pred <- predict(model, test, ncomp=15)
sqrt(mean((pls_pred - test$V16)^2))#calculate RMSE
