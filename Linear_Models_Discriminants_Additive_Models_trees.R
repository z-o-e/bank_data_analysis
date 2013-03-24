#**************************************************************************************
#
#  		Linear Models, Discriminants,  Additive Models and trees
#
#**************************************************************************************

# Code for evaluation 
source("ROC.R")

# Training & Test data sets
bank_train<-read.table("bank-full.csv",sep=";",head=TRUE)
bank_test<-read.table("bank.csv",sep=";",head=TRUE)
summary(bank_train)


# Comparing quantitative variables for 2 data sets
par(mfrow=c(3,3))
for(i in c(1,6,10,12:15)){
  boxplot(bank_train[,i],at=1,xaxt="n",xlim=c(0,3),main=colnames(bank_train)[i])
  boxplot(bank_test[,i],at=2,xat="n",add=TRUE)
  axis(1,at=1:2,labels=c("Original/Train","Test"),tick=TRUE)
} 




# For discriminant analysis we need to remove the categorical predictors: 
# 2-job, 3-marital, 4-education, 5-default, 7-housing, 8-loan, 9-contact, 11-month, 16-outcome
bank_trainDiscriminants<-bank_train[,-c(2,3,4,5,7,8,9,11,16)]
bank_testDiscriminants<-bank_test[,-c(2,3,4,5,7,8,9,11,16)]



#**************************************************************************************
#
#  		Linear Discriminants
#
#**************************************************************************************

library(MASS)

# Linear discriminant regression
train_lda <- lda(y~., data = bank_trainDiscriminants)


# Training set fit
fit_lda <- predict(train_lda)
# Confusion matrix for training set (uses ROC code)
score.table(fit_lda$posterior[,2], bank_trainDiscriminants$y)
# Plot of training set performance
par(mfrow=c(1,1))
plot(fit_lda$posterior[,2], bank_trainDiscriminants$y, pch = 21, main = "LDA Results for Bank Data (train)", xlab = "Posterior", ylab = "Actual" )


# Test set fit
pred_lda <- predict(train_lda, newdata = bank_testDiscriminants)
# Confusion matrix for test set
score.table(pred_lda$posterior[,2], bank_testDiscriminants$y)
# Plot of test set performance
par(mfrow=c(1,1))
plot(pred_lda$posterior[,2], bank_testDiscriminants$y, pch = 21, main = "LDA Results for Bank Data (test)", xlab = "Posterior", ylab = "Actual" )




#**************************************************************************************
#
#    	Quadratic Discriminants
#
#**************************************************************************************

# Quadratic discriminant regression
train_qda <- qda(y~., data = bank_trainDiscriminants)

# Training set fit
fit_qda <- predict(train_qda)
# Confusion matrix for training set 
score.table(fit_qda$posterior[,2], bank_trainDiscriminants$y)
# Plot of training set performance
plot(fit_qda$posterior[,2], bank_trainDiscriminants$y, pch = 21, main = "QDA Results for Bank Data (train)", xlab = "Posterior", ylab = "Actual" )

# Test set fit
pred_qda <- predict(train_qda, newdata = bank_testDiscriminants)
# Confusion matrix for training set
score.table(pred_qda$posterior[,2], bank_testDiscriminants$y)
# Plot of training set performance
plot(pred_qda$posterior[,2], bank_testDiscriminants$y, pch = 21, main = "QDA Results for Bank Data (test)", xlab = "Posterior", ylab = "Actual" )




#**************************************************************************************
#
#      Mixture Model Discriminants
#
#**************************************************************************************

library(class)
library("mda",lib.loc="/Users/huiyingzhang/Downloads")

# mixed model discriminants regression
train_mda <- mda(y~., data = bank_trainDiscriminants)


# Training set fit
fit_mda <- predict(train_mda,bank_trainDiscriminants,type = "post")
# Confusion matrix for training set
score.table(fit_mda[,2], bank_trainDiscriminants$y)
# Plot of training set performance
par(mfrow=c(1,1))
plot(fit_mda[,2], bank_trainDiscriminants$y, pch = 21, main = "MDA Results for Bank Data (train)", xlab = "Posterior", ylab = "Actual" )


# Test set fit
pred_mda <- predict(train_mda,newdata = bank_testDiscriminants,type = "post")
# Confusion matrix for test set
score.table(pred_mda[,2], bank_testDiscriminants$y)
# Plot of test set performance
plot(pred_mda[,2], bank_testDiscriminants$y, pch = 21, main = "MDA Results for Bank Data (test)", xlab = "Posterior", ylab = "Actual" )




#**************************************************************************************
#
#    	GLM - Logistic Regression
#
#**************************************************************************************

# For logistic regression we can use the both quantitative and qualitative predictors
train_glm <- glm(y~., data = bank_train, family = "binomial")


# Training set fit
fit_glm <- predict(train_glm, type = "response")
# Confusion matrix for training set
score.table(fit_glm, bank_train$y)
# Plot of training set performance
plot(fit_glm, bank_train$y, pch = 21, main = "GLM Results for Bank Data (train)", xlab = "Posterior", ylab = "Actual" )


# Test set fit
pred_glm<- predict(train_glm, newdata = bank_test, type = "response")
# Confusion matrix for training set
score.table(pred_glm, bank_test$y)
# Plot of training set performance
plot(pred_glm, bank_test$y, pch = 21, main = "GLM Results for Bank Data (test)", xlab = "Posterior", ylab = "Actual" )




#****************************************************************************************
#
#    GAM - Generalized Additive Models
#
#****************************************************************************************

library("gam",lib.loc="/Users/huiyingzhang/Downloads")


#*****************************************************
# GAM - Generalized Additive Model with all variables
#*****************************************************

# GAM on all quantitative and qualitative variables
train_gam<-gam(y~.,data=bank_train,family="binomial")
summary(train_gam)


# Training set fit
fit_gam <- predict(train_gam, type = "response")
# Confusion matrix for training set
score.table(fit_gam, bank_train$y)
# Plot of training set performance
plot(fit_gam, bank_train$y, pch = 21, main = "GAM (witl all variables) Results for Bank Data (train)", xlab = "Posterior", ylab = "Actual" )


# Test set fit
pred_gam <- predict(train_gam, newdata = bank_test, type = "response")
# Confusion matrix for training set
score.table(pred_gam, bank_test$y)
# Plot of training set performance
plot(pred_gam, bank_test$y, pch = 21, main = "GAM (witl all variables) Results for Bank Data (test)", xlab = "Posterior", ylab = "Actual" )


#*****************************************************
# GAM - Generalized Additive Model step-wise
#*****************************************************

train_gamStep <- step.gam(train_gam, scope=list("age"=~1+ age +s(age),"job"=~1+job ,"marital"=~1+marital,
                                                "education"=~1+ education,"default"=~1+ default,"balance"=~1+balance +s(balance),
                                                "housing"=~1+housing ,"loan"=~1+loan ,"contact"=~1+contact ,
                                                "day"=~1+day +s(day),"month"=~1+month ,"duration"=~1+duration +s(duration),
                                                "campaign "=~1+ campaign +s(campaign ),"pdays"=~1+pdays +s(pdays),"previous"=~1+previous +s(previous),
                                                "poutcome"=~1+ poutcome))
summary(train_gamStep)
# Training set
fit_gamStep <- predict(train_gamStep, type = "response")
# Confusion matrix for training set
score.table(fit_gamStep, bank_train$y)
# Plot of training set performance
plot(fit_gamStep, bank_train$y, pch = 21, main = "GAM Stepwise Results for Bank Data (train)", xlab = "Posterior", ylab = "Actual" )

# Test set 
pred_gamStep <- predict(train_gamStep, newdata=bank_test, type = "response")
# Confusion matrix for training set
score.table(pred_gamStep, bank_test$y)
# Plot of training set performance
plot(pred_gamStep, bank_test$y, pch = 21, main = "GAM Stepwise Results for Bank Data (test)", xlab = "Posterior", ylab = "Actual" )




#*********************************************************
#
#    	Trees
#
#*********************************************************

#library("tree",lib.loc="/Users/huiyingzhang/Downloads")

# Gini
#train_treeGini <- tree(y~., data = bank_train, split = "gini")
#summary(train_treeGini)

# Entropy
#train_treeEntropy <- tree(y~., data = bank_train,split = "deviance")
#summary(train_treeEntropy)


#plot the trees
#plot(train_treeGini,main="Gini Tree")
#text(train_treeGini)
#par(mfrow = c(1,1))
#plot(train_treeEntropy,main="Entropy Tree")
#text(train_treeEntropy)

# entropy tree misclassification rate  #4 is good enough
#plot(cv.tree(train_treeEntropy,FUN=prune.tree,method="misclass"),main="entropy tree missclassification rate")
#default is impurity in terms of deviance/cross-entropy  # or 7 is the best
#plot(cv.tree(train_treeEntropy),main="Entropy tree CV error")

# gini tree misclassification rate  
#plot(cv.tree(train_treeGini,,FUN=prune.tree,method="misclass")) #no leave is the best?
#default is impurity in terms of deviance/cross-entropy  
#plot(cv.tree(train_treeGini,,FUN=prune.tree),main="Gini tree CV error ")


#prune the entropy tree (leaves=4)
#pruneTreeEntropy1<-prune.tree(train_treeEntropy,best=4)
#plot(pruneTreeEntropy1,main="pruned entropy tree(leaves=4)")
#text(pruneTreeEntropy1)

#prune the entropy tree (leaves=7)
#pruneTreeEntropy2<-prune.tree(train_treeEntropy,best=7)
#plot(pruneTreeEntropy2,,main="pruned entropy tree(leaves=7)")
#text(pruneTreeEntropy2)

#prediction on test set(Entropy leaves=4)
#pred_treeEntropy_L4<-predict(pruneTreeEntropy1,newdata=bank_test,type="class")
#table(pred_treeEntropy_L4,bank_test$y)

#prediction on test set(Entropy leaves=7)
#pred_treeEntropy_L7<-predict(pruneTreeEntropy2,newdata=bank_test,type="class")
#table(pred_treeEntropy_L7,bank_test$y)



#prune the gini tree (leaves=4)
#pruneTreeGini<-prune.tree(train_treeGini,best=4)
#plot(pruneTreeGini,main="pruned gini tree(leaf?)")
#text(pruneTreeGini)


#prediction on test set(Entropy leaves=4)
#pred_treeGini_best<-predict(pruneTreeGini,newdata=bank_test,type="class")
#table(pred_treeGini_best,bank_test$y)


#*********************************************************
#
#    	Trees using rpart
#
#*********************************************************

library("rpart",lib.loc="/Users/huiyingzhang/Downloads")




# Rpart tree
# Gini
train_rpart <- rpart(y~., data = bank_train, method = "class")
# Entropy
train_rparti <- rpart(y~., data = bank_train,method = "class", parms = list(split = "information"))

# Summary
summary(train_rpart)
summary(train_rparti)

# Plots
par(mar=c(0.5, 2, 2, 0.5) + 0.1)
plot(train_rpart,main="full tree") # gini tree
text(train_rpart,use.n=TRUE,pretty=3)
plot(train_rparti,main="full tree") # entropy tree
text(train_rparti,use.n=TRUE,pretty=3)

# Plot cross-validation error
plotcp(train_rpart)
plotcp(train_rparti)


# Selecting the best tree by cv error
best.rpart <- function(rpart.obj)
{
  cp <- rpart.obj$cptable[which.min(rpart.obj$cptable[,4]),1]
  prune.rpart(rpart.obj, cp = cp)
}


train_rpart_best<- best.rpart(train_rpart) 
plot(train_rpart_best,main="pruned Gini tree train")
text(train_rpart_best, use.n=TRUE)

train_rparti_best<- best.rpart(train_rparti) 
plot(train_rparti_best,main="pruned Entropy tree train")
text(train_rparti_best, use.n=TRUE)

# predict tree results
rpart_pred <- predict(train_rpart_best, newdata = bank_test, type = "prob")
rparti_pred <- predict(train_rparti_best, newdata = bank_test, type = "prob")
summary(rpart_pred)
summary(rparti_pred)

# misclassification rate
score.table(rpart_pred[,2], bank_test$y)
score.table(rparti_pred[,2], bank_test$y)

# Just trees
plot.roc(rpart_pred[,2], bank_test$y)
lines.roc(rparti_pred[,2], bank_test$y, col = "red2")
legend(.5, .25, legend = c("gini tree", "entropy tree"), lwd = 2, col = c("blue", "red2"))


#*********************************************************
#
#  		ROC 
#
#*********************************************************


# Just Discriminants
plot.roc(pred_lda$post[,2], bank_test$y)
lines.roc(pred_qda$post[,2], bank_test$y, col = "red2")
lines.roc(pred_mda[,2], bank_test$y, col = "green")
legend(.7, .3, legend = c("LDA", "QDA", "MDA"), lwd = 2, col = c("blue", "red2", "green"))


# GLM and GAM
plot.roc(pred_glm, bank_test$y)
lines.roc(pred_gam, bank_test$y, col = "green")
lines.roc(pred_gamStep,bank_test$y, col = "purple")
legend(.7, .3, legend = c("GLM",  "GAM", "GAM Step"), lwd = 2, col = c("blue",  "green", "purple"))
