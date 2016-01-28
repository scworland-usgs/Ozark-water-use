
library(ggplot2); library(reshape2)

# Set local working directory
setwd("C:/Users/scworlan/Documents/MAP/OzarkWaterUse/Ozark-water-use")

# Load raw data (from B. Clarke 12/23/2015)
raw.data <- read.csv("WU_exportPopPrecip2.csv", na.strings="-9999")

# Remove locations without mgd values (or zeros) for building the model
raw.data$mgd[raw.data$mgd==0] <- NA
d <- raw.data[complete.cases(raw.data),]

# EXPLORATORY ANALYSIS----
## Histogram of mgd values
hist(d$mgd,1000, xlim=c(0,1),main="Histogram of MGD",xlab=NULL)

## subset for pairs plot 
d2 <- d[,c(9,2,6,8,10,11,14,15)]
d2$mgd <- log10(d2$mgd)
d2$divs <- as.numeric(d2$divs)

## pairs plot (require custom pairs function) !! takes a long time to run
pairs(d2, lower.panel = panel.lm, upper.panel = panel.cor)

## Boxplot of depth and divisions
dm = melt(d[,c(2,8)])

ggplot(data=dm) + geom_boxplot(aes(divs,value), fill = "grey") + ylim(c(0,3000)) +
  ylab("Depth (ft)") + theme_bw(base_size=18) + xlab(NULL)

# STATISTICAL MODELS----
library(rpart); library(rpart.plot); library(randomForest);

## Subset mgd and predictors for building models
dsub <- d[,c(9,2,8,14,15)]
row.names(dsub) <- NULL

## partition data into 70% training and 30% testing set
set.seed(5)
x <- runif(0.7*nrow(dsub), 1, nrow(dsub))
train <- dsub[x,]
test <- dsub[-x,]

## Build the NULL model
null.means <- aggregate(train$mgd, list(train$divs), mean)
colnames(null.means) <- c("divs","null.mean")

## build regression tree
set.seed(5)
t1 <- rpart(mgd ~., data = train, control=rpart.control(minsplit=5000, cp=0))
t1.predict <- predict(t1,test)

## scale the predictors
dsub[,3:5] <- scale(dsub[,3:5])

## build RF model
set.seed(11)
rf1 <- randomForest(mgd ~., data = train, importance = TRUE, ntree = 100)
rf1.predict <- predict(rf1,test)

### Variable importance plot
varImpPlot(rf1, type=1, main = "Importance of Variables in 100 regression trees",pch=21)

## build knn model
library(kknn)
knn.train <- train.kknn(mgd ~., data = train,  kmax = 25, 
                        kernel = c("rectangular", "triangular", "biweight", "gaussian", "triweight", "inv"));
plot(knn.train)

knn <- kknn(mgd ~., train = train, test = test, kernel = "inv", k = 25)

## Build boosted regression trees
library(gbm); library(doParallel)

fitControl <- trainControl(method = "cv", number = 10, allowParallel=TRUE)

gbmGrid <-  expand.grid(.interaction.depth = seq(15,30,by=5),
                        .n.trees = c(1000,5000,7000,9000,11000),
                        .shrinkage = c(0.1),
                        .n.minobsinnode=100)

cl <- makeCluster(detectCores())
registerDoParallel(cl)

# Start the clock
ptm <- proc.time()

set.seed(100)
gbmFit2 <- caret::train(mgd ~., data = train,
                 method = "gbm",
                 verbose = FALSE,
                 trControl = fitControl,
                 tuneGrid = gbmGrid)

stopCluster(cl)

# Stop the clock
proc.time() - ptm

gbm1 <- gbmFit2$results[complete.cases(gbmFit2$results),]

ggplot(gbm1) + geom_line(aes(n.trees,RMSE,color=factor(interaction.depth))) +
  geom_point(aes(n.trees,RMSE,color=factor(interaction.depth),shape=factor(interaction.depth))) +
  facet_wrap(~shrinkage) + theme_bw(base_size=18) + xlab("Boosting Iterations") + 
  labs(color="Max tree depth",shape="Max tree depth")

ggplot(gbmFit2) + theme_bw(base_size=18)

gbm.predict <- predict(gbmFit2, test)


## assemble predictions
predictions <- data.frame(test$mgd,knn$fitted.values,t1.predict,rf1.predict, gbm.predict)
row.names(predictions) = NULL
colnames(predictions) = c("actual","knn","reg.tree","random.forest","gbm")
#predictions <- round(predictions,5)
predictions$divs <- test$divs
predictions <- predictions[-which(predictions$divs==""),]
predictions <- merge(null.means,predictions, by = "divs")
predictions <- predictions[,c(3,2,4:7,1)]

## Calculate the RMSE and MPE
library(hydroGOF)

null.rmse <- rmse(predictions$null.mean,predictions$actual)
#null.rmse <- (mean((predictions$null - predictions$actual)^2))^0.5
knn.rmse <- rmse(predictions$knn, predictions$actual)
t1.rmse <- rmse(predictions$reg.tree,predictions$actual)
rf1.rmse <- rmse(predictions$random.forest,predictions$actual)
gbm1.rmse <- rmse(predictions$gbm,predictions$actual)

null.mpe <- median(abs((predictions$actual - predictions$null.mean)/predictions$actual))*100
knn.mpe <- median(abs((predictions$actual - predictions$knn)/predictions$actual))*100
t1.mpe <- median(abs((predictions$actual - predictions$reg.tree)/predictions$actual))*100
rf1.mpe <- median(abs((predictions$actual - predictions$random.forest)/predictions$actual))*100
gbm1.mpe <- median(abs((predictions$actual - predictions$gbm)/predictions$actual))*100

errors <- data.frame(c("null model","knn regression","regression tree","Bagged forest","Boosted forest"),
                     c(null.rmse,knn.rmse,t1.rmse,rf1.rmse,gbm1.rmse), 
                     c(null.mpe,knn.mpe,t1.mpe,rf1.mpe,gbm1.mpe))

colnames(errors) <- c("model","rmse","mpe")

ggplot(data=errors) + geom_point(aes(mpe,rmse,fill=model),shape= 21, size=7, color = "black") + 
  theme_bw(base_size=18) + xlab("median percentage error (%)") + ylab("room mean squared error (mgd)") +
  ggtitle("Comparison of Statistical Model errors") + scale_fill_brewer(palette="Set2")

## Boxplot of depth and divisions
pm = melt(predictions)
pm$value <- log10(pm$value)

ggplot(data=pm) + geom_boxplot(aes(variable,value,fill=variable),outlier.shape = NA)  +
  ylab("log10(mgd)") + theme_bw(base_size=18) + xlab(NULL) + ylim(-5,1) +
  facet_wrap(~divs) + theme(axis.text.x=element_blank(),axis.ticks.x=element_blank()) +
  labs(fill="Model") + scale_fill_brewer(palette="Set2")
  
## Resampling
resamps <- resamples(list(GBM = gbmFit3, SVM = svmFit, RDA = rdaFit))

# Caret package using multiple cores ----
library(doParallel)

control <- trainControl(method = "cv", number = 10)

cl <- makeCluster(detectCores())
registerDoParallel(cl)

# Start the clock
ptm <- proc.time()

rf.caret <- train(mgd ~., dsub2,
                     method='rf',
                     preProc=c('center','scale'),
                     trControl=control)

# Stop the clock
proc.time() - ptm

stopCluster(cl)









