
library(ggplot2); library(reshape2); 

# Set local working directory
setwd("C:/Users/scworlan/Documents/MAP/OzarkWaterUse/Ozark-water-use")

# Load raw data (from B. Clarke 12/23/2015)
raw.data <- read.csv("WU_exportPopPrecipSSWU.csv", na.strings="-9999")

# Remove locations without mgd values (or zeros) for building the model
raw.data$mgd[raw.data$mgd==0] <- NA #Kathy K. said that the 0 values are not real
raw.data <- raw.data[-which(raw.data$yr==c(2011:2015)),] #remove 2011-2015 bc no values
raw.data <- raw.data[-which(raw.data$divs==""),] #remove rows with no division
d <- raw.data[complete.cases(raw.data),] #remove rows with NA to build models

ggplot(d) + geom_point(aes(yr,log10(mgd)),alpha=0.3, color ="darkgreen") + 
  geom_smooth(aes(yr,log10(mgd)),method=glm, formula=y ~ poly(x, 2)) + 
  facet_wrap(~divs) + theme_bw(base_size=18)

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

# LINEAR INTERPOLATION ----
# Convert raw.data into marix for interpolation
dmat <- dcast(raw.data, wellno~yr, value.var="mgd", fun.aggregate=median)
dmat2 <- as.matrix(dmat[,2:116])

# number of NAs in each rows
dmat2[,1] = 0
numNA <- apply(dmat2, 1, function(x) length(which(!is.na(x))))
dmat2 <- dmat2[which(numNA>1),] # remove rows with only NAs

dinterp1 <- zoo::na.approx(t(dmat2), rule=2)
dinterp2 <- as.data.frame(t(dinterp1))
dinterp3 <- log10(dinterp2[order(dinterp2$V115),])
colnames(dinterp3) <- 1901:2015

# heat map of values
library(pheatmap)

brks <- c(0.01,0.1,0.3,0.7,0.9,1,5,10)
brkslog <- c(-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0.5,1)
col.pal <- RColorBrewer::brewer.pal(9, "GnBu")
pheatmap(dinterp3[,2:115], cluster_row = F, cluster_col = F, show_rownames = F, color=col.pal, breaks=brkslog,
         legend_breaks=brkslog)

totals <- colSums(10^dinterp3[2:115])
plot(colnames(dinterp3)[2:115],totals, type="l",xlab="year",ylab="total mgd")

# PREPARE DATA ----
## Subset mgd and predictors for building models
dsub <- d[,c(9,2,6,8,14,15)]
row.names(dsub) <- NULL

## take the log of mgd for models
dsub[,1] <- log10(dsub[,1])

## scale the predictors
dsub[,3:6] <- scale(dsub[,3:6])

## partition data into 70% training and 30% testing set
set.seed(5)
x <- runif(0.7*nrow(dsub), 1, nrow(dsub))
train <- dsub[x,]
test <- dsub[-x,]

# STATISTICAL MODELS ----
library(doParallel)

## NULL model ----
null.means <- aggregate(train$mgd, list(train$divs), mean)
colnames(null.means) <- c("divs","null.mean")

## second order polynomial multi-level model ----
library(lme4)

poly <- lmer(mgd ~ poly(pop, 2) + (1|divs), data = train)

poly.predict <- predict(poly,test)

## Regression tree ----
library(rpart); library(rpart.plot); 
set.seed(5)
t1 <- rpart(mgd ~., data = train, control=rpart.control(minsplit=100, cp=0))
t1.predict <- predict(t1,test)

## Random Forest model ----
library(randomForest);

set.seed(11)
rf1 <- randomForest(mgd ~., data = train, importance = TRUE, ntree = 100)
rf1.predict <- predict(rf1,test)

### Variable importance plot
varImpPlot(rf, type=1, main = "Importance of Variables in 100 regression trees",pch=21)

## K nearest neighbors ----
library(kknn)
knn.train <- train.kknn(mgd ~., data = train,  kmax = 25, 
                        kernel = c("rectangular", "triangular", "biweight", "gaussian", "triweight", "inv"));
plot(knn.train)

knn <- kknn(mgd ~., train = train, test = test, kernel = "inv", k = 25)

## Boosted regression trees ----
load("gbmFit3.Rdata")
library(gbm); library(doParallel)

fitControl <- trainControl(method = "cv", number = 10)

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
                 tuneGrid = gbmGrid)

stopCluster(cl)

# Stop the clock
proc.time() - ptm

gbm1 <- gbmFit2$results[complete.cases(gbmFit2$results),]
gbm2 <- gbmFit3$results

ggplot(gbm2) + geom_line(aes(n.trees,RMSE,color=factor(interaction.depth))) +
  geom_point(aes(n.trees,RMSE,color=factor(interaction.depth),shape=factor(interaction.depth))) +
  theme_bw(base_size=18) + xlab("Boosting Iterations") + 
  labs(color="Max tree depth",shape="Max tree depth")

ggplot(gbmFit2) + theme_bw(base_size=18)

gbm.predict <- predict(gbmFit2, test)

## neural network ----
load("nnet.Rdata")
ggplot(nnet.fit) + theme_bw(base_size=18)

nnet2 <- nnet.fit$results

nnet.predict <- predict(nnet.fit, test)

ggplot(nnet2) + geom_line(aes(size,RMSE,color=factor(decay))) +
  geom_point(aes(size,RMSE,color=factor(decay),shape=factor(decay))) +
  theme_bw(base_size=18) + xlab("# Hidden Units") + 
  labs(color="Weight decay",shape="Weight decay")

# ASSEMBLE PREDICTIONS ON TEST SET ----
predictions <- data.frame(test$mgd,knn$fitted.values,t1.predict,rf1.predict, gbm.predict,nnet.predict)
row.names(predictions) = NULL
colnames(predictions) = c("actual","knn","reg.tree","random.forest","gbm","ann")
#predictions <- round(predictions,5)
predictions$divs <- test$divs
predictions <- predictions[-which(predictions$divs==""),]
predictions <- merge(null.means,predictions, by = "divs")
predictions <- predictions[,c(3,2,4:8,1)]

## Calculate the RMSE and MPE
library(hydroGOF)

null.rmse <- rmse(predictions$null.mean,predictions$actual)
#null.rmse <- (mean((predictions$null - predictions$actual)^2))^0.5
knn.rmse <- rmse(predictions$knn, predictions$actual)
t1.rmse <- rmse(predictions$reg.tree,predictions$actual)
rf1.rmse <- rmse(predictions$random.forest,predictions$actual)
gbm1.rmse <- rmse(predictions$gbm,predictions$actual)
ann.rmse <- rmse(predictions$ann,predictions$actual)

null.mpe <- median(abs((predictions$actual - predictions$null.mean)/predictions$actual))*100
knn.mpe <- median(abs((predictions$actual - predictions$knn)/predictions$actual))*100
t1.mpe <- median(abs((predictions$actual - predictions$reg.tree)/predictions$actual))*100
rf1.mpe <- median(abs((predictions$actual - predictions$random.forest)/predictions$actual))*100
gbm1.mpe <- median(abs((predictions$actual - predictions$gbm)/predictions$actual))*100
ann.mpe <- median(abs((predictions$actual - predictions$ann)/predictions$actual))*100

errors <- data.frame(c("null model","knn regression","regression tree","Random forest","Boosted forest","Neural Network"),
                     c(null.rmse,knn.rmse,t1.rmse,rf1.rmse,gbm1.rmse,ann.rmse), 
                     c(null.mpe,knn.mpe,t1.mpe,rf1.mpe,gbm1.mpe,ann.mpe))

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

# MAKE FINAL PREDICTIONS AND PLOT ----
d.final <- raw.data[,c(9,2,6,8,14,15)]
row.names(d.final) <- NULL

# plot precip or pop through time and by division as a check
d.finalCheck <- aggregate(precip ~ divs + yr, data = d.final, mean)
ggplot(data=d.finalCheck) + geom_line(aes(yr,precip,color=divs)) + theme_bw(base_size=18)

## scale the predictors
d.final[,3:6] <- scale(d.final[,3:6])

## predict mgd
poly.final <- data.frame(10^(predict(poly, d.final)))

rt.final <- data.frame(predict(t1, d.final))
rf.final <- data.frame(predict(rf1, d.final))
knn.final <- data.frame(predict(knn.train, d.final))
gbm.final <- data.frame(predict(gbmFit2, d.final))
ann.final <- data.frame(predict(nnet.fit, d.final))

## assemble final predictions
out.data <- data.frame(c(raw.data[,c(1,2,6)],poly.final))
colnames(out.data)[4] <- c("mlm.poly")
out.data.sum <- aggregate(cbind(mlm.poly) ~ yr + divs, data = out.data, sum)
actual.mgd <- aggregate(mgd ~ yr + divs, data = raw.data, sum)


## plots through time and divisions
out.melt = melt(out.data.sum, id=c("yr"))


ggplot() + geom_line(data=out.data.sum, aes(yr,mlm.poly))  +
  geom_point(data=actual.mgd, aes(yr,mgd)) + geom_line(data=actual.mgd, aes(yr,mgd),alpha=0.5) +
  ylab("mgd") + theme_bw(base_size=18) + xlab("year") + 
  facet_wrap(~divs,scales = "free_y") + labs(color="Model") + scale_fill_brewer(palette="Set2")

ggplot() + geom_point(data=actual.mgd, aes(yr,mgd)) + geom_line(data=actual.mgd, aes(yr,mgd)) +
  ylab("mgd") + theme_bw(base_size=18) + xlab("year") + facet_wrap(~divs,scales = "free_y") 











