
library(ggplot2); library(GGally); library(reshape2); 
library(hydroGOF); library(lme4); library(rpart);
library(kknn); library(randomForest); library(gbm);
library(caret); library(doParallel); library(gridExtra)

# Set local working directory
setwd("C:/Users/scworlan/Documents/MAP/OzarkWaterUse/Ozark-water-use")

# Load raw data (from B. Clarke 12/23/2015)
raw.data <- read.csv("WU_exportPopPrecipSSWU.csv", na.strings="-9999")

# Raw data processing
MO.raw <- subset(raw.data,State=="MO") # subset MO
MO.raw <- subset(MO.raw, yr %in% 1901:2010) #remove 2011-2015 bc no values
MO.raw <- MO.raw[-which(MO.raw$divs==""),] #remove rows with no division
MO.raw$mgd[MO.raw$mgd==0] <- NA #Kathy K. said that the 0 values are not real

# plot of annual sums
MO.annual.mgd <- aggregate(mgd ~ yr + divs, data = MO.raw, sum)
ggplot(MO.annual.mgd) + geom_point(aes(yr,mgd)) + geom_line(aes(yr,mgd)) +
  ylab("mgd") + theme_bw(base_size=14) + xlab("year") + facet_wrap(~divs,scales = "free_y") 

# MODEL FOR ONLY MO PUBLIC SUPPLY

## Modeling pre-processing
MO.ps <- subset(MO.raw, divs == "Public Supply") # subset public supply
MO.ps$mgd[MO.ps$yr==1901] <- min(MO.ps$mgd,na.rm=T)/2
mod.dat <- MO.ps[complete.cases(MO.ps),c(9,3,6,8,10,11,14,15)] #remove NA and subset features
mod.dat$log.mgd <- log10(mod.dat$mgd) # take log10 transformation
mod.dat[,3:8] <- scale(mod.dat[,3:8]) # scale the predictors

hist1 <- ggplot(mod.dat) + geom_histogram(aes(mgd),bins=60, color="white") + theme_bw(base_size=14)
hist2 <- ggplot(mod.dat) + geom_histogram(aes(log.mgd),bins=60, color="white") + theme_bw(base_size=14)

grid.arrange(hist1, hist2, ncol=1)

## null model
lm0 <- lm(log.mgd ~ 1, data=mod.dat)

## Linear model
lm <- lm(log.mgd ~ pop, data = mod.dat)

## second order polynomial multilevel (div_lu) model for population 
poly <- lmer(log.mgd ~ poly(pop,2,raw=T) + (1|div_lu), data = mod.dat)

## regression tree
set.seed(5)
tree <- rpart(log.mgd ~ poly(pop,2,raw=T) + div_lu, data = mod.dat, control=rpart.control(minsplit=100, cp=0))

## K-nearest neighbors
knn.train <- train.kknn(log.mgd ~ poly(pop,2,raw=T), data = mod.dat,  kmax = 100, 
                        kernel = c("rectangular", "triangular", 
                                   "biweight", "gaussian", 
                                   "triweight", "inv"));

knn <- kknn(log.mgd ~ poly(pop,2,raw=T), train = mod.dat, test = mod.dat, kernel = "inv", k = 100)

plot(knn.train)

## Random Forest
cl <- makeCluster(detectCores())
registerDoParallel(cl)

ptm <- proc.time() # Start the clock

set.seed(100)
rf.fit <- train(log.mgd ~ poly(pop,2,raw=T), data = mod.dat,
                method = "rf",
                tuneGrid = expand.grid(mtry=c(10,100,200)),
                importance =T,
                trControl = fitControl)

stopCluster(cl)

proc.time() - ptm # Stop the clock

set.seed(11)
rf <- randomForest(log.mgd ~ pop + I(pop^2), data = mod.dat, importance = TRUE, ntree = 10)

## Gradient boosting
gb_m <- gbm(log.mgd ~ poly(pop,2), data=mod.dat, distribution="gaussian", n.trees=100)

### train GBM model
fitControl <- trainControl(method = "cv", number = 10)

gbmGrid <-  expand.grid(.interaction.depth = seq(15,30,by=5),
                        .n.trees = c(1000,5000,7000,9000,11000),
                        .shrinkage = c(0.1),
                        .n.minobsinnode=100)

cl <- makeCluster(detectCores())
registerDoParallel(cl)

ptm <- proc.time() # Start the clock

set.seed(100)
gbm1<- caret::train(log.mgd ~ pop, data = mod.dat,
                        method = "gbm",
                        verbose = FALSE,
                        tuneGrid = gbmGrid)

stopCluster(cl)

proc.time() - ptm # Stop the clock

ggplot(gbm1) + theme_bw(base_size=14)

## predict 1901-2010 using different models
MO.features <- MO.ps[,c(3,6,8,10,11,14,15)]
MO.features[,2:7] <- scale(MO.features[,2:7])

MO.predict <- data.frame(cbind(MO.ps$yr,
                               MO.ps$mgd,
                               10^predict(poly, MO.features),
                               10^predict(tree, MO.features),
                               10^predict(knn.train, MO.features),
                               10^predict(rf, MO.features),
                               10^predict(gbm1, MO.features)))

colnames(MO.predict) <- c("yr","actual","poly","reg_tree","KNN","RF","GBM")

## Annual sums for plot
MO.annual.predict <- aggregate(.~yr, data = MO.predict, sum, na.action = na.pass, na.rm=TRUE)
MO.annual.predict$actual[MO.annual.predict$actual  == 0] <- NA
MO.annual.predict$actual[MO.annual.predict$yr ==1901] <- 0


## plot predictions
dmelt <- melt(MO.annual.predict,id="yr")

ggplot() + geom_line(data=subset(dmelt,variable != "actual"),aes(yr,value,color=variable),size=1)  +
  geom_point(data=subset(dmelt,variable == "actual"), aes(yr,value,fill=variable), shape=23, size = 3, alpha=0.7) + 
  ylab("mgd") + theme_bw(base_size=14) + xlab("year") +  labs(color="Models") + scale_color_brewer(palette="Set2") +
  labs(fill=NULL)











