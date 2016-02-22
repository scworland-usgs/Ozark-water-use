
library(ggplot2); library(GGally); library(reshape2); 
library(hydroGOF); library(lme4); library(rpart);
library(kknn); library(randomForest); library(gbm);
library(caret); library(doParallel)

# Set local working directory
setwd("C:/Users/scworlan/Documents/MAP/OzarkWaterUse/Ozark-water-use")

# Load raw data (from B. Clarke 12/23/2015)
raw.data <- read.csv("WU_exportPopPrecipSSWU.csv", na.strings="-9999")

# Raw data processing
MO.raw <- subset(raw.data,State=="MO") # subset MO
MO.raw <- subset(MO.raw, yr %in% 1901:2010) #remove 2011-2015 bc no values
MO.raw <- MO.raw[-which(MO.raw$divs==""),] #remove rows with no division
MO.raw$mgd[MO.raw$mgd==0] <- NA #Kathy K. said that the 0 values are not real
MO.raw$mgd[MO.raw$yr == 1901] <- 0 #make 1901 values zero

# plot of annual sums
MO.annual.mgd <- aggregate(mgd ~ yr + divs, data = MO.raw, sum)
ggplot(MO.annual.mgd) + geom_point(aes(yr,mgd)) + geom_line(aes(yr,mgd)) +
  ylab("mgd") + theme_bw(base_size=14) + xlab("year") + facet_wrap(~divs,scales = "free_y") 

# MODEL FOR ONLY MO PUBLIC SUPPLY

## Modeling pre-processing
MO.ps <- subset(MO.raw, divs == "Public Supply") # subset public supply
mod.dat <- MO.ps[complete.cases(MO.ps),c(9,3,6,8,10,11,14,15)] #remove NA and subset features
mod.dat[,3:8] <- scale(mod.dat[,3:8]) # scale the predictors

mod.dat$mgd[which(mod.dat$mgd==0)] = min(mod.dat$mgd[mod.dat$mgd>0])/10 # replace zero with small value
mod.dat[,1] <- log10(mod.dat[,1])

## null model
lm0 <- lm(mgd ~ 1, data=mod.dat)

## Linear model
lm <- lm(mgd ~ pop, data = mod.dat)

## second order polynomial multilevel (div_lu) model for population 
poly <- lmer(mgd ~ poly(pop,2) + (1|div_lu), data = mod.dat)

## regression tree
set.seed(5)
tree <- rpart(mgd ~ pop + div_lu, data = mod.dat, control=rpart.control(minsplit=100, cp=0))

## K-nearest neighbors
knn.train <- train.kknn(mgd ~ pop, data = mod.dat,  kmax = 100, 
                        kernel = c("rectangular", "triangular", 
                                   "biweight", "gaussian", 
                                   "triweight", "inv"));

knn <- kknn(mgd ~ pop, train = mod.dat, test = mod.dat, kernel = "inv", k = 100)

plot(knn.train)

## Random Forest
set.seed(11)
rf <- randomForest(mgd ~ pop, data = mod.dat, importance = TRUE, ntree = 100)

## Gradient boosting
gb_m <- gbm(mgd ~ pop, data=mod.dat, distribution="gaussian", n.trees=100)

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
gbm1<- caret::train(mgd ~ pop, data = mod.dat,
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
MO.annual.predict[MO.annual.predict  == 0] <- NA

## plot predictions
dmelt <- melt(MO.annual.predict,id="yr")

ggplot() + geom_line(data=subset(dmelt,variable != "actual"),aes(yr,value,color=variable),size=1)  +
  geom_point(data=subset(dmelt,variable == "actual"), aes(yr,value), shape=8,color="red", size = 3) + 
  ylab("mgd") + theme_bw(base_size=14) + xlab("year") +  labs(color="Models") + scale_color_brewer(palette="Set2")











