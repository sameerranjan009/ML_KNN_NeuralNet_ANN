library(tidyverse)
library(MASS)
library(readr)
library(modelr)
library(class)
library (dplyr)
library(VIM)

install.packages("VIM")
energy <- read.csv("energydata_complete.csv")
getwd()
setwd("C:/JSOM/ML")

High = ifelse(energy$Appliances<=100,0,1)
energy = data.frame(energy,High)
energy<- energy[,-c(1,2)]
sample_ind <- sample(nrow(energy),nrow(energy)*0.70)
train <- energy[sample_ind,]
test <- energy[-sample_ind,]
train_class<- energy[sample_ind,28]
test_class <- energy[-sample_ind,28]
library(class)
pr <- knn(train,test,cl=train_class,k=6)
plot(pr)
tab <- table(pr,test_class)
tab
pr1 <- knn(train,test,cl=train_class,k=10)
plot(pr1)
tab1 <- table(pr1,test_class)

pr2 <- knn(train,test,cl=train_class,k=15)

tab2 <- table(pr2,test_class)

pr3 <- knn(train,test,cl=train_class,k=3)

tab3 <- table(pr3,test_class)

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
a<- accuracy(tab)
a1<-accuracy(tab1)
a2<-accuracy(tab2)
a3<-accuracy(tab3)
a3
accuracy <- c(a,a1,a2,a3)
k <- c(6,10,15,3)
d <- data.frame(accuracy,k)
ggplot()+
  geom_point(aes(k,accuracy))+
  geom_smooth()

# for different partition of dataset at 60% training
sample_ind <- sample(nrow(energy),nrow(energy)*0.60)
train <- energy[sample_ind,]
test <- energy[-sample_ind,]
train_class<- energy[sample_ind,28]
test_class <- energy[-sample_ind,28]

pr3 <- knn(train,test,cl=train_class,k=3)

tab3 <- table(pr3,test_class)

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tab3)

# for different partition of dataset at 80% training
sample_ind <- sample(nrow(energy),nrow(energy)*0.80)
train <- energy[sample_ind,]
test <- energy[-sample_ind,]
train_class<- energy[sample_ind,28]
test_class <- energy[-sample_ind,28]

pr4 <- knn(train,test,cl=train_class,k=3)

tab4 <- table(pr4,test_class)

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tab4)


install.packages('caret')
library(caret)
NROW(test$High)
NROW(pr)
confusionMatrix(table(pr,test$High))

i=1
k.optm=1
for (i in 1:28){
  knn.mod <- knn(train=train, test=test, cl=train_class, k=i)
  k.optm[i] <- 100 * sum(test_class == knn.mod)/NROW(test_class)
  k=i
  cat(k,'=',k.optm[i],'
')}

plot(k.optm, type="b", xlab="K- Value",ylab="Accuracy level")


#Simple Neural Net

install.packages("neuralnet")

# load library
require(neuralnet)

# fit neural network
nn=neuralnet(High~.,data=energy, hidden=3,act.fct = "tanh",
             linear.output = FALSE)

plot(nn)

sigmoid = function(x) {
  1 / (1 + exp(-x))
}

nn=neuralnet(High~.,data=energy, hidden=3,act.fct =sigmoid,
             linear.output = TRUE)

nn=neuralnet(High~.,data=energy, hidden=3,act.fct ="logistic",
             linear.output = TRUE)
plot(nn)

# H20 package 

install.packages("h2o")
library(h2o)
h2o.init(nthreads=-1)

d.hex <- as.h2o(energy,destination_frame="d.hex")
head(d.hex)

set.seed(99)
split <-h2o.splitFrame(data=d.hex,ratios = 0.75)
train <- split[[1]] 
test <- split[[2]]

model_nn <- h2o.deeplearning(x=1:27,y="High", training_frame = train,hidden =5,model_id ="model_nn")
perf <- h2o.performance(model_nn,test)
perf

pred<- as.data.frame(h2o.predict(model_nn, test))
test1 <- as.data.frame(test)
cor(pred,test1$High)


model_nn2 <- h2o.deeplearning(x=1:27,y="High", training_frame = train,hidden =5,activation ="Tanh",model_id ="model_nn")
perf <- h2o.performance(model_nn2,test)
perf

pred<- as.data.frame(h2o.predict(model_nn2, test))
test1 <- as.data.frame(test)
cor(pred,test1$High)


model_nn3 <- h2o.deeplearning(x=1:27,y="High", training_frame = train,hidden =5,activation ="TanhWithDropout",model_id ="model_nn")
perf <- h2o.performance(model_nn3,test)
perf

pred<- as.data.frame(h2o.predict(model_nn3, test))
test1 <- as.data.frame(test)
cor(pred,test1$High)

model_nn4 <- h2o.deeplearning(x=1:27,y="High", training_frame = train,hidden =5,activation ="Maxout",model_id ="model_nn")
perf <- h2o.performance(model_nn4,test)
perf

pred<- as.data.frame(h2o.predict(model_nn4, test))
test1 <- as.data.frame(test)
cor(pred,test1$High)

model_nn5 <- h2o.deeplearning(x=1:27,y="High", training_frame = train,hidden =5,activation ="MaxoutWithDropout",model_id ="model_nn")
perf <- h2o.performance(model_nn5,test)
perf

pred<- as.data.frame(h2o.predict(model_nn5, test))
test1 <- as.data.frame(test)
cor(pred,test1$High)

model_nn6 <- h2o.deeplearning(x=1:27,y="High", training_frame = train,hidden =5,activation ="RectifierWithDropout",model_id ="model_nn")
perf <- h2o.performance(model_nn6,test)
perf

pred<- as.data.frame(h2o.predict(model_nn6, test))
test1 <- as.data.frame(test)
cor(pred,test1$High)

#Experiment with Layers 
model_nn6 <- h2o.deeplearning(x=1:27,y="High", training_frame = train,hidden =5,activation ="RectifierWithDropout",model_id ="model_nn")
perf <- h2o.performance(model_nn6,test)
perf

pred<- as.data.frame(h2o.predict(model_nn6, test))
test1 <- as.data.frame(test)
a<- cor(pred,test1$High)

model_nn6 <- h2o.deeplearning(x=1:27,y="High", training_frame = train,hidden =15,activation ="RectifierWithDropout",model_id ="model_nn")
perf <- h2o.performance(model_nn6,test)
perf

pred<- as.data.frame(h2o.predict(model_nn6, test))
test1 <- as.data.frame(test)
a1<- cor(pred,test1$High)

model_nn6 <- h2o.deeplearning(x=1:27,y="High", training_frame = train,hidden =25,activation ="RectifierWithDropout",model_id ="model_nn")
perf <- h2o.performance(model_nn6,test)
perf

pred<- as.data.frame(h2o.predict(model_nn6, test))
test1 <- as.data.frame(test)
a2<- cor(pred,test1$High)

model_nn6 <- h2o.deeplearning(x=1:27,y="High", training_frame = train,hidden =45,activation ="RectifierWithDropout",model_id ="model_nn")
perf <- h2o.performance(model_nn6,test)
perf

pred<- as.data.frame(h2o.predict(model_nn6, test))
test1 <- as.data.frame(test)
a3<- cor(pred,test1$High)

model_nn6 <- h2o.deeplearning(x=1:27,y="High", training_frame = train,hidden =75,activation ="RectifierWithDropout",model_id ="model_nn")
perf <- h2o.performance(model_nn6,test)
perf

pred<- as.data.frame(h2o.predict(model_nn6, test))
test1 <- as.data.frame(test)
a4<- cor(pred,test1$High)

Layers <- c(5,15,25,45,75)
predictions <- c(a,a1,a2,a3,a4)
d <- data.frame(Layers,predictions)
plot(Layers,predictions)
ggplot(d, aes(Layers,predictions))+
  geom_point()+
  geom_smooth()

#Experiment with rate 

model_nn6 <- h2o.deeplearning(x=1:27,y="High", training_frame = train,hidden =45,activation ="RectifierWithDropout",model_id ="model_nn",adaptive_rate=F, rate = 0.01)
perf <- h2o.performance(model_nn6,test)
perf

pred<- as.data.frame(h2o.predict(model_nn6, test))
test1 <- as.data.frame(test)
a<- cor(pred,test1$High)

model_nn6 <- h2o.deeplearning(x=1:27,y="High", training_frame = train,hidden =45,activation ="RectifierWithDropout",model_id ="model_nn",adaptive_rate=F, rate = 0.001)
perf <- h2o.performance(model_nn6,test)
perf

pred<- as.data.frame(h2o.predict(model_nn6, test))
test1 <- as.data.frame(test)
a1<- cor(pred,test1$High)

model_nn6 <- h2o.deeplearning(x=1:27,y="High", training_frame = train,hidden =45,activation ="RectifierWithDropout",model_id ="model_nn",adaptive_rate=F, rate = 0.0001)
perf <- h2o.performance(model_nn6,test)
perf

pred<- as.data.frame(h2o.predict(model_nn6, test))
test1 <- as.data.frame(test)
a2<- cor(pred,test1$High)

model_nn6 <- h2o.deeplearning(x=1:27,y="High", training_frame = train,hidden =45,activation ="RectifierWithDropout",model_id ="model_nn",adaptive_rate=F, rate = 0.00001)
perf <- h2o.performance(model_nn6,test)
perf

pred<- as.data.frame(h2o.predict(model_nn6, test))
test1 <- as.data.frame(test)
a3<- cor(pred,test1$High)

rate <- c(0.005,0.01,0.001,0.0001,0.00001)
predictions <- c("0.4521",a,a1,a2,a3)
d <- data.frame(rate,predictions)

ggplot(d, aes(rate,predictions))+
  geom_point()+
  geom_smooth()


# NewYork Dataset

newyork <- read.csv('newyork_hotel.csv')
getwd()
newyork1 <- na.omit(newyork)
summary(newyork1$price)
High = ifelse(newyork1$price<=142.3,0,1)
newyork1 = data.frame(newyork1,High)
newyork1<- newyork1%>%mutate(as.numeric(room_type))

newyork1<- newyork1[,-10]
newyork1<- newyork1[,-c(2,4)]
newyork1<- newyork1[,-c(3,4)]
newyork1<- newyork1[,-5]
newyork1<- newyork1[,-7]
sample_ind1 <- sample(nrow(newyork1),nrow(newyork1)*0.70)
train1 <- newyork1[sample_ind1,]
test1 <- newyork1[-sample_ind1,]
train_class1<- newyork1[sample_ind1,11]
test_class1 <- newyork1[-sample_ind1,11]
room_type <- as.numeric(newyork1$room_type)

pr <- knn(train1,test1,cl=train_class1,k=6)

tab <- table(pr,test_class1)

pr1 <- knn(train1,test1,cl=train_class1,k=10)

tab1 <- table(pr1,test_class1)

pr2 <- knn(train1,test1,cl=train_class1,k=15)

tab2 <- table(pr2,test_class1)

pr3 <- knn(train1,test1,cl=train_class1,k=3)

tab3 <- table(pr3,test_class1)

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tab)
accuracy(tab1)
accuracy(tab2)
accuracy(tab3)
plot(pr)
plot(pr1)
plot(pr2)


i=1
k.optm=1
for (i in 1:50){
  knn.mod1 <- knn(train=train1, test=test1, cl=train_class1, k=i)
  k.optm[i] <- 100 * sum(test_class1 == knn.mod1)/NROW(test_class1)
  k=i
  cat(k,'=',k.optm[i],'
')}

plot(k.optm, type="b", xlab="K- Value",ylab="Accuracy level")

# Changing Partition rate for K = 46

sample_ind1 <- sample(nrow(newyork1),nrow(newyork1)*0.70)
train1 <- newyork1[sample_ind1,]
test1 <- newyork1[-sample_ind1,]
train_class1<- newyork1[sample_ind1,11]
test_class1 <- newyork1[-sample_ind1,11]


pr <- knn(train1,test1,cl=train_class1,k=46)

tab <- table(pr,test_class1)
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tab)

# for 60% dataset
sample_ind1 <- sample(nrow(newyork1),nrow(newyork1)*0.60)
train1 <- newyork1[sample_ind1,]
test1 <- newyork1[-sample_ind1,]
train_class1<- newyork1[sample_ind1,11]
test_class1 <- newyork1[-sample_ind1,11]

pr <- knn(train1,test1,cl=train_class1,k=46)

tab <- table(pr,test_class1)
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tab)

# for 80% dataset
sample_ind1 <- sample(nrow(newyork1),nrow(newyork1)*0.80)
train1 <- newyork1[sample_ind1,]
test1 <- newyork1[-sample_ind1,]
train_class1<- newyork1[sample_ind1,11]
test_class1 <- newyork1[-sample_ind1,11]

pr <- knn(train1,test1,cl=train_class1,k=46)

tab <- table(pr,test_class1)
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tab)


#ANN
 
install.packages("h2o")
library(h2o)
h2o.init(nthreads=-1)

d.hex1 <- as.h2o(newyork1,destination_frame="d.hex1")
head(d.hex1)

set.seed(99)
split1 <-h2o.splitFrame(data=d.hex1,ratios = 0.75)
train_1 <- split1[[1]] 
test_1 <- split1[[2]]

model_nn_1 <- h2o.deeplearning(x=1:9,y="High", training_frame = train_1,hidden =5,model_id ="model_nn_1")
perf_1 <- h2o.performance(model_nn_1,test_1)
perf_1
model_nn_1
pred_1<- as.data.frame(h2o.predict(model_nn_1, test_1))
test_1 <- as.data.frame(test_1)
cor(pred_1,test_1$High)
head(as.data.frame(h2o.varimp(model_nn_1)))
plot(model_nn_1)

model_nn_2 <- h2o.deeplearning(x=1:9,y="High", training_frame = train_1,hidden =5,activation ="Tanh",model_id ="model_nn_2")
perf_2 <- h2o.performance(model_nn_2,test_1)
perf_2
model_nn_2 < as.h2o(model_nn_2)
pred_2<- as.data.frame(h2o.predict(model_nn_2, test_1))
test <- as.data.frame(test_1)
cor(pred_2,test$High)


model_nn_3 <- h2o.deeplearning(x=1:9,y="High", training_frame = train_1,hidden =5,activation ="TanhWithDropout",model_id ="model_nn_3")
perf <- h2o.performance(model_nn_3,test_1)
perf
model_nn_3
pred<- as.data.frame(h2o.predict(model_nn_3, test_1))
test1 <- as.data.frame(test_1)
cor(pred,test1$High)

model_nn_4 <- h2o.deeplearning(x=1:8,y="High", training_frame = train_1,hidden =5,activation ="Maxout",model_id ="model_nn")
perf <- h2o.performance(model_nn4,test_1)
perf

pred<- as.data.frame(h2o.predict(model_nn_4, test_1))
test1 <- as.data.frame(test_1)
cor(pred,test1$High)

model_nn_5 <- h2o.deeplearning(x=1:8,y="High", training_frame = train_1,hidden =5,activation ="MaxoutWithDropout",model_id ="model_nn")
perf <- h2o.performance(model_nn4,test_1)
perf

pred<- as.data.frame(h2o.predict(model_nn_5, test_1))
test1 <- as.data.frame(test_1)
cor(pred,test1$High)

model_nn_6 <- h2o.deeplearning(x=1:8,y="High", training_frame = train_1,hidden =5,activation ="RectifierWithDropout",model_id ="model_nn")
perf <- h2o.performance(model_nn4,test_1)
perf

pred<- as.data.frame(h2o.predict(model_nn_6, test_1))
test1 <- as.data.frame(test_1)
cor(pred,test1$High)


# Experiment with Layer

#with layers = 10
model_nn_2 <- h2o.deeplearning(x=1:9,y="High", training_frame = train_1,hidden =10,activation ="Tanh",model_id ="model_nn_2", )

perf_2 <- h2o.performance(model_nn_2,test_1)

pred_2<- as.data.frame(h2o.predict(model_nn_2, test_1))
test <- as.data.frame(test_1)
a<- cor(pred_2,test$High)

# with layers =15
model_nn_3 <- h2o.deeplearning(x=1:9,y="High", training_frame = train_1,hidden =15,activation ="Tanh",model_id ="model_nn_2", )

perf_2 <- h2o.performance(model_nn_3,test_1)

pred_2<- as.data.frame(h2o.predict(model_nn_3, test_1))
test <- as.data.frame(test_1)
a1<- cor(pred_2,test$High)

#with layers = 20
model_nn_4 <- h2o.deeplearning(x=1:9,y="High", training_frame = train_1,hidden =20,activation ="Tanh",model_id ="model_nn_2", )

perf_2 <- h2o.performance(model_nn_2,test_1)

pred_2<- as.data.frame(h2o.predict(model_nn_2, test_1))
test <- as.data.frame(test_1)
a2<- cor(pred_2,test$High)

#with layers = 25
model_nn_2 <- h2o.deeplearning(x=1:9,y="High", training_frame = train_1,hidden =25,activation ="Tanh",model_id ="model_nn_2", )

perf_2 <- h2o.performance(model_nn_2,test_1)

pred_2<- as.data.frame(h2o.predict(model_nn_2, test_1))
test <- as.data.frame(test_1)
a3<- cor(pred_2,test$High)

#with layers = 50
model_nn_2 <- h2o.deeplearning(x=1:9,y="High", training_frame = train_1,hidden =50,activation ="Tanh",model_id ="model_nn_2", )

perf_2 <- h2o.performance(model_nn_2,test_1)

pred_2<- as.data.frame(h2o.predict(model_nn_2, test_1))
test <- as.data.frame(test_1)
a4<- cor(pred_2,test$High)

#with layers = 100
model_nn_2 <- h2o.deeplearning(x=1:9,y="High", training_frame = train_1,hidden =100,activation ="Tanh",model_id ="model_nn_2", )

perf_2 <- h2o.performance(model_nn_2,test_1)

pred_2<- as.data.frame(h2o.predict(model_nn_2, test_1))
test <- as.data.frame(test_1)
a5<- cor(pred_2,test$High)
#with layers = 65
model_nn_2 <- h2o.deeplearning(x=1:9,y="High", training_frame = train_1,hidden =68,activation ="Tanh",model_id ="model_nn_2", )

perf_2 <- h2o.performance(model_nn_2,test_1)

pred_2<- as.data.frame(h2o.predict(model_nn_2, test_1))
test <- as.data.frame(test_1)
a6<- cor(pred_2,test$High)
Layers <- c(10,15,20,25,50,68,100)
predictions <- c(a,a1,a2,a3,a4,a5,a6)
d <- data.frame(Layers,predictions)
plot(Layers,predictions)
ggplot(d, aes(Layers,predictions))+
  geom_point()+
  geom_smooth()


# Adaptive rate for rate = 0.01
model_nn_2 <- h2o.deeplearning(x=1:9,y="High", training_frame = train_1,hidden =15,activation ="Tanh",model_id ="model_nn_2", adaptive_rate=F, rate = 0.01)
perf_2 <- h2o.performance(model_nn_2,test_1)

pred_2<- as.data.frame(h2o.predict(model_nn_2, test_1))
test <- as.data.frame(test_1)
a<- cor(pred_2,test$High)

# Adaptive rate for rate = 0.001
model_nn_2 <- h2o.deeplearning(x=1:9,y="High", training_frame = train_1,hidden =15,activation ="Tanh",model_id ="model_nn_2", adaptive_rate=F, rate = 0.001)
perf_2 <- h2o.performance(model_nn_2,test_1)

pred_2<- as.data.frame(h2o.predict(model_nn_2, test_1))
test <- as.data.frame(test_1)
a1<- cor(pred_2,test$High)

# Adaptive rate for rate = 0.0001
model_nn_2 <- h2o.deeplearning(x=1:9,y="High", training_frame = train_1,hidden =15,activation ="Tanh",model_id ="model_nn_2", adaptive_rate=F, rate = 0.0001)
perf_2 <- h2o.performance(model_nn_2,test_1)

pred_2<- as.data.frame(h2o.predict(model_nn_2, test_1))
test <- as.data.frame(test_1)
a3<- cor(pred_2,test$High)

# Adaptive rate for rate = 0.005
model_nn_2 <- h2o.deeplearning(x=1:9,y="High", training_frame = train_1,hidden =15,activation ="Tanh",model_id ="model_nn_2", adaptive_rate=F, rate = 0.005)
perf_2 <- h2o.performance(model_nn_2,test_1)

pred_2<- as.data.frame(h2o.predict(model_nn_2, test_1))
test <- as.data.frame(test_1)
a4<- cor(pred_2,test$High)

# Adaptive rate for rate = 0.00099
model_nn_2 <- h2o.deeplearning(x=1:9,y="High", training_frame = train_1,hidden =15,activation ="Tanh",model_id ="model_nn_2", adaptive_rate=F, rate = 0.00099)
perf_2 <- h2o.performance(model_nn_2,test_1)

pred_2<- as.data.frame(h2o.predict(model_nn_2, test_1))
test <- as.data.frame(test_1)
a5<- cor(pred_2,test$High)


# Adaptive rate for rate = 0.00050
model_nn_2 <- h2o.deeplearning(x=1:9,y="High", training_frame = train_1,hidden =15,activation ="Tanh",model_id ="model_nn_2", adaptive_rate=F, rate = 0.0005)
perf_2 <- h2o.performance(model_nn_2,test_1)

pred_2<- as.data.frame(h2o.predict(model_nn_2, test_1))
test <- as.data.frame(test_1)
a6<- cor(pred_2,test$High)

rate <- c(0.01,0.001,0.0001,0.005,0.00099,0.0005)
prediction <- c(a,a1,a3,a4,a5,a6)
d <- data.frame(rate,prediction)
ggplot(d, aes(rate,prediction))+
  geom_point()+
  geom_smooth(method = lm)

# Adaptive rate for rate = 0.01
model_nn_2 <- h2o.deeplearning(x=1:9,y="High", training_frame = train_1,hidden =68,activation ="Tanh",model_id ="model_nn_2", adaptive_rate=F, rate = 0.01)
perf_2 <- h2o.performance(model_nn_2,test_1)

pred_2<- as.data.frame(h2o.predict(model_nn_2, test_1))
test <- as.data.frame(test_1)
a<- cor(pred_2,test$High)

# Adaptive rate for rate = 0.001
model_nn_2 <- h2o.deeplearning(x=1:9,y="High", training_frame = train_1,hidden =68,activation ="Tanh",model_id ="model_nn_2", adaptive_rate=F, rate = 0.001)
perf_2 <- h2o.performance(model_nn_2,test_1)

pred_2<- as.data.frame(h2o.predict(model_nn_2, test_1))
test <- as.data.frame(test_1)
a1<- cor(pred_2,test$High)

# Adaptive rate for rate = 0.0001
model_nn_2 <- h2o.deeplearning(x=1:9,y="High", training_frame = train_1,hidden =68,activation ="Tanh",model_id ="model_nn_2", adaptive_rate=F, rate = 0.0001)
perf_2 <- h2o.performance(model_nn_2,test_1)

pred_2<- as.data.frame(h2o.predict(model_nn_2, test_1))
test <- as.data.frame(test_1)
a3<- cor(pred_2,test$High)

# Adaptive rate for rate = 0.005
model_nn_2 <- h2o.deeplearning(x=1:9,y="High", training_frame = train_1,hidden =68,activation ="Tanh",model_id ="model_nn_2", adaptive_rate=F, rate = 0.005)
perf_2 <- h2o.performance(model_nn_2,test_1)

pred_2<- as.data.frame(h2o.predict(model_nn_2, test_1))
test <- as.data.frame(test_1)
a4<- cor(pred_2,test$High)

# Adaptive rate for rate = 0.00099
model_nn_2 <- h2o.deeplearning(x=1:9,y="High", training_frame = train_1,hidden =68,activation ="Tanh",model_id ="model_nn_2", adaptive_rate=F, rate = 0.00099)
perf_2 <- h2o.performance(model_nn_2,test_1)

pred_2<- as.data.frame(h2o.predict(model_nn_2, test_1))
test <- as.data.frame(test_1)
a5<- cor(pred_2,test$High)


# Adaptive rate for rate = 0.00050
model_nn_2 <- h2o.deeplearning(x=1:9,y="High", training_frame = train_1,hidden =68,activation ="Tanh",model_id ="model_nn_2", adaptive_rate=F, rate = 0.0005)
perf_2 <- h2o.performance(model_nn_2,test_1)

pred_2<- as.data.frame(h2o.predict(model_nn_2, test_1))
test <- as.data.frame(test_1)
a6<- cor(pred_2,test$High)

rate <- c(0.01,0.001,0.0001,0.005,0.00099,0.0005)
prediction <- c(a,a1,a3,a4,a5,a6)
d <- data.frame(rate,prediction)
ggplot(d, aes(rate,prediction))+
  geom_point()+
  geom_smooth(method = lm)