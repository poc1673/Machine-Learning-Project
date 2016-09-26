# Practical Machine Learning Class Project
Peter Caya  
September 21, 2016  



# Preamble

The goal of this project is to take the exercise data from groupware and use that to predict the manner in which the respondents in the test dataset exercised.  To do this I will train and compare three models:

1. Random forest.
2. LDA
3. Naive Bayes Classification

These methods were solely implemented using the caret package.  trainControl was used to specify 5-fold cross-validation with five repititions times.  The train function was then used to implement the four methods discussed above.  

The training dataset was partitioned into two pieces in order to compare the efficacy of the models.  The set which the models were trained on was 80% of the original training data.  This training data was then used on the much smaller testing dataset.


# Data Loading and Cleaning

The training data and testing data are loaded.  Cleaning the data consists of several steps:

  1.  Some columns are largely empty.  Columns which consist of more than 50% N/A values are removed from the training data set.
  2.  Several variables seem to exist as indexes or ID values.  Since these aren't relevant to the classe variable, they are removed.  These variables are X,user_name,raw_timestamp_part_1,raw_timestamp_part_2,cvtd_timestamp,new_window and num_window.      
3.  The two operations mentioned above are then used to clean the testing data as well so that the columns in the testing data are consistent with those in the training data.



```r
set.seed(13)

# A function which returns TRUE if the number of NA values is less than portion and false if it is greater.
portion.na <- function(X, portion = .5)
{
  if( sum(is.na(X))/length(X) > portion){return(FALSE)}
  else{return(TRUE)}
}

testingData <-read.table(file = "F:/Machine Learning/Project/pml-testing.csv",header = TRUE, sep = ",",na.strings=c("NA","#DIV/0!",""))
trainingData <-read.table(file = "F:/Machine Learning/Project/pml-training.csv",header = TRUE, sep = ",",na.strings=c("NA","#DIV/0!",""))

na_acceptable <- apply(X = trainingData,MARGIN = 2 , FUN = portion.na, portion = .5)
trainingData<-trainingData[,na_acceptable]

classe_training <- trainingData%>% select_("classe")

training_indices <- createDataPartition(trainingData$classe,p = .01,list = FALSE)
trainingData <- trainingData[-60]
training_data1<- trainingData[training_indices,]
training_data2 <- trainingData[-training_indices,]
classe1 <- as.factor(classe_training[training_indices])
classe2 <-as.factor( classe_training[-training_indices])
training_data1<- cbind(training_data1,classe1)
training_data2<- cbind(training_data2,classe2)
training_data1<- training_data1 %>% rename_(classe = "classe1"  ) %>% 
                                    dplyr::select(-c(X:num_window))
training_data2<- training_data2 %>% rename_(classe = "classe2"  ) %>%
                                    dplyr::select(-c(X:num_window))

names_used <- na.omit(match(names(training_data1),names(testingData)))
data_to_test <- testingData[names_used]


fitControl <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 5)
```

## Random Forest

```r
RF_model <- train(data = training_data1, 
                  classe~.,
                  method  = "rf" , trControl = fitControl)
RF_training2 <-predict(RF_model, training_data2,type="raw")
plot(RF_training2)
```

![](Course_Project_files/figure-html/unnamed-chunk-2-1.png)<!-- -->

```r
RF_confusion <- confusionMatrix(RF_training2, training_data2$classe)
print(RF_confusion)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4683  601  195  358  119
##          B  211 2090  325   76  450
##          C  387  469 2733  797  393
##          D  180  510   64 1757   83
##          E   63   89   70  195 2525
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7099          
##                  95% CI : (0.7034, 0.7163)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6322          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8478   0.5560   0.8069  0.55199   0.7073
## Specificity            0.9084   0.9322   0.8724  0.94846   0.9737
## Pos Pred Value         0.7863   0.6631   0.5719  0.67733   0.8583
## Neg Pred Value         0.9376   0.8974   0.9553  0.91527   0.9366
## Prevalence             0.2844   0.1935   0.1744  0.16388   0.1838
## Detection Rate         0.2411   0.1076   0.1407  0.09046   0.1300
## Detection Prevalence   0.3066   0.1623   0.2460  0.13355   0.1515
## Balanced Accuracy      0.8781   0.7441   0.8397  0.75023   0.8405
```

Based on the results above I expect my implementation of random forest to have accuracy of 0.70988 when applied to an out of sample group.  We see from the sensitivity and specificy statistics that this managed to keep misclassifications uncommon.


## LDA

```r
LDA_model <- train(data = training_data1, 
                   classe~.,
                   method  = "lda", trControl = fitControl)
LDA_training <-predict(LDA_model, training_data2,type="raw")
#plot(NB_model)
LDA_confusion <- confusionMatrix(LDA_training, training_data2$classe)
print(LDA_confusion)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3817  577  486  396  245
##          B  487 1870  394   98  655
##          C  565  421 2026  623  417
##          D  418  487  370 1921  564
##          E  237  404  111  145 1689
## 
## Overall Statistics
##                                          
##                Accuracy : 0.583          
##                  95% CI : (0.576, 0.5899)
##     No Information Rate : 0.2844         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.4732         
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.6910  0.49747   0.5982   0.6035  0.47311
## Specificity            0.8774  0.89568   0.8737   0.8868  0.94342
## Pos Pred Value         0.6914  0.53368   0.5000   0.5109  0.65313
## Neg Pred Value         0.8772  0.88134   0.9115   0.9194  0.88828
## Prevalence             0.2844  0.19353   0.1744   0.1639  0.18380
## Detection Rate         0.1965  0.09628   0.1043   0.0989  0.08696
## Detection Prevalence   0.2843  0.18040   0.2086   0.1936  0.13314
## Balanced Accuracy      0.7842  0.69658   0.7359   0.7451  0.70826
```

LDA performed much more poorly than the random forest algorithm with an accuracy of 0.5829686 with similar losses in sensitivity and specificity.


## Naive-Bayes

```r
NB_model <- train(data = training_data1, 
                  classe~.,
                  method  = "nb"  )
NB_training <-predict(NB_model, training_data2,type="raw", trControl = fitControl)
plot(NB_model)
```

![](Course_Project_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

```r
NB_confusion <- confusionMatrix(NB_training, training_data2$classe)
print(NB_confusion)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3239  344 1119  923  208
##          B  421 2016  562  100  732
##          C  972  444 1169  453  464
##          D  803  829  429 1385  445
##          E   89  126  108  322 1721
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4907          
##                  95% CI : (0.4836, 0.4977)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3551          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.5864   0.5363  0.34514  0.43512  0.48207
## Specificity            0.8134   0.8841  0.85451  0.84569  0.95931
## Pos Pred Value         0.5553   0.5262  0.33381  0.35595  0.72739
## Neg Pred Value         0.8319   0.8882  0.86069  0.88424  0.89160
## Prevalence             0.2844   0.1935  0.17438  0.16388  0.18380
## Detection Rate         0.1668   0.1038  0.06019  0.07131  0.08861
## Detection Prevalence   0.3003   0.1972  0.18030  0.20033  0.12181
## Balanced Accuracy      0.6999   0.7102  0.59983  0.64041  0.72069
```

Naive Bayes performed slightly better than LDA but still more poorly than random forest.  It had an accuracy of 0.4906554 with fairly unever presentations for sensitivity and specificity.

## Combining the Three Methods:

Displayed below is the end product of this project.  I have taken the three algorithms I trained for this and used them on the testing data set.  To further improve accuracy I then used a simple vote to determine the final answer which was then output to a csv file.


```r
vote <- function(X)
{
  names(sort(X,decreasing = TRUE))[1]
}

predictions_RF<- predict(RF_model, data_to_test, type = "raw")
predictions_LDA<- predict(LDA_model, data_to_test, type = "raw")
predictions_NB<- predict(NB_model, data_to_test, type = "raw")
# predictions_GBM <-predict(GBM_model, data_to_test, type = "raw")

Predict_Table <-data.frame(predictions_RF,predictions_LDA,predictions_NB)
Predict_Table <-as.matrix(Predict_Table)

voted <- as.character()


for(i in 1:dim(Predict_Table)[1])
  voted[i] <- names(sort(table(Predict_Table[i,]),decreasing = TRUE))[1]

write.csv(voted,"quiz_answers.csv",row.names = FALSE)
```







