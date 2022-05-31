# Experimental Logs

## FEVER-2L | bert-base-uncased
###### 2022-05-23 10:55:15
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          20         |     3     |   3e-05   |     adamw_hf    |       400       |


#### Metrics

```
              precision    recall  f1-score   support

    SUPPORTS       0.74      0.90      0.81      3333
     REFUTES       0.87      0.68      0.76      3333

    accuracy                           0.79      6666
   macro avg       0.81      0.79      0.79      6666
weighted avg       0.81      0.79      0.79      6666

```

----
----
## FEVER-2L | roberta-base
###### 2022-05-23 12:10:18
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          20         |     3     |   3e-05   |     adamw_hf    |       400       |


#### Metrics

```
              precision    recall  f1-score   support

    SUPPORTS       0.74      0.89      0.81      3333
     REFUTES       0.86      0.68      0.76      3333

    accuracy                           0.79      6666
   macro avg       0.80      0.79      0.78      6666
weighted avg       0.80      0.79      0.78      6666

```

----
----
## FEVER-2L | albert-base-v2
###### 2022-05-23 13:10:02
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          20         |     3     |   3e-05   |     adamw_hf    |       400       |


#### Metrics

```
              precision    recall  f1-score   support

    SUPPORTS       0.71      0.89      0.79      3333
     REFUTES       0.85      0.64      0.73      3333

    accuracy                           0.76      6666
   macro avg       0.78      0.76      0.76      6666
weighted avg       0.78      0.76      0.76      6666

```

----
----
## FEVER-2L | distilbert-base-uncased
###### 2022-05-23 14:01:06
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          20         |     3     |   3e-05   |     adamw_hf    |       400       |


#### Metrics

```
              precision    recall  f1-score   support

    SUPPORTS       0.74      0.90      0.81      3333
     REFUTES       0.87      0.68      0.76      3333

    accuracy                           0.79      6666
   macro avg       0.80      0.79      0.79      6666
weighted avg       0.80      0.79      0.79      6666

```

----
----
## FEVER-2L | xlnet-base-cased
###### 2022-05-23 15:32:10
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          20         |     3     |   3e-05   |     adamw_hf    |       400       |


#### Metrics

```
              precision    recall  f1-score   support

    SUPPORTS       0.73      0.88      0.80      3333
     REFUTES       0.85      0.68      0.76      3333

    accuracy                           0.78      6666
   macro avg       0.79      0.78      0.78      6666
weighted avg       0.79      0.78      0.78      6666

```

----
----
## FEVER-2L | google/bigbird-roberta-base
###### 2022-05-23 16:53:58
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          20         |     3     |   3e-05   |     adamw_hf    |       400       |


#### Metrics

```
              precision    recall  f1-score   support

    SUPPORTS       0.71      0.90      0.79      3333
     REFUTES       0.86      0.63      0.73      3333

    accuracy                           0.76      6666
   macro avg       0.78      0.76      0.76      6666
weighted avg       0.78      0.76      0.76      6666

```

----
----
## FEVER-2L | YituTech/conv-bert-base
###### 2022-05-23 18:31:44
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          20         |     3     |   3e-05   |     adamw_hf    |       400       |


#### Metrics

```
              precision    recall  f1-score   support

    SUPPORTS       0.77      0.88      0.82      3333
     REFUTES       0.86      0.74      0.80      3333

    accuracy                           0.81      6666
   macro avg       0.82      0.81      0.81      6666
weighted avg       0.82      0.81      0.81      6666

```

----
----
## FEVER-3L | bert-base-uncased
###### 2022-05-24 10:25:35
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          30          |          30         |     3     |   3e-05   |     adamw_hf    |       400       |


#### Metrics

```
                 precision    recall  f1-score   support

       SUPPORTS       0.54      0.79      0.65      3333
        REFUTES       0.74      0.60      0.66      3333
NOT ENOUGH INFO       0.62      0.46      0.53      3333

       accuracy                           0.62      9999
      macro avg       0.64      0.62      0.61      9999
   weighted avg       0.64      0.62      0.61      9999

```

----
----
## FEVER-3L | roberta-base
###### 2022-05-24 11:46:39
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          30          |          30         |     3     |   3e-05   |     adamw_hf    |       400       |


#### Metrics

```
                 precision    recall  f1-score   support

       SUPPORTS       0.54      0.81      0.65      3333
        REFUTES       0.74      0.59      0.66      3333
NOT ENOUGH INFO       0.63      0.44      0.52      3333

       accuracy                           0.61      9999
      macro avg       0.64      0.61      0.61      9999
   weighted avg       0.64      0.61      0.61      9999

```

----
----
## FEVER-3L | albert-base-v2
###### 2022-05-24 13:54:56
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          30          |          30         |     3     |   3e-05   |     adamw_hf    |       400       |


#### Metrics

```
                 precision    recall  f1-score   support

       SUPPORTS       0.53      0.76      0.63      3333
        REFUTES       0.75      0.55      0.63      3333
NOT ENOUGH INFO       0.57      0.48      0.52      3333

       accuracy                           0.60      9999
      macro avg       0.62      0.60      0.59      9999
   weighted avg       0.62      0.60      0.59      9999

```

----
----
## FEVER-3L | distilbert-base-uncased
###### 2022-05-24 15:08:05
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          30          |          30         |     3     |   3e-05   |     adamw_hf    |       400       |


#### Metrics

```
                 precision    recall  f1-score   support

       SUPPORTS       0.54      0.80      0.65      3333
        REFUTES       0.75      0.59      0.66      3333
NOT ENOUGH INFO       0.62      0.45      0.52      3333

       accuracy                           0.61      9999
      macro avg       0.63      0.61      0.61      9999
   weighted avg       0.63      0.61      0.61      9999

```

----
----
## FEVER-3L | xlnet-base-cased
###### 2022-05-24 16:57:21
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          30          |          30         |     3     |   3e-05   |     adamw_hf    |       400       |


#### Metrics

```
                 precision    recall  f1-score   support

       SUPPORTS       0.53      0.82      0.64      3333
        REFUTES       0.74      0.59      0.66      3333
NOT ENOUGH INFO       0.64      0.42      0.50      3333

       accuracy                           0.61      9999
      macro avg       0.64      0.61      0.60      9999
   weighted avg       0.64      0.61      0.60      9999

```

----
----
## FEVER-3L | google/bigbird-roberta-base
###### 2022-05-24 20:01:59
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          30          |          30         |     3     |   3e-05   |     adamw_hf    |       400       |


#### Metrics

```
                 precision    recall  f1-score   support

       SUPPORTS       0.54      0.78      0.64      3333
        REFUTES       0.75      0.57      0.65      3333
NOT ENOUGH INFO       0.61      0.47      0.53      3333

       accuracy                           0.61      9999
      macro avg       0.63      0.61      0.61      9999
   weighted avg       0.63      0.61      0.61      9999

```

----
----
## FEVER-3L | YituTech/conv-bert-base
###### 2022-05-24 21:35:32
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          30          |          30         |     3     |   3e-05   |     adamw_hf    |       400       |


#### Metrics

```
                 precision    recall  f1-score   support

       SUPPORTS       0.57      0.79      0.67      3333
        REFUTES       0.76      0.62      0.68      3333
NOT ENOUGH INFO       0.61      0.50      0.55      3333

       accuracy                           0.64      9999
      macro avg       0.65      0.64      0.63      9999
   weighted avg       0.65      0.64      0.63      9999

```

----
----
## MultiFC-2L | bert-base-uncased                                                                              
###### 2022-05-27 17:06:24                                                                                     
#### Hyperparameters                                                                                           


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS
** |                                                                                                           
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:------------
--:|                                                                                                           
|        128       |          20          |          20         |     3     |   3e-05   |     adamw_hf    |    
   300       |                                                                                                 


#### Metrics                                                                                                   

```                                                                                                            
              precision    recall  f1-score   support                                                          

        TRUE       0.52      0.44      0.47       438                                                          
       FALSE       0.77      0.82      0.80      1010                                                          

    accuracy                           0.71      1448                                                          
   macro avg       0.64      0.63      0.63      1448                                                          
weighted avg       0.69      0.71      0.70      1448                                                          

```                                                                                                            

----                                                                                                           
----   
## MultiFC-2L | roberta-base                                                                                   
###### 2022-05-27 17:21:51                                                                                     
#### Hyperparameters                                                                                           


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS
** |                                                                                                           
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:------------
--:|                                                                                                           
|        128       |          20          |          20         |     3     |   3e-05   |     adamw_hf    |    
   300       |                                                                                                 


#### Metrics                                                                                                   

```                                                                                                            
              precision    recall  f1-score   support                                                          

        TRUE       0.54      0.43      0.48       438                                                          
       FALSE       0.77      0.84      0.81      1010                                                          

    accuracy                           0.72      1448                                                          
   macro avg       0.66      0.63      0.64      1448                                                          
weighted avg       0.70      0.72      0.71      1448                                                          

```                                                                                                            

----                                                                                                           
----
## MultiFC-2L | albert-base-v2
###### 2022-05-29 01:54:36
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          20         |     3     |   3e-05   |     adamw_hf    |       300       |


#### Metrics

```
              precision    recall  f1-score   support

        TRUE       0.52      0.21      0.30       438
       FALSE       0.73      0.92      0.81      1010

    accuracy                           0.70      1448
   macro avg       0.63      0.56      0.56      1448
weighted avg       0.67      0.70      0.66      1448

```

----
----
## MultiFC-2L | distilbert-base-uncased                                                                        
###### 2022-05-27 17:48:32                                                                                     
#### Hyperparameters                                                                                           


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS
** |                                                                                                           
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:------------
--:|                                                                                                           
|        128       |          20          |          20         |     3     |   3e-05   |     adamw_hf    |    
   300       |                                                                                                 


#### Metrics                                                                                                   

```                                                                                                            
              precision    recall  f1-score   support                                                          

        TRUE       0.54      0.42      0.47       438                                                          
       FALSE       0.77      0.84      0.81      1010                                                          

    accuracy                           0.72      1448                                                          
   macro avg       0.66      0.63      0.64      1448                                                          
weighted avg       0.70      0.72      0.71      1448                                                          

```                                                                                                            

----                                                                                                           
----  
## MultiFC-2L | xlnet-base-cased                                                                               
###### 2022-05-27 18:09:49                                                                                     
#### Hyperparameters                                                                                           


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS
** |                                                                                                           
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:------------
--:|                                                                                                           
|        128       |          20          |          20         |     3     |   3e-05   |     adamw_hf    |    
   300       |                                                                                                 


#### Metrics                                                                                                   

```                                                                                                            
              precision    recall  f1-score   support                                                          

        TRUE       0.53      0.47      0.50       438                                                          
       FALSE       0.78      0.82      0.80      1010                                                          

    accuracy                           0.71      1448                                                          
   macro avg       0.65      0.64      0.65      1448                                                          
weighted avg       0.70      0.71      0.71      1448                                                          

```                                                                                                            

----                                                                                                           
----  
## MultiFC-2L | google/bigbird-roberta-base                                                                    
###### 2022-05-28 14:47:48                                                                                     
#### Hyperparameters                                                                                           


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS
** |                                                                                                           
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:------------
--:|                                                                                                           
|        128       |          20          |          20         |     3     |   3e-05   |     adamw_hf    |    
   300       |                                                                                                 


#### Metrics                                                                                                   

```                                                                                                            
              precision    recall  f1-score   support                                                          

        TRUE       0.51      0.38      0.43       438                                                          
       FALSE       0.76      0.84      0.80      1010                                                          

    accuracy                           0.70      1448
   macro avg       0.63      0.61      0.61      1448
weighted avg       0.68      0.70      0.69      1448

```

----
----
## MultiFC-2L | YituTech/conv-bert-base
###### 2022-05-28 15:11:28
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          20         |     3     |   3e-05   |     adamw_hf    |    
   300       |


#### Metrics

```
              precision    recall  f1-score   support

        TRUE       0.55      0.35      0.43       438
       FALSE       0.76      0.88      0.81      1010

    accuracy                           0.72      1448
   macro avg       0.65      0.61      0.62      1448
weighted avg       0.69      0.72      0.70      1448

```

----
----
## Liar-2L | bert-base-uncased
###### 2022-05-29 17:30:54
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          10          |          10         |     2     |   2e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

       FALSE       0.71      0.88      0.78       823
        TRUE       0.61      0.35      0.45       460

    accuracy                           0.69      1283
   macro avg       0.66      0.61      0.61      1283
weighted avg       0.67      0.69      0.66      1283

```

----
----
## Liar-2L | roberta-base
###### 2022-05-29 17:41:14
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          10          |          10         |     2     |   2e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

       FALSE       0.73      0.78      0.76       823
        TRUE       0.56      0.49      0.52       460

    accuracy                           0.68      1283
   macro avg       0.65      0.64      0.64      1283
weighted avg       0.67      0.68      0.67      1283

```

----
----
## Liar-2L | albert-base-v2
###### 2022-05-29 18:10:46
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          10          |          10         |     2     |   2e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

       FALSE       0.68      0.91      0.78       823
        TRUE       0.60      0.23      0.34       460

    accuracy                           0.67      1283
   macro avg       0.64      0.57      0.56      1283
weighted avg       0.65      0.67      0.62      1283

```

----
----
## Liar-2L | distilbert-base-uncased
###### 2022-05-29 18:21:15
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          10          |          10         |     2     |   2e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

       FALSE       0.71      0.80      0.75       823
        TRUE       0.54      0.41      0.47       460

    accuracy                           0.66      1283
   macro avg       0.62      0.61      0.61      1283
weighted avg       0.65      0.66      0.65      1283

```

----
----
## Liar-2L | xlnet-base-cased
###### 2022-05-29 18:39:56
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          10          |          10         |     2     |   2e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

       FALSE       0.70      0.87      0.77       823
        TRUE       0.58      0.32      0.42       460

    accuracy                           0.67      1283
   macro avg       0.64      0.60      0.59      1283
weighted avg       0.65      0.67      0.64      1283

```

----
----
## Liar-2L | google/bigbird-roberta-base
###### 2022-05-29 18:55:58
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          10          |          10         |     2     |   2e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

       FALSE       0.69      0.92      0.79       823
        TRUE       0.64      0.25      0.35       460

    accuracy                           0.68      1283
   macro avg       0.66      0.58      0.57      1283
weighted avg       0.67      0.68      0.63      1283

```

----
----
## Liar-2L | YituTech/conv-bert-base
###### 2022-05-29 19:19:56
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          10          |          10         |     2     |   2e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

       FALSE       0.73      0.77      0.75       823
        TRUE       0.55      0.50      0.52       460

    accuracy                           0.67      1283
   macro avg       0.64      0.63      0.63      1283
weighted avg       0.66      0.67      0.67      1283

```

----
----
## MultiFC-5L | bert-base-uncased
###### 2022-05-30 09:25:03
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          10         |     2     |   3e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

        TRUE       0.42      0.38      0.40       261
       FALSE       0.63      0.71      0.67       633
MOSTLY-FALSE       0.35      0.10      0.15       168
 MOSTLY-TRUE       0.28      0.28      0.28       177
  IN-BETWEEN       0.28      0.38      0.32       209

    accuracy                           0.48      1448
   macro avg       0.39      0.37      0.36      1448
weighted avg       0.47      0.48      0.46      1448

```

----
----
## MultiFC-5L | roberta-base
###### 2022-05-30 09:47:19
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          10         |     2     |   3e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

        TRUE       0.49      0.38      0.43       261
       FALSE       0.63      0.74      0.68       633
MOSTLY-FALSE       0.27      0.07      0.11       168
 MOSTLY-TRUE       0.30      0.32      0.31       177
  IN-BETWEEN       0.28      0.36      0.32       209

    accuracy                           0.49      1448
   macro avg       0.39      0.37      0.37      1448
weighted avg       0.47      0.49      0.47      1448

```

----
----
## MultiFC-5L | albert-base-v2
###### 2022-05-30 10:23:43
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          10         |     2     |   3e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

        TRUE       0.54      0.12      0.20       261
       FALSE       0.51      0.88      0.65       633
MOSTLY-FALSE       0.00      0.00      0.00       168
 MOSTLY-TRUE       0.27      0.41      0.33       177
  IN-BETWEEN       0.35      0.07      0.11       209

    accuracy                           0.47      1448
   macro avg       0.34      0.29      0.26      1448
weighted avg       0.41      0.47      0.38      1448

```

----
----
## MultiFC-5L | distilbert-base-uncased
###### 2022-05-30 10:40:16
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          10         |     2     |   3e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

        TRUE       0.49      0.33      0.39       261
       FALSE       0.61      0.74      0.67       633
MOSTLY-FALSE       0.29      0.05      0.09       168
 MOSTLY-TRUE       0.27      0.32      0.30       177
  IN-BETWEEN       0.24      0.30      0.27       209

    accuracy                           0.47      1448
   macro avg       0.38      0.35      0.34      1448
weighted avg       0.46      0.47      0.45      1448

```

----
----
## MultiFC-5L | xlnet-base-cased
###### 2022-05-30 11:10:58
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          10         |     2     |   3e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

        TRUE       0.42      0.37      0.39       261
       FALSE       0.62      0.73      0.67       633
MOSTLY-FALSE       0.33      0.08      0.13       168
 MOSTLY-TRUE       0.26      0.23      0.24       177
  IN-BETWEEN       0.25      0.33      0.29       209

    accuracy                           0.47      1448
   macro avg       0.38      0.35      0.34      1448
weighted avg       0.45      0.47      0.45      1448

```

----
----
## MultiFC-5L | google/bigbird-roberta-base
###### 2022-05-30 11:35:55
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          10         |     2     |   3e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

        TRUE       0.46      0.15      0.22       261
       FALSE       0.59      0.78      0.67       633
MOSTLY-FALSE       0.26      0.07      0.10       168
 MOSTLY-TRUE       0.25      0.32      0.28       177
  IN-BETWEEN       0.26      0.32      0.29       209

    accuracy                           0.46      1448
   macro avg       0.36      0.33      0.31      1448
weighted avg       0.44      0.46      0.42      1448

```

----
----
## MultiFC-5L | YituTech/conv-bert-base
###### 2022-05-30 11:58:11
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          10         |     2     |   3e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

        TRUE       0.48      0.48      0.48       261
       FALSE       0.65      0.70      0.68       633
MOSTLY-FALSE       0.33      0.19      0.24       168
 MOSTLY-TRUE       0.33      0.41      0.36       177
  IN-BETWEEN       0.27      0.24      0.25       209

    accuracy                           0.50      1448
   macro avg       0.41      0.40      0.40      1448
weighted avg       0.49      0.50      0.49      1448

```

----
----
## Liar-6L | bert-base-uncased
###### 2022-05-30 14:14:24
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          10         |     2     |   3e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

   HALF-TRUE       0.28      0.36      0.31       267
 MOSTLY-TRUE       0.28      0.34      0.31       249
       FALSE       0.30      0.38      0.33       250
        TRUE       0.34      0.17      0.22       211
 BARELY-TRUE       0.29      0.18      0.22       214
  PANTS-FIRE       0.34      0.29      0.31        92

    accuracy                           0.29      1283
   macro avg       0.30      0.29      0.29      1283
weighted avg       0.30      0.29      0.29      1283

```

----
----
## Liar-6L | roberta-base
###### 2022-05-30 14:44:13
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          10         |     2     |   3e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

   HALF-TRUE       0.29      0.33      0.31       267
 MOSTLY-TRUE       0.30      0.43      0.35       249
       FALSE       0.31      0.34      0.32       250
        TRUE       0.33      0.20      0.25       211
 BARELY-TRUE       0.36      0.27      0.31       214
  PANTS-FIRE       0.34      0.20      0.25        92

    accuracy                           0.31      1283
   macro avg       0.32      0.29      0.30      1283
weighted avg       0.32      0.31      0.31      1283

```

----
----
## Liar-6L | albert-base-v2
###### 2022-05-30 14:53:15
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          10         |     2     |   3e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

   HALF-TRUE       0.20      0.28      0.24       267
 MOSTLY-TRUE       0.28      0.41      0.33       249
       FALSE       0.25      0.53      0.34       250
        TRUE       0.00      0.00      0.00       211
 BARELY-TRUE       0.25      0.01      0.03       214
  PANTS-FIRE       0.00      0.00      0.00        92

    accuracy                           0.24      1283
   macro avg       0.16      0.21      0.16      1283
weighted avg       0.19      0.24      0.18      1283

```

----
----
## Liar-6L | distilbert-base-uncased
###### 2022-05-30 15:01:37
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          10         |     2     |   3e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

   HALF-TRUE       0.28      0.34      0.31       267
 MOSTLY-TRUE       0.28      0.36      0.32       249
       FALSE       0.27      0.32      0.30       250
        TRUE       0.34      0.23      0.28       211
 BARELY-TRUE       0.30      0.23      0.26       214
  PANTS-FIRE       0.22      0.09      0.12        92

    accuracy                           0.29      1283
   macro avg       0.28      0.26      0.26      1283
weighted avg       0.29      0.29      0.28      1283

```

----
----
## Liar-6L | xlnet-base-cased
###### 2022-05-30 15:21:10
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          10         |     2     |   3e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

   HALF-TRUE       0.26      0.30      0.28       267
 MOSTLY-TRUE       0.28      0.40      0.33       249
       FALSE       0.28      0.41      0.33       250
        TRUE       0.30      0.10      0.15       211
 BARELY-TRUE       0.23      0.18      0.21       214
  PANTS-FIRE       0.45      0.05      0.10        92

    accuracy                           0.27      1283
   macro avg       0.30      0.24      0.23      1283
weighted avg       0.28      0.27      0.25      1283

```

----
----
## Liar-6L | google/bigbird-roberta-base
###### 2022-05-30 15:33:22
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          10         |     2     |   3e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

   HALF-TRUE       0.25      0.40      0.31       267
 MOSTLY-TRUE       0.28      0.36      0.32       249
       FALSE       0.30      0.36      0.33       250
        TRUE       0.29      0.16      0.21       211
 BARELY-TRUE       0.25      0.10      0.14       214
  PANTS-FIRE       0.44      0.12      0.19        92

    accuracy                           0.28      1283
   macro avg       0.30      0.25      0.25      1283
weighted avg       0.28      0.28      0.26      1283

```

----
----
## Liar-6L | YituTech/conv-bert-base
###### 2022-05-30 15:43:11
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          10         |     2     |   3e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

   HALF-TRUE       0.26      0.30      0.28       267
 MOSTLY-TRUE       0.28      0.34      0.31       249
       FALSE       0.31      0.38      0.34       250
        TRUE       0.30      0.23      0.26       211
 BARELY-TRUE       0.31      0.23      0.26       214
  PANTS-FIRE       0.31      0.16      0.21        92

    accuracy                           0.29      1283
   macro avg       0.30      0.27      0.28      1283
weighted avg       0.29      0.29      0.29      1283

```

----
----
## COVID19-2L | bert-base-uncased
###### 2022-05-31 11:42:32
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          20         |     3     |   3e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

        REAL       0.97      0.99      0.98       552
        FAKE       0.99      0.96      0.97       518

    accuracy                           0.98      1070
   macro avg       0.98      0.98      0.98      1070
weighted avg       0.98      0.98      0.98      1070

```

----
----
## COVID19-2L | roberta-base
###### 2022-05-31 11:53:06
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          20         |     3     |   3e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

        REAL       0.96      0.99      0.98       552
        FAKE       0.99      0.96      0.97       518

    accuracy                           0.97      1070
   macro avg       0.97      0.97      0.97      1070
weighted avg       0.97      0.97      0.97      1070

```

----
----
## COVID19-2L | albert-base-v2                                                                                                                               
###### 2022-05-31 12:50:27                                                                                                                                   
#### Hyperparameters                                                                                                                                         


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |                                          
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|                                          
|        128       |          20          |          20         |     3     |   3e-05   |     adamw_hf    |       100       |                                


#### Metrics

```
              precision    recall  f1-score   support

        REAL       0.96      0.99      0.97       552
        FAKE       0.99      0.96      0.97       518

    accuracy                           0.97      1070
   macro avg       0.97      0.97      0.97      1070
weighted avg       0.97      0.97      0.97      1070

```

----
----
## COVID19-2L | distilbert-base-uncased
###### 2022-05-31 12:55:12
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          20         |     3     |   3e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

        REAL       0.96      0.99      0.98       552
        FAKE       0.99      0.96      0.97       518

    accuracy                           0.98      1070
   macro avg       0.98      0.98      0.98      1070
weighted avg       0.98      0.98      0.98      1070

```

----
----
## COVID19-2L | xlnet-base-cased
###### 2022-05-31 13:03:20
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          20         |     3     |   3e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

        REAL       0.98      0.97      0.98       552
        FAKE       0.97      0.98      0.98       518

    accuracy                           0.98      1070
   macro avg       0.98      0.98      0.98      1070
weighted avg       0.98      0.98      0.98      1070

```

----
----
## COVID19-2L | google/bigbird-roberta-base
###### 2022-05-31 13:22:55
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          20         |     3     |   3e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

        REAL       0.96      0.98      0.97       552
        FAKE       0.98      0.95      0.97       518

    accuracy                           0.97      1070
   macro avg       0.97      0.97      0.97      1070
weighted avg       0.97      0.97      0.97      1070

```

----
----
## COVID19-2L | YituTech/conv-bert-base
###### 2022-05-31 13:31:19
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          20         |     3     |   3e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

        REAL       0.96      0.98      0.97       552
        FAKE       0.98      0.96      0.97       518

    accuracy                           0.97      1070
   macro avg       0.97      0.97      0.97      1070
weighted avg       0.97      0.97      0.97      1070

```

----
----
<!-- ## COVID19-2L | bert-base-uncased
###### 2022-05-31 14:09:32
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          20          |          20         |     3     |   3e-05   |     adamw_hf    |       100       |


#### Metrics

```
              precision    recall  f1-score   support

        REAL    0.97776   0.98125   0.97950      1120
        FAKE    0.97933   0.97549   0.97741      1020

    accuracy                        0.97850      2140
   macro avg    0.97854   0.97837   0.97845      2140
weighted avg    0.97851   0.97850   0.97850      2140

```

----
---- -->
