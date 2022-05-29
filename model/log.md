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
                                                                                                               
                                                                                                               
| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS
** |                                                                                                           
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:------------
--:|                                                                                                           
|        128       |          10          |          10         |     2     |   2e-05   |     adamw_hf    |    
   100       |                                                                                                 
                                                                                                               
                                                                                                               
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