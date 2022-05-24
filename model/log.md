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
