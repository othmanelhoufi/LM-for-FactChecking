## Liar-2L | distilbert-base-uncased
###### 2022-05-20 14:41:04
#### Hyperparameters


| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        128       |          10          |          10         |     1     |   3e-05   |     adamw_hf    |       100       |

    
#### Metrics
              precision    recall  f1-score   support

       FALSE       0.60      1.00      0.75         6
        TRUE       0.00      0.00      0.00         4

    accuracy                           0.60        10
   macro_avg       0.30      0.50      0.37        10
weighted avg       0.36      0.60      0.45        10

----
----
