#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES']="1"

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
import torch
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import EarlyStoppingCallback
from transformers.utils import logging
from sklearn.metrics import classification_report, confusion_matrix
import tools.preproc_dataset as preproc_dataset
import wandb
import pathlib
import warnings
from torchinfo import summary
from conf import DEFAULT_PARAMS, MODEL_LIST, WANDB_PARAMS
from tools.utils import *

#**************************************************************************************************#
""" Logging GPU specs if it exists """
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('USING DEVICE:', device)
#Additional Info when using cuda
if device.type == 'cuda':
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
#**************************************************************************************************#

#**************************************************************************************************#
""" Initiating default parameters """
DEFAULTS = DEFAULT_PARAMS.copy()
#**************************************************************************************************#


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


# Splitting text and label columns
def split_dataset(dataset):
    # Preprocess data
    X = list(dataset['text'])
    y = list(dataset['label'])
    return X, y


# Import Model
def init_model(num_of_labels=2, name=DEFAULTS['MODEL_NAME']):
    # import pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=name, num_labels=num_of_labels)
    return model

# Import Tokenizer
def init_tokenizer(name=DEFAULTS['MODEL_NAME']):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=name)
    return tokenizer



# Tokenization
def init_tokens(X_train, X_val, X_test, tokenizer, max_seq_len=DEFAULTS['MAX_SEQ_LEN']):
    # tokenize and encode sequences in the training set
    tokens_train = tokenizer(X_train, padding=True, truncation=True, max_length=max_seq_len)

    # tokenize and encode sequences in the validation set
    tokens_val = tokenizer(X_val, padding=True, truncation=True, max_length=max_seq_len)

    # tokenize and encode sequences in the test set
    tokens_test = tokenizer(X_test, padding=True, truncation=True, max_length=max_seq_len)

    return tokens_train, tokens_val, tokens_test


# Define Trainer eval metrics
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    mcc = matthews_corrcoef(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "mcc": mcc}

# Big loop for training, and testing
def model_training_loop(EXPERIMENT_NAME):
    print("\n**************** ", DEFAULTS['MODEL_NAME'] , "Model ****************\n")

    # Init dataset
    dataset = preproc_dataset.Dataset(name=DEFAULTS['DATASET_NAME'], split_dev=True, num_labels=DEFAULTS['DATA_NUM_LABEL'])
    train_dataset = dataset.get_train_dataset()
    val_dataset = dataset.get_val_dataset()
    test_dataset = dataset.get_test_dataset()
    labels, encoded_labels, decoded_labels = dataset.get_encodings()

    train_text, train_labels = split_dataset(train_dataset)
    val_text, val_labels = split_dataset(val_dataset)
    test_text, test_labels = split_dataset(test_dataset)

    model = None
    tokenizer = init_tokenizer(name=DEFAULTS['MODEL_NAME'])
    tokens_train, tokens_val, tokens_test = init_tokens(train_text, val_text, test_text, tokenizer, DEFAULTS['MAX_SEQ_LEN'])

    y_pred = []

    # model_size = model.num_parameters()
    # print("SIZE = ", model_size)

    # summary(model)

    answer = -1
    while answer != 0 :

        answer = ask_user_for_tasks()
        print("\n")

        if(answer == 1):
            dataset.get_describtion()
            dataset.print_data_example()
        elif(answer == 2):
            print("STARTING LEARNING ...\n")
            model = init_model(DEFAULTS['DATA_NUM_LABEL'], name=DEFAULTS['MODEL_NAME'])

            train_dataset_torch = Dataset(tokens_train, train_labels)
            val_dataset_torch = Dataset(tokens_val, val_labels)

            # Define Trainer
            args = TrainingArguments(
                output_dir="outputs/" + EXPERIMENT_NAME,
                overwrite_output_dir=True,
                save_strategy='steps',
                save_total_limit=DEFAULTS['SAVE_TOTAL_LIMIT'],
                evaluation_strategy='steps',
                eval_steps=DEFAULTS['EVAL_STEPS'],
                save_steps=DEFAULTS['SAVE_STEPS'],
                per_device_train_batch_size=DEFAULTS['TRAIN_BATCH_SIZE'],
                per_device_eval_batch_size=DEFAULTS['EVAL_BATCH_SIZE'],
                num_train_epochs=DEFAULTS['EPOCHS'],
                learning_rate=DEFAULTS['LR'],
                optim=DEFAULTS['OPTIM'],
                seed=0,
                logging_steps=DEFAULTS['LOGGING_STEPS'],
                report_to=DEFAULTS['REPORT'],
                load_best_model_at_end=True,
                metric_for_best_model='accuracy',
                dataloader_drop_last=True,
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset_torch,
                eval_dataset=val_dataset_torch,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=DEFAULTS['EARLY_STOPPING_PATIENCE'])],
            )

            # Train pre-trained model
            trainer.train()

        elif(answer == 3):
            print("STARTING PREDICTIONS ...\n")
            test_dataset_torch = Dataset(tokens_test, test_labels)

            model_path = "outputs/" + EXPERIMENT_NAME
            file = pathlib.Path(model_path)

            if model is not None :
                # Define test trainer
                test_trainer = Trainer(model)
                # Make prediction
                raw_pred, _, _ = test_trainer.predict(test_dataset_torch)
                # Preprocess raw predictions
                y_pred = np.argmax(raw_pred, axis=1)

            elif model is None and file.exists() :
                checkpoint_num = input("Enter checkpoint number: ")
                model_path = "outputs/" + EXPERIMENT_NAME + "/checkpoint-" + checkpoint_num
                file = pathlib.Path(model_path)

                while not file.exists() :
                    checkpoint_num = input("Enter checkpoint number: ")
                    model_path = "outputs/" + EXPERIMENT_NAME + "/checkpoint-" + checkpoint_num
                    file = pathlib.Path(model_path)

                # Load trained model
                model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path, num_labels=DEFAULTS['DATA_NUM_LABEL'])

                # Define test trainer
                test_trainer = Trainer(model)
                # Make prediction
                raw_pred, _, _ = test_trainer.predict(test_dataset_torch)
                # Preprocess raw predictions
                y_pred = np.argmax(raw_pred, axis=1)
            else:
                print("TRAIN MODEL FIRST, DUDE!")
                continue


        elif(answer == 4):
            print("MODEL METRICS ...\n")

            if len(y_pred) > 0:
                report = classification_report(test_labels, y_pred, target_names=encoded_labels, digits=2)
                log_metrics(report, DEFAULTS)
                print(report)
                # print(classification_report(test_labels, y_pred, target_names=encoded_labels, digits=4))
                # Confusion Matrices
                if DEFAULTS['REPORT'] == "wandb":
                    wandb.sklearn.plot_confusion_matrix(test_labels, y_pred, labels)
            else:
                print("START PREDICTIONS FIRST !!\n")

        elif(answer == 0):
            if DEFAULTS['REPORT'] == "wandb": wandb.finish()
            break


def start():

    e = models_initiation_loop(MODEL_LIST, DEFAULTS)
    if e != 0:
        EXPERIMENT_NAME = DEFAULTS['MODEL_NAME'] + '-' + DEFAULTS['DATASET_NAME'] + '-' + str(DEFAULTS['DATA_NUM_LABEL']) + 'L'

        if DEFAULTS['REPORT'] == "wandb":
            # init wandb
            wandb.init(project=WANDB_PARAMS['project'], name=EXPERIMENT_NAME, entity=WANDB_PARAMS['entity'])

        model_training_loop(EXPERIMENT_NAME)



if __name__ == '__main__':
    init_args(DEFAULTS)
    start()
