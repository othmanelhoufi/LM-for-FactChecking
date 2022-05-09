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
import preproc_dataset
import wandb
import datetime
import pathlib
import warnings

from torchinfo import summary


# specify GPU
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('USING DEVICE:', device)
#Additional Info when using cuda
if device.type == 'cuda':
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)


# Transformer model
MODEL_NAME = None

# Dataset name
DATASET_NAME = 'FEVER'
DATA_NUM_LABEL = 2 # minimum 2 labels

# hyperparams
MAX_SEQ_LEN = 128
TRAIN_BATCH_SIZE = 20
EVAL_BATCH_SIZE = 20
EPOCHS = 3
LR = 3e-5
OPTIM = 'adamw_hf'
EVAL_STEPS = 100
SAVE_STEPS = EVAL_STEPS * 3
LOGGING_STEPS = 500
SAVE_TOTAL_LIMIT = 1
EARLY_STOPPING_PATIENCE = 3
REPORT="none"

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
def init_model(num_of_labels, model_name=MODEL_NAME):
    # import pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=MODEL_NAME, num_labels=num_of_labels)
    return model

# Import Tokenizer
def init_tokenizer(model_name=MODEL_NAME):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_NAME)
    return tokenizer



# Tokenization
def init_tokens(X_train, X_val, X_test, tokenizer, max_seq_len=MAX_SEQ_LEN):
    # tokenize and encode sequences in the training set
    tokens_train = tokenizer(X_train, padding=True, truncation=True, max_length=max_seq_len)

    # tokenize and encode sequences in the validation set
    tokens_val = tokenizer(X_val, padding=True, truncation=True, max_length=max_seq_len)

    # tokenize and encode sequences in the test set
    tokens_test = tokenizer(X_test, padding=True, truncation=True, max_length=max_seq_len)

    return tokens_train, tokens_val, tokens_test


# Define Trainer parameters
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    mcc = matthews_corrcoef(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "mcc": mcc}

def log_metrics(metrics):
    # Append-adds at last
    f = open("log.txt", "a+")
    now = datetime.datetime.now()
    header = "#"*20 + "  " + MODEL_NAME + " | " + DATASET_NAME + '-' + str(DATA_NUM_LABEL) + 'L' + " | " + now.strftime("%Y-%m-%d %H:%M:%S") + "  " + "#"*20 + "\n" + "#"*90 + "\n\n"
    hyperparams = {
        'MAX_SEQ_LEN' : MAX_SEQ_LEN,
        'TRAIN_BATCH_SIZE' : TRAIN_BATCH_SIZE,
        'EVAL_BATCH_SIZE' : EVAL_BATCH_SIZE,
        'EPOCHS' : EPOCHS,
        'LR' : LR,
        'OPTIM' : OPTIM,
        'EVAL_STEPS' : EVAL_STEPS
    }

    f.write(header)
    f.write(str(hyperparams) + "\n\n")
    f.write(metrics + "\n")
    f.write("#"*90 + "\n\n")
    f.close()

def ask_user_for_model():
    answer = None
    while True:
        try:
            print("\nHi, choose a Language Model:")
            print("\n1 - bert-base-uncased",
                  "\n2 - roberta-base",
                  "\n3 - albert-base-v2",
                  "\n4 - distilbert-base-uncased",
                  "\n5 - xlnet-base-cased",
                  "\n6 - google/bigbird-roberta-base",
                  "\n7 - YituTech/conv-bert-base")
            answer = input()
            answer = int(answer)
            break
        except ValueError:
            print("Oops!  That was no valid number.  Try again...")

    return answer

def ask_user_for_tasks():
    answer = None
    while True:
        try:
            print("\n1 - Show dataset description",
                  "\n2 - Start model fine-tuning",
                  "\n3 - Start model predictions",
                  # "\n4 - Delete pre-fine-tuned model",
                  # "\n5 - Show learning loss and accuracy",
                  "\n4 - Show model metrics",
                  "\n5 - Quit program")
            answer = input()
            answer = int(answer)
            break
        except ValueError:
            print("Oops!  That was no valid number.  Try again...")

    return answer

def modules_initiation_loop():
    answer = -1
    while answer not in [i for i in range(1, 8)] :
        answer = ask_user_for_model()

    global MODEL_NAME
    if answer == 1:
        MODEL_NAME = 'bert-base-uncased'
    elif answer == 2:
        MODEL_NAME = 'roberta-base'
    elif answer == 3:
        MODEL_NAME = 'albert-base-v2'
    elif answer == 4:
        MODEL_NAME = 'distilbert-base-uncased'
    elif answer == 5:
        MODEL_NAME = 'xlnet-base-cased'
    elif answer == 6:
        MODEL_NAME = 'google/bigbird-roberta-base'
    elif answer == 7:
        MODEL_NAME = 'YituTech/conv-bert-base'

def models_training_loop(EXPERIMENT_NAME):
    print("\n**************** ", MODEL_NAME , "Model ****************\n")

    # Init dataset
    dataset = preproc_dataset.Dataset(name=DATASET_NAME, split_dev=True, num_labels=DATA_NUM_LABEL)
    train_dataset = dataset.get_train_dataset()
    val_dataset = dataset.get_val_dataset()
    test_dataset = dataset.get_test_dataset()
    labels, encoded_labels, decoded_labels = dataset.get_encodings()

    train_text, train_labels = split_dataset(train_dataset)
    val_text, val_labels = split_dataset(val_dataset)
    test_text, test_labels = split_dataset(test_dataset)

    model = None
    tokenizer = init_tokenizer(model_name=MODEL_NAME)
    tokens_train, tokens_val, tokens_test = init_tokens(train_text, val_text, test_text, tokenizer, MAX_SEQ_LEN)

    y_pred = []

    # model_size = model.num_parameters()
    # print("SIZE = ", model_size)

    # summary(model)

    answer = -1
    while answer != 5 :

        answer = ask_user_for_tasks()
        print("\n")

        if(answer == 1):
            dataset.get_describtion()
            dataset.print_data_example()
        elif(answer == 2):
            print("STARTING LEARNING ...\n")
            model = init_model(DATA_NUM_LABEL, model_name=MODEL_NAME)

            train_dataset_torch = Dataset(tokens_train, train_labels)
            val_dataset_torch = Dataset(tokens_val, val_labels)

            # Define Trainer
            args = TrainingArguments(
                output_dir="outputs/" + EXPERIMENT_NAME,
                overwrite_output_dir=True,
                save_strategy='no',
                save_total_limit=SAVE_TOTAL_LIMIT,
                evaluation_strategy='steps',
                eval_steps=EVAL_STEPS,
                save_steps=SAVE_STEPS,
                per_device_train_batch_size=TRAIN_BATCH_SIZE,
                per_device_eval_batch_size=EVAL_BATCH_SIZE,
                num_train_epochs=EPOCHS,
                learning_rate=LR,
                optim=OPTIM,
                seed=0,
                logging_steps=LOGGING_STEPS,
                report_to=REPORT,
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
                callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
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
                model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path, num_labels=DATA_NUM_LABEL)

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
                log_metrics(report)
                print(report)
                # print(classification_report(test_labels, y_pred, target_names=encoded_labels, digits=4))
                # Confusion Matrices
                if REPORT == "wandb":
                    wandb.sklearn.plot_confusion_matrix(test_labels, y_pred, labels)
            else:
                print("START PREDICTIONS FIRST !!\n")

        elif(answer == 5):
            if REPORT == "wandb": wandb.finish()
            print("GOODBYE ...")


def start():

    # logging.set_verbosity_error()
    # warnings.filterwarnings("ignore", category=DeprecationWarning)

    modules_initiation_loop()
    EXPERIMENT_NAME = MODEL_NAME + '-' + DATASET_NAME + '-' + str(DATA_NUM_LABEL) + 'L'

    if REPORT == "wandb":
        # init wandb
        wandb.init(project="LM-for-fact-checking", name=EXPERIMENT_NAME, entity="othmanelhoufi")

    models_training_loop(EXPERIMENT_NAME)



if __name__ == '__main__':
    start()
