#!/usr/bin/env python3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
import torch
from transformers import TrainingArguments, Trainer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import EarlyStoppingCallback
from transformers.utils import logging
from sklearn.metrics import classification_report, confusion_matrix


import preproc_dataset

import wandb

# specify GPU
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('USING DEVICE:', device)
#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

# Transformer model
MODEL_NAME = 'roberta-base'

# hyperparams
MAX_SEQ_LEN = 128
TRAIN_BATCH_SIZE = 20
EVAL_BATCH_SIZE = 20
EPOCHS = 3
LR = 3e-5
OPTIM = 'adamw_hf'
SAVE_STEPS = 4362
EVAL_STEPS = 500
SAVE_STRATEGY = 'epoch'
SAVE_TOTAL_LIMIT = 3
EARLY_STOPPING_PATIENCE = 3
REPORT="none"

if REPORT == "wandb":
    # init wandb
    wandb.init(project="NLP-for-fact-checking-using-FEVER-dataset", name=MODEL_NAME + '-b' + str(TRAIN_BATCH_SIZE), entity="othmanelhoufi")


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


# Import Model and Tokenizer
def init_model(model_name=MODEL_NAME):
    # import pretrained model
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    # Load the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    return model, tokenizer



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


def ask_user():
    print("\n1 - Show dataset description",
          "\n2 - Start model fine-tuning",
          "\n3 - Start model predictions",
          # "\n4 - Delete pre-fine-tuned model",
          # "\n5 - Show learning loss and accuracy",
          "\n4 - Show model metrics",
          "\n5 - Quit program")
    answer = input()
    return int(answer)

def start():
    # logging.set_verbosity(40)
    # logging.enable_progress_bar()

    print("\n**************** ", MODEL_NAME , "Model ****************\n")

    # FEVER dataset
    fever = preproc_dataset.FeverDataset()
    train_dataset = fever.get_train_dataset()
    val_dataset = fever.get_val_dataset()
    test_dataset = fever.get_test_dataset()

    model, tokenizer = init_model(MODEL_NAME)
    train_text, train_labels = split_dataset(train_dataset)
    val_text, val_labels = split_dataset(val_dataset)
    test_text, test_labels = split_dataset(test_dataset)

    tokens_train, tokens_val, tokens_test = init_tokens(train_text, val_text, test_text, tokenizer, MAX_SEQ_LEN)

    y_pred = []

    answer = -1
    while answer != 5 :

        answer = ask_user()
        print("\n")

        if(answer == 1):
            fever.get_describtion()
            fever.print_data_example()
        elif(answer == 2):
            print("STARTING LEARNING ...\n")

            train_dataset_torch = Dataset(tokens_train, train_labels)
            val_dataset_torch = Dataset(tokens_val, val_labels)

            # Define Trainer
            args = TrainingArguments(
                output_dir="outputs/" + MODEL_NAME,
                overwrite_output_dir=True,
                save_strategy=SAVE_STRATEGY,
                save_total_limit=SAVE_TOTAL_LIMIT,
                evaluation_strategy="steps",
                save_steps=SAVE_STEPS,
                eval_steps=EVAL_STEPS,
                per_device_train_batch_size=TRAIN_BATCH_SIZE,
                per_device_eval_batch_size=EVAL_BATCH_SIZE,
                num_train_epochs=EPOCHS,
                learning_rate=LR,
                optim=OPTIM,
                seed=0,
                report_to=REPORT,
                run_name = MODEL_NAME + '-b' + str(TRAIN_BATCH_SIZE),
                load_best_model_at_end=False,
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset_torch,
                eval_dataset=val_dataset_torch,
                compute_metrics=compute_metrics,
                # callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
            )

            # Train pre-trained model
            trainer.train()

        elif(answer == 3):
            print("STARTING PREDICTIONS ...\n")
            checkpoint_num = input("Enter checkpoint number: ")
            test_dataset_torch = Dataset(tokens_test, test_labels)
            # Load trained model
            model_path = "outputs/" + MODEL_NAME + "/checkpoint-" + checkpoint_num
            model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=3)
            # Define test trainer
            test_trainer = Trainer(model)
            # Make prediction
            raw_pred, _, _ = test_trainer.predict(test_dataset_torch)
            # Preprocess raw predictions
            y_pred = np.argmax(raw_pred, axis=1)

        # elif(answer == 4):
        #     print("FINE TUNED MODEL DELETED ...\n")
        # elif(answer == 5):
        #     print("LEARNING LOSS & ACCURACY ...\n")
        elif(answer == 4):
            print("MODEL METRICS ...\n")
            labels = {'REFUTES':0, 'SUPPORTS':1, 'NOT ENOUGH INFO':2}

            if len(y_pred) > 0:
                print(classification_report(test_labels, y_pred, target_names=labels, digits=4))
                # Confusion Matrices
                if REPORT == "wandb":
                    wandb.sklearn.plot_confusion_matrix(test_labels, y_pred, ['REFUTES', 'SUPPORTS', 'NEI'])

            else:
                print("START PREDICTIONS FIRST !!\n")

        elif(answer == 5):
            if REPORT == "wandb": wandb.finish()
            print("GOODBYE ...")


if __name__ == '__main__':
    start()
