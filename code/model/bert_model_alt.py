#!/usr/bin/env python3

import preproc_dataset

import torch
from tqdm import tqdm as tqdm

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import numpy as np


fever = preproc_dataset.FeverDataset()
df = fever.get_train_val(fever.claim_dataset)


# exit()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].claim.values,
    add_special_tokens=True,
    return_attention_mask=True,
    padding=True,
    max_length=256,
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=='val'].claim.values,
    add_special_tokens=True,
    return_attention_mask=True,
    padding=True,
    max_length=256,
    return_tensors='pt'
)


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type=='train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type=='val'].label.values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

encoded_dict = {'REFUTES':0, 'SUPPORTS':1, 'NOT ENOUGH INFO':2}

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(encoded_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

batch_size = 3

dataloader_train = DataLoader(dataset_train,
                              sampler=RandomSampler(dataset_train),
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val,
                                   sampler=SequentialSampler(dataset_val),
                                   batch_size=batch_size)

optimizer = AdamW(model.parameters(),
                  lr=1e-5,
                  eps=1e-8)

epochs = 5

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    encoded_dict_inverse = {v: k for k, v in encoded_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    print('Overall Accuracy : ', accuracy_score(labels_flat, preds_flat))

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {encoded_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')
        print('Accuracy 2 : ', accuracy_score(y_true, y_preds))
        print('Precision : ', precision_score(y_true, y_preds, average='weighted'))
        print('F1 : ', f1_score(y_true, y_preds, average='weighted'))
        print('Recall : ', recall_score(y_true, y_preds, average='weighted'))


import random

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(dataloader_val):

    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals

# for epoch in tqdm(range(1, epochs+1)):
#
#     model.train()
#
#     loss_train_total = 0
#
#     progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
#     for batch in progress_bar:
#
#         model.zero_grad()
#
#         batch = tuple(b.to(device) for b in batch)
#
#         inputs = {'input_ids':      batch[0],
#                   'attention_mask': batch[1],
#                   'labels':         batch[2],
#                  }
#
#         outputs = model(**inputs)
#
#         loss = outputs[0]
#         loss_train_total += loss.item()
#         loss.backward()
#
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#
#         optimizer.step()
#         scheduler.step()
#
#         progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
#
#
#     torch.save(model.state_dict(), f'bin/finetuned_BERT_epoch_{epoch}.model')
#
#     tqdm.write(f'\nEpoch {epoch}')
#
#     loss_train_avg = loss_train_total/len(dataloader_train)
#     tqdm.write(f'Training loss: {loss_train_avg}')
#
#     val_loss, predictions, true_vals = evaluate(dataloader_validation)
#     val_f1 = f1_score_func(predictions, true_vals)
#     tqdm.write(f'Validation loss: {val_loss}')
#     tqdm.write(f'F1 Score (Weighted): {val_f1}')



model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(encoded_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.to(device)

model.load_state_dict(torch.load('bin/finetuned_BERT_epoch_4.model', map_location=torch.device('cpu')))

_, predictions, true_vals = evaluate(dataloader_validation)
accuracy_per_class(predictions, true_vals)
