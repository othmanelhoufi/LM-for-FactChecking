#!/usr/bin/env python3

from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging

import preproc_dataset
from sklearn.model_selection import train_test_split

import torch

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data and eval data
fever = preproc_dataset.FeverDataset()
df = fever.claim_dataset
df = df.sample(500)

train_df, eval_df = train_test_split(df, test_size=0.3)

# print("TRAIN : \n", train_data['label'].value_counts())
# print("EVAL : \n", eval_data['label'].value_counts())

# Preparing

# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=1)

# Create a ClassificationModel
cuda_available = torch.cuda.is_available()

model = ClassificationModel(
    'bert',
    'bert-base-uncased',
    num_labels=3,
    args=model_args,
    use_cuda=cuda_available
)

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

# Make predictions with the model
predictions, raw_outputs = model.predict(["Sam was a Wizard"])
