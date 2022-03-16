#!/usr/bin/env python3
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import preproc_dataset
from sklearn.model_selection import train_test_split
import sklearn
import torch

import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# Get the logger for the huggingface/transformers library.
transformers_logger = logging.getLogger("transformers")

# Set the logging level to error, meaning display errors and worse, but
# don't display any `INFO` logs.
transformers_logger.setLevel(logging.ERROR)

# Preparing train data and eval data
fever = preproc_dataset.FeverDataset()
df = fever.claim_dataset
df = df.sample(60)
# print(df.describe())
# exit()

train_df, eval_df = train_test_split(df, test_size=0.3)
train_df.columns = ["text", "labels"]
eval_df.columns = ["text", "labels"]

# print("TRAIN : \n", train_df['label'].value_counts())
# print("EVAL : \n", eval_df['label'].value_counts())

# Preparing

# Optional model configuration
model_args = ClassificationArgs()
model_args.output_dir = 'outputs/albert_base_v2'
model_args.num_train_epochs = 7
# model_args.train_batch_size = 500
# model_args.optimizer = 'AdamW'
# model_args.learning_rate = 1e-3
# model_args.evaluate_during_training = True
# model_args.evaluate_during_training_steps = 100
# model_args.use_early_stopping = True
# model_args.early_stopping_delta = 0.01
# model_args.early_stopping_metric = 'eval_loss'
# model_args.early_stopping_metric_minimize = True
# model_args.early_stopping_patience = 3
# model_args.use_cached_eval_features = True


# Create a ClassificationModel
cuda_available = torch.cuda.is_available()
print('GPU available : ', cuda_available)

model = ClassificationModel(
    'albert',
    'albert-base-v2',
    num_labels=3,
    args=model_args,
    use_cuda=cuda_available
)

# Train the model
model.train_model(train_df=train_df, eval_df=eval_df, acc=sklearn.metrics.accuracy_score)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

# Make predictions with the model
# predictions, raw_outputs = model.predict(["Sam was a Wizard"])
