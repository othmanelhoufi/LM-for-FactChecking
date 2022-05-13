DEFAULT_PARAMS = {
    # Transformer model
    'MODEL_NAME' : None,

    # Dataset name
    'DATASET_NAME' : 'FEVER',
    'DATA_NUM_LABEL' : 2, # minimum 2 labels

    # hyperparams
    'MAX_SEQ_LEN' : 128,
    'TRAIN_BATCH_SIZE' : 20,
    'EVAL_BATCH_SIZE' : 20,
    'EPOCHS' : 3,
    'LR' : 3e-5,
    'OPTIM' : 'adamw_hf',

    # Huggingface Trainer params
    'EVAL_STEPS' : 100,
    'SAVE_STEPS' : 100,
    'LOGGING_STEPS' : 100,
    'SAVE_TOTAL_LIMIT' : 1,
    'EARLY_STOPPING_PATIENCE' : 3,
    'REPORT':'none',
}

WANDB_PARAMS = {
    'project' : 'LM-for-fact-checking',
    'entity' : 'othmanelhoufi',
}

""" Here you can add more LMs to the list for more experiments """
MODEL_LIST = [
    'bert-base-uncased',
    'roberta-base',
    'albert-base-v2',
    'distilbert-base-uncased',
    'xlnet-base-cased',
    'google/bigbird-roberta-base',
    'YituTech/conv-bert-base'
]
