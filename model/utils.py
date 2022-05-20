def ask_user_for_model(MODEL_LIST):
    print("\nHi, choose a Language Model:\n")
    for i, lm in enumerate(MODEL_LIST):
        print(f"{i+1} - {lm}")
    print("0 - Quit program")

    answer = None

    while True:
        try:
            answer = input()
            answer = int(answer)
            break
        except ValueError:
            print("\nOops!  That was no valid number.  Try again...")

    return answer

def models_initiation_loop(MODEL_LIST, DEFAULTS):
    answer = -1
    while answer not in [i for i in range(0, 8)] :
        answer = ask_user_for_model(MODEL_LIST)

    if answer == 0: return answer
    DEFAULTS['MODEL_NAME'] = MODEL_LIST[answer-1]

def ask_user_for_tasks():
    print("\n1 - Show dataset description",
          "\n2 - Start model fine-tuning",
          "\n3 - Start model predictions",
          "\n4 - Show model metrics",
          "\n0 - Quit program")

    answer = None
    while True:
        try:
            answer = input()
            answer = int(answer)
            break
        except ValueError:
            print("\nOops!  That was no valid number.  Try again...")

    return answer


""" Logging metrics & hyperparams of each experiment in a file """
import datetime
def log_metrics(metrics, DEFAULTS):
    # Append-adds at last
    f = open("log.md", "a+")
    now = datetime.datetime.now()

    header_name = DEFAULTS['DATASET_NAME'] + '-' + str(DEFAULTS['DATA_NUM_LABEL']) + 'L'
    title = f"## {header_name} | {DEFAULTS['MODEL_NAME']}"
    subtitle = f"###### {now.strftime('%Y-%m-%d %H:%M:%S')}"
    section1 = f"#### Hyperparameters"
    table = f"""

| **MAX_SEQ_LEN** | **TRAIN_BATCH_SIZE** | **EVAL_BATCH_SIZE** | **EPOCHS** | **LR** | **OPTIM** | **EVAL_STEPS** |
|:---------------:|:--------------------:|:-------------------:|:----------:|:------:|:---------:|:--------------:|
|        {DEFAULTS['MAX_SEQ_LEN']}       |          {DEFAULTS['TRAIN_BATCH_SIZE']}          |          {DEFAULTS['EVAL_BATCH_SIZE']}         |     {DEFAULTS['EPOCHS']}     |   {DEFAULTS['LR']}   |     {DEFAULTS['OPTIM']}    |       {DEFAULTS['EVAL_STEPS']}       |

    """
    section2 = f"#### Metrics"
    metrics = f"""
```
{metrics}
```
    """
    sep_line = "----"

    f.write(title + "\n")
    f.write(subtitle + "\n")
    f.write(section1 + "\n")
    f.write(table + "\n")
    f.write(section2 + "\n")
    f.write(metrics + "\n")
    f.write(sep_line + "\n")
    f.write(sep_line + "\n")
    f.close()


""" Handling command line args for better & faster executions """
import argparse

def init_args(DEFAULTS):
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional arguments
    parser.add_argument("-r", "--report", help = "Possible values [\"none\", \"wandb\"]. If used all logs during training and evaluation are reported through Wandb API (must be connected with the right credentials).", action="store_true")
    parser.add_argument("-d", "--dataset", help = "Choose dataset by specifying its name: FEVER, MultiFC, Liar...")
    parser.add_argument("-n", "--num_labels", type=int, help = "Specify the number of labels in the dataset. Required if dataset is manually specified (minimum is 2).")
    parser.add_argument("-e", "--epoch", type=int, help = "Specify the number of training epochs.")
    parser.add_argument("-t", "--train_batch", type=int, help = "Specify the size of training batch.")
    parser.add_argument("-v", "--eval_batch", type=int, help = "Specify the size of validation batch.")


    # Read arguments from command line
    args = parser.parse_args()

    print(args)

    if args.report:
        DEFAULTS['REPORT'] = "wandb"

    if args.dataset:
        DEFAULTS['DATASET_NAME'] = args.dataset

    if args.num_labels:
        DEFAULTS['DATA_NUM_LABEL'] = args.num_labels

    if args.epoch:
        DEFAULTS['EPOCHS'] = args.epoch

    if args.train_batch:
        DEFAULTS['TRAIN_BATCH_SIZE'] = args.train_batch

    if args.eval_batch:
        DEFAULTS['EVAL_BATCH_SIZE'] = args.eval_batch
