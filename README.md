# NLP for Fact-Checking and Claim Assessment
> A Language Model based approach
### Abstract
As false information and fake news are propagating throughout the internet and social networks, the need of fact-checking operations becomes necessary in order to maintain a truthful digital environment where general information can be reliably exploited whether in politics, finance or other domains.
The need of this online claim assessment comes from the fact that fake news and false information can have a big negative impact on politics, economy (2016 USA Elections) and public health (COVID-19).
A number of solutions have been proposed to deal with this problem and limit the spread of false information, both manual and automatic. Undoubtedly the manual approaches done on websites such as PolitiFact.com, FactCheck.org and Snopes.com don’t construct a viable solution for the long term as the speed and scale of information propagation increase exponentially rendering this manual fact-checking operation where human fact- checkers can’t scale up at the same rate limited and incapable of solving the problem.
Here, we present our contribution in this regard: an automated solution for fact-checking using FEVER dataset as a source of truth and a state of the art language models used today for NLP tasks (BERT, RoBERTa, XLNet...) in order to classify a given claim as Supports, Refutes or Not enough information (NEI). We successfully prove that fine-tuning a LM with the correct settings can achieve an accuracy of 62% and F1-score of 61% which is more advanced than the majority of fact-checking methods that exists today.

### Setup & Execution
For the development of the LMs we used Python programming language and Hugging Face library that provides Transformers APIs to easily download and train state-of-the-art pretrained models. Using pretrained models can reduce your compute costs, carbon footprint, and save time from training a model from scratch.

In order to track training metrics, validation metrics, disk usage, CPU usage and other environment changes during our experiments we used Weights & Biases API (wandb). For each step or epoch we send all metrics and changes to our wandb project profile and visualize everything in real time. This feature can be enabled by changing the global variable REPORT to "wandb".

We created a generic code for each LM in order to provide other developers with an easy plug and play application. If you prefer to fine-tune LMs with different parameters you only change the following global variables in the LM Python file:

```python
# Transformer model
MODEL_NAME = ’bert-base-uncased’
# hyperparams
MAX_SEQ_LEN = 128
TRAIN_BATCH_SIZE = 20
EVAL_BATCH_SIZE = 20
EPOCHS = 3
LR = 3e-5
OPTIM = ’adamw_hf’
SAVE_STEPS = 1000
EVAL_STEPS = 500
SAVE_STRATEGY = ’epoch’
SAVE_TOTAL_LIMIT = 3
EARLY_STOPPING_PATIENCE = 3
REPORT="none"
```
