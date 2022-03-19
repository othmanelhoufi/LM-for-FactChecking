#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt


df_raw = pd.read_csv('./training_progress_scores.csv')
train_loss = df_raw['train_loss']
eval_loss = df_raw['eval_loss']
steps = df_raw['global_step']
acc = df_raw['acc']
mcc = df_raw['mcc']

plt.plot(steps, train_loss, label = "train_loss")
plt.plot(steps, eval_loss, label = "eval_loss")
plt.legend()
plt.show()

plt.plot(steps, acc, label = "acc")
plt.plot(steps, mcc, label = "mcc")
plt.legend()
plt.show()
