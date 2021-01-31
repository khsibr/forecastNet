"""
This script demonstrates initialisation, training, evaluation, and forecasting of ForecastNet. The dataset used for the
time-invariance test in section 6.1 of the ForecastNet paper is used for this demonstration.

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""

import numpy as np
import matplotlib.pyplot as plt
from forecastNet import forecastNet
from train import train
from evaluate import evaluate
from dataHelpers import generate_data, process_data
import pandas as pd

from window_generator import WindowGenerator
import os

OUT_STEPS = 48
shift = 24
input_width = 48

data_path = os.environ.get('DATA_PATH') or '/tmp/data/'
model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'
d15 = pd.read_parquet('/Users/ryadhkhisb/Dev/workspaces/m/finance-scrape/data/nasdaq100_15min.parquet')

column_indices = {name: i for i, name in enumerate(d15.columns)}

d15['close'] = d15['last']

n = len(d15)
train_df = d15[0:int(n * 0.7)]
val_df = d15[int(n * 0.7):int(n * 0.9)]
test_df = d15[int(n * 0.9):]

num_features = d15.shape[1]

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std


label_columns = train_df.columns.tolist()
num_labels = len(label_columns)
wide_window = WindowGenerator(input_width=input_width, label_width=OUT_STEPS, shift=shift, label_columns=label_columns,
                              train_df=train_df, val_df=val_df, test_df=test_df)

test_data = [x for x in wide_window.test.as_numpy_iterator()]
train_data = [x for x in wide_window.train.as_numpy_iterator()]
valid_data = [x for x in wide_window.val.as_numpy_iterator()]
test_y = np.concatenate(np.array([x[1] for x in test_data]))
test_x = np.concatenate(np.array([x[0] for x in test_data]))
train_y = np.concatenate(np.array([x[1] for x in train_data]))
train_x = np.concatenate(np.array([x[0] for x in train_data]))
valid_y = np.concatenate(np.array([x[1] for x in valid_data]))
valid_x = np.concatenate(np.array([x[0] for x in valid_data]))

test_y = np.swapaxes(test_y, 1, 0)
test_x = np.swapaxes(test_x, 1, 0)
train_y = np.swapaxes(train_y, 1, 0)
train_x = np.swapaxes(train_x, 1, 0)
valid_y = np.swapaxes(valid_y, 1, 0)
valid_x = np.swapaxes(valid_x, 1, 0)


# Model parameters
model_type = 'conv' #'dense' or 'conv', 'dense2' or 'conv2'

hidden_dim = 24
input_dim = train_x.shape[-1]
output_dim = train_x.shape[-1]
learning_rate = 0.0001
n_epochs=2
batch_size = 64

# Initialise model
fcstnet = forecastNet(in_seq_length=input_width, out_seq_length=input_width, input_dim=input_dim,
                    hidden_dim=hidden_dim, output_dim=output_dim, model_type = model_type, batch_size = batch_size,
                    n_epochs = n_epochs, learning_rate = learning_rate, save_file = './forecastnet.pt')

# Train the model
training_costs, validation_costs = train(fcstnet, train_x, train_y, valid_x, valid_y, restore_session=False)
# Plot the training curves
plt.figure()
plt.plot(training_costs)
plt.plot(validation_costs)

# Evaluate the model
mase, smape, nrmse = evaluate(fcstnet, test_x, test_y, return_lists=False)
print('')
print('MASE:', mase)
print('SMAPE:', smape)
print('NRMSE:', nrmse)

# Generate and plot forecasts for various samples from the test dataset
samples = [0, 10, 20]
# Models with a Gaussian Mixture Density Component output
if model_type == 'dense' or model_type == 'conv':
    # Generate a set of n_samples forecasts (Monte Carlo Forecasts)
    num_forecasts = 10
    y_pred = np.zeros((test_y.shape[0], len(samples), test_y.shape[2], num_forecasts))
    mu = np.zeros((test_y.shape[0], len(samples), test_y.shape[2], num_forecasts))
    sigma = np.zeros((test_y.shape[0], len(samples), test_y.shape[2], num_forecasts))
    for i in range(num_forecasts):
        y_pred[:, :, :, i], mu[:, :, :, i], sigma[:, :, :, i] = fcstnet.forecast(test_x[:, samples, :])
    s_mean = np.mean(y_pred, axis=3)
    s_std = np.std(y_pred, axis=3)
    botVarLine = s_mean - s_std
    topVarLine = s_mean + s_std

    for i in range(len(samples)):
        plt.figure()
        plt.plot(np.arange(0, input_width), test_x[:, samples[i], 0],
                 '-o', label='input')
        plt.plot(np.arange(input_width, input_width + input_width), test_y[:, samples[i], 0],
                 '-o', label='data')
        plt.plot(np.arange(input_width, input_width + input_width), s_mean[:, i, 0],
                 '-*', label='forecast')
        plt.fill_between(np.arange(input_width, input_width + input_width),
                         botVarLine[:, i, 0], topVarLine[:, i, 0],
                         color='gray', alpha=0.3, label='Uncertainty')
        plt.legend()
# Models with a linear output
elif model_type == 'dense2' or model_type == 'conv2':
    # Generate a forecast
    y_pred = fcstnet.forecast(test_x[:,samples,:])

    for i in range(len(samples)):
        # Plot the forecast
        plt.figure()
        plt.plot(np.arange(0, fcstnet.in_seq_length),
                 test_x[:, samples[i], 0],
                 'o-', label='test_data')
        plt.plot(np.arange(fcstnet.in_seq_length, fcstnet.in_seq_length + fcstnet.out_seq_length),
                 test_y[:, samples[i], 0],
                 'o-')
        plt.plot(np.arange(fcstnet.in_seq_length, fcstnet.in_seq_length + fcstnet.out_seq_length),
                 y_pred[:, i, 0],
                 '*-', linewidth=0.7, label='mean')

plt.show()
