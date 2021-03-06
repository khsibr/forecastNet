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

#Use a fixed seed for repreducible results
np.random.seed(1)

# Generate the dataset
# train_x, train_y, test_x, test_y, valid_x, valid_y, period = generate_data(T=2750, period = 50, n_seqs = 4)

# df = pd.read_csv('/Users/ryadhkhisb/Dev/workspaces/m/finance-scrape/LSTNet/data/aapl_15min.csv')
# df = df[[c for c in df.columns if 'aapl_15min' in c]]
df = pd.read_parquet('/Users/ryadhkhisb/Dev/workspaces/m/finance-scrape/data/nasdaq100_15min.parquet')
# df = pd.DataFrame(
#     np.array([
#         np.arange(100), np.arange(100), np.arange(100),
#         np.arange(100).astype(np.float32) / 100,
#         np.arange(100).astype(np.float32) / 100]).transpose(),
#     columns=['c1', 'c2', 'c3', 'open', 'close'])

# df=(df-df.mean())/df.std()

in_seq_length = 8
out_seq_length = 8
shift = 2
train_mean = df.mean()
train_std = df.std()

df = (df - train_mean) / train_std
train_x, train_y, test_x, test_y, valid_x, valid_y = process_data(df.to_numpy(),
                                                                  T_in_seq=in_seq_length,
                                                                  T_out_seq = out_seq_length,
                                                                  shift=shift)
# train_mean = train_x.mean()
# train_std = train_x.std()
#
# train_x = (train_x - train_mean) / train_std
# valid_x = (valid_x - train_mean) / train_std
# test_x = (test_x - train_mean) / train_std


# train_size = int(len(df) * 0.66)
# df_train, df_test = df[0:train_size], df[train_size:len(df)]
#
# train_size = int(len(df_train) * 0.90)
# df_train, df_val = df_train[0:train_size], df_train[train_size:len(df_train)]
# def from_df(df):
#     return df.to_numpy()[np.newaxis, :]  , df.iloc[:,-3:-2].to_numpy()[np.newaxis, :]
#
# train_x, train_y = from_df(df_train)
# valid_x, valid_y = from_df(df_val)
# test_x, test_y = from_df(df_test)

# Model parameters
model_type = 'conv2' #'dense' or 'conv', 'dense2' or 'conv2'

hidden_dim = 24
input_dim = train_x.shape[-1]
output_dim = train_x.shape[-1]
learning_rate = 0.0001
n_epochs=20
batch_size = 32

# Initialise model
fcstnet = forecastNet(in_seq_length=in_seq_length, out_seq_length=out_seq_length, input_dim=input_dim,
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
        plt.plot(np.arange(0, in_seq_length), test_x[:, samples[i], 0],
                 '-o', label='input')
        plt.plot(np.arange(in_seq_length, in_seq_length + out_seq_length), test_y[:, samples[i], 0],
                 '-o', label='data')
        plt.plot(np.arange(in_seq_length, in_seq_length + out_seq_length), s_mean[:, i, 0],
                 '-*', label='forecast')
        plt.fill_between(np.arange(in_seq_length, in_seq_length + out_seq_length),
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
