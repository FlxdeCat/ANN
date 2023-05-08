# Contains the responses of a gas multisensor device deployed on the field in an Italian city (hourly)

# Date: the date the measurement was taken
# Time: the time of day the measurement was taken
# CO(GT): carbon monoxide levels in the air in mg/m^3
# PT08.S1(CO): tin oxide sensor resistance for CO in ppm
# NMHC(GT): non-methane hydrocarbons levels in micrograms/m^3
# C6H6(GT): benzene levels in the air in micrograms/m^3
# PT08.S2(NMHC): tin oxide sensor resistance for NMHC in ppm
# NOx(GT): nitrogen oxides levels in the air in ppb
# PT08.S3(NOx): tungsten oxide sensor resistance for NOx in ppb
# NO2(GT): nitrogen dioxide levels in the air in micrograms/m^3
# PT08.S4(NO2): tungsten oxide sensor resistance for NO2 in micrograms/m^3
# PT08.S5(O3): indium oxide sensor resistance for O3 in ppb
# T: temperature in Celsius
# RH: relative humidity
# AH: absolute humidity

# Output --> CO(GT): 2.3
# The input variables represent the measurements of various air quality parameters at a specific date and time, and the output variable represents the measured concentration of carbon monoxide (CO) at the same date and time.

# You can use this set of inputs and output as a sample to train an RNN model to predict the concentration of CO based on historical measurements of the other input variables.

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def get_data():
    df = pd.read_csv('AirQualityUCI.csv', delimiter=';', usecols=['PT08.S1(CO)','NMHC(GT)','C6H6(GT)','PT08.S2(NMHC)','NOx(GT)','PT08.S3(NOx)','NO2(GT)','PT08.S4(NO2)','PT08.S5(O3)','T','RH','AH','CO(GT)'])
    df.dropna(inplace=True)
    col = ['PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH', 'CO(GT)']
    df[col] = df[col].astype(float)
    return df

dataset = get_data()

input = 12
output = 1
context_unit = 5
time_seq = 3
lr = 0.1
epoch = 1000
batch = 3

train = dataset[:int(len(dataset)*0.7)]
test = dataset[len(train):]

minMaxScaler = MinMaxScaler()
fit_train = minMaxScaler.fit_transform(train)
fit_test = minMaxScaler.fit_transform(test)

cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(context_unit, activation=tf.nn.relu)
cell = tf.compat.v1.nn.rnn_cell.OutputProjectionWrapper(cell, output_size=output, activation=tf.nn.relu)

feature_placeholder = tf.placeholder(tf.float32, [None, time_seq, input])
target_placeholder = tf.placeholder(tf.float32, [None, time_seq, output])
output, _ = tf.compat.v1.nn.dynamic_rnn(cell, feature_placeholder, dtype=tf.float32)
error = tf.reduce_mean((0.5 * (target_placeholder - output)) ** 2)
optimizer = tf.train.AdamOptimizer(lr).minimize(error)

def switch_batch(dataset, batch):
    x = np.zeros([batch, time_seq, input])
    y = np.zeros([batch, time_seq, output])
    for i in range(batch):
        start = np.random.randint(0, len(dataset) - time_seq)
        x[i] = dataset[start:(start + time_seq)]
        y[i] = dataset[(start + 1):(start + 1 + time_seq)]
    return x, y

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1, epoch+1):
        x, y = switch_batch(fit_train, batch)
        train = {
            feature_placeholder: x,
            target_placeholder: y
        }
        sess.run(optimizer, feed_dict=train)
        if(i%50==0):
            loss = sess.run(error, feed_dict=train)
            print(f"Loss = {loss}")
    seed_data = list(fit_test)
    for i in range(len(test)):
        x = np.array(seed_data[-time_seq:]).reshape([1, time_seq, input])
        test_dict = {
            feature_placeholder: x
        }
        predict = sess.run(output, feed_dict=test_dict)
        seed_data.append(predict[0, -1, 0])
    result = minMaxScaler.inverse_transform(np.array(seed_data[-len(test):]).reshape(-1, 1)).reshape([len(test), 1])
    test['prediction'] = result[:,0]
    test.plot()
    plt.show()