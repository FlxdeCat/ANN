import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def get_data():
    df = pd.read_csv('AirQualityUCI.csv', delimiter=';', usecols=['PT08.S1(CO)','NMHC(GT)','C6H6(GT)','PT08.S2(NMHC)','NOx(GT)','PT08.S3(NOx)','NO2(GT)','PT08.S4(NO2)','PT08.S5(O3)','T','RH','AH','CO(GT)'])
    df.dropna(inplace=True)
    col = ['PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH', 'CO(GT)']
    df[col] = df[col].astype(float)
    return df
  
epoch = 1000
batch_size = 10
learning_rate = 0.01
context_unit = 3
layers = {
  'input': 12,
  'output': 1
}

dataset = get_data()
feature = dataset.drop('CO(GT)', axis=1).values
target = dataset['CO(GT)'].values

x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.3)
minMaxScaler = MinMaxScaler()
x_train = minMaxScaler.fit_transform(x_train)
x_test = minMaxScaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], 1, layers['input'])
x_test = x_test.reshape(x_test.shape[0], 1, layers['input'])

feature_ph = tf.placeholder(tf.float32, [None, 1, layers['input']])
target_ph = tf.placeholder(tf.float32, [None, layers['output']])

cell = tf.contrib.rnn.LSTMCell(num_units=context_unit)
cell_output, _ = tf.nn.dynamic_rnn(cell, feature_ph, dtype=tf.float32)
logits = tf.layers.dense(cell_output[:, -1], 1)
result = tf.nn.sigmoid(logits, name="Prediction")

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target_ph, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
correct = tf.equal(tf.round(result), target_ph)
acc = tf.reduce_mean(tf.cast(correct, tf.float32))

loss_list, acc_list = [], []
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for e in range(epoch):
      i = 0
      while i < len(x_train):
          x = x_train[i: i + batch_size]
          y = y_train[i: i + batch_size].reshape(-1, 1)
          feed_dict = {feature_ph: x, target_ph: y}
          sess.run(optimizer, feed_dict=feed_dict)
          i += batch_size
      train_dict = {
          feature_ph: x_train,
          target_ph: y_train.reshape(-1, 1)
      }
      train_loss, train_acc = sess.run([loss, acc], feed_dict=train_dict)
      print(f'Epoch {e + 1}, Loss: {train_loss}, Accuracy: {train_acc}')
      loss_list.append(train_loss)
      acc_list.append(train_acc)
  test_acc = sess.run(acc, feed_dict={feature_ph: x_test, target_ph: y_test.reshape(-1, 1)})
  print(f'Test Accuracy: {test_acc * 100}%')
  plt.plot(range(e), loss_list)
  plt.plot(range(e), acc_list)
  plt.show()