'''
An implementation of Bi-Directional LSTMs. 

Consider a 10 time-step input sequence. 
Input = [0.29414551, 0.91587952, 0.95189228, 0.32195638, 0.60742236, 0.83895793, 0.18023048, 0.84762691, 0.29165514]  # 10 time steps 
output = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1] 

'''

import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional 
from keras.layers import BatchNormalization, Dropout

# create a sequence classification instance
config = tf.ConfigProto( device_count = {'GPU': 4 } ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

INPUT_SIZE = 30*15
OUTPUT_SIZE = 60
FULLY_CONNECTED_SIZE = 512
HIDDEN_LAYER_1 = 256 
HIDDEN_LAYER_2 = 256


# sample data creator to check the code

def get_sequence(input_size):
	# create a sequence of random numbers in [0,1]
	X = np.random.rand(input_size)
	
	# calculate cut-off value to change class values
	limit = input_size/4.0
	# determine the class outcome for each item in cumulative sequence
	y = np.array([0 if x < limit else 1 for x in np.cumsum(X)])
	
	#reshape input and output data to be suitable for LSTMs
	X = X.reshape(1, input_size, 1)
	y = y.reshape(1, input_size, 1)
	
	return X, y



# define Model

# gru = GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', 
# 						recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, 
# 						bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, 
# 						dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, 
# 						unroll=False, reset_after=False)

# lstm = LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
# 						bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
# 						kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, 
# 						return_state=False, go_backwards=False, stateful=False, unroll=False)

model = Sequential()

model.add(Bidirectional(LSTM(HIDDEN_LAYER_1, return_sequences=True), input_shape=(INPUT_SIZE, 1)))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(HIDDEN_LAYER_2, return_sequences=True)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(OUTPUT_SIZE, activation='relu')))    #Fully Connected layer

rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.9)


model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['acc'])


for epoch in range(1000):
	# generate new random sequence
	X,y = get_sequence(INPUT_SIZE)
	
	# fit model for one epoch on this sequence
	model.fit(X, y, epochs=1, batch_size=1, verbose=2)


# # evaluate LSTM
# X,y = get_sequence(INPUT_SIZE)
# yhat = model.predict_classes(X, verbose=0)
# for i in range(n_timesteps):
# 	print('Expected:', y[0, i], 'Predicted', yhat[0, i])	