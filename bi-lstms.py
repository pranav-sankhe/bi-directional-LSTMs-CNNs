'''
An implementation of Bi-Directional LSTMs. 

Consider a 10 time-step input sequence. 
Input = [0.29414551, 0.91587952, 0.95189228, 0.32195638, 0.60742236, 0.83895793, 0.18023048, 0.84762691, 0.29165514]  # 10 time steps 
output = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1] 

'''

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional 
from keras.layers import BatchNormalization
# create a sequence classification instance
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

INPUT_SIZE = 30*15
OUTPUT_SIZE = 1
HIDDEN_LAYER_1 = 30*15 
HIDDEN_LAYER_2 = 30*15


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


# define problem properties


'''
We will define the sequences as having 10 timesteps.
Next, we can define an LSTM for the problem. The input layer will have 10 timesteps with 1 feature a piece, input_shape=(10, 1).

The first hidden layer will have 20 memory units and the output layer will be a fully connected layer that outputs one value per timestep. 
A sigmoid activation function is used on the output to predict the binary value.

A TimeDistributed wrapper layer is used around the output layer so that one value per timestep can be predicted given the full sequence provided as input. 
This requires that the LSTM hidden layer returns a sequence of values (one per timestep) rather than a single value for the whole input sequence.

Finally, because this is a binary classification problem, the binary log loss (binary_crossentropy in Keras) is used. 
The efficient ADAM optimization algorithm is used to find the weights and the accuracy metric is calculated and reported each epoch.
'''



# define LSTM
model = Sequential()
model.add(Bidirectional(LSTM(HIDDEN_LAYER_1, return_sequences=True), input_shape=(INPUT_SIZE, 1)))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(HIDDEN_LAYER_2, return_sequences=True)))
model.add(BatchNormalization())
model.add(TimeDistributed(Dense(OUTPUT_SIZE, activation='sigmoid')))    #Fully Connected layer

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


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