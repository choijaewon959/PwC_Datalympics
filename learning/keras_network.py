from keras.models import Sequential
from keras.layers import Dense
import numpy
dataset = numpy.loadtxt("dataset.csv", delimiter=",")
X = dataset[:,0:8] #comma takes the columns from 0 to 7th only applicable for numpy 2d array!
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu')) #weights are initialized to be random number from 0 to 0.05 (default)
model.add(Dense(9, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
   
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  # tenserflow automatically chooses the best way to represent the network for training 
  # binary classification problem = binary_crossentry
  # gradient descent method = adam
#train model  
model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
  # weights are updated after each batch 
  #epoches = number of times the entire dataset gets traversed by nn model
  #          epoch too high will cause overfitting
  #batch_size = dividing training set into batch_size number of training data / there is a tradeoff between batch_size and accuracy 
  #             however bigger than batch_size the faster the training
  #verbose = layout of the output during training
scores = model.evaluate(X, Y)
predictions = model.predict(X)
rounded = [round(x[0]) for x in predictions]
print(rounded)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))