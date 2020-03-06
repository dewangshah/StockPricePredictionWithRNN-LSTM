"""
Created on Thu Mar  5 13:35:01 2020

@author: Dewang
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Part 1 - Data Preprocessing

# Importing the training set
dataset_train=pd.read_csv("C:\\Users\\Dewang\\Desktop\\Practice Apps\\Deep Learning A-Z\\RNN\\Google_Stock_Price_Train.csv")
training_set=dataset_train.iloc[:,1:2].values

# Feature Scaling

#2 best ways to apply feature scaling - 
#1] stardardization, X_stand = (x-mean(x))/std(x), 
#2] Normalization, X_norm = (x-min(x))/(max(x)-min(x))
#RNN with Sigmoid Activation function works best with applying Normalization feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))#feature_range = (0,1) - All scaled stock prices will be between 0 and 1
training_set_scaled=sc.fit_transform(training_set)

#Creating a data structure with 60 timesteps and 1 output - At each time t, the RNN will look at 60 stock prices (3 months)
#before time t, and based on the trends captured, it predicts the next output, ie. stock price at time t+1
X_train=[]
y_train=[]

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
    
X_train,y_train = np.array(X_train), np.array(y_train)#X_train - 1198x60 numpy array, each row containing last 60 stock prices

#Reshaping
#Adding more dimensionality (more parameters) to the numpy array if needed
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1)) #(X_train.shape(0) - No. of rows, X_train.shape(1) - No. of Columns, 1 - No. of Parameters)
 
  
# Part 2 - Building the RNN

# Importing the Keras libraries and Packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initialize the RNN
regressor= Sequential() #We call our NN as regressor, as we are predicting continuous values

# Adding the First LSTM Layer and some Dropout regularization
regressor.add(LSTM(units=50,return_sequences=True, input_shape = (X_train.shape[1],1))) # units - No. of LSTM cells in the layer, 
#return_sequences = True if there are going to be more LSTM layers after this layer
#input_shape - 60 timesteps and 1 predictor (last 2 dimensions of X_train)
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularization
regressor.add(LSTM(units=50,return_sequences=True)) #Input_shape is recognized automatically
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularization
regressor.add(LSTM(units=50,return_sequences=True)) #Input_shape is recognized automatically
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularization
regressor.add(LSTM(units=50)) #Input_shape is recognized automatically
regressor.add(Dropout(0.2))

#Adding the output Layer
regressor.add(Dense(units=1))#Only 1 Neuron in Output layer

#Compiling the RNN
regressor.compile(optimizer = 'adam', loss='mean_squared_error') #For RNN - RMSProp is recommended

#Fitting the RNN to the Training Set
regressor.fit(X_train,y_train, epochs=100, batch_size=32)

# Part 3 - Making the predictions and visualizing the results

#Getting the real stock price of 2017
dataset_test=pd.read_csv("C:\\Users\\Dewang\\Desktop\\Practice Apps\\Deep Learning A-Z\\RNN\\Google_Stock_Price_Test.csv")
real_stock_price=dataset_test.iloc[:,1:2].values

#Getting the predicted stock price of 2017
dataset_total=pd.concat((dataset_train['Open'],dataset_test['Open']), axis=0) #axis=0 - Vertical Concat, axis=1 - Horizontal Concat
#dataset_total contains stock prices from 2012 to January 2017

inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values

inputs= inputs.reshape(-1,1) #With this, we get the inputs with the different stock prices of Jan 3 - 3months upto the final
#stock prices in rows, in one column, for a numpy array

inputs = sc.transform(inputs) #sc was already fitted with the training set, so we use only sc.transform instead of fit_transform

X_test=[]

for i in range(60, 80):
    X_test.append(inputs[i-60:i,0])
    
X_test = np.array(X_test)#X_train - 1198x60 numpy array, each row containing last 60 stock prices

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price) #Getting back original stock prices before Feature Scaling

#Visualizing the results
plt.plot(real_stock_price, color='red', label='Google Real Stock Price')
plt.plot(predicted_stock_price, color='green', label='Google Predicted Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()