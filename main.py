from sklearn.preprocessing import MinMaxScaler
from helping_functions import *
from prediction import *
from model import *

# Data files to be used
totrain = "data/amzn00-22.csv"
topredict = "data/ttm22-23.csv"
number_of_predictions = 10

# Processing data
scaler = MinMaxScaler()
scaled_data_train, df_train = data_scaling(totrain, scaler)


# The number of previous values to be sent in input
n_steps = 5

# Dividing training and testing data
train_size = int(len(df_train) * 0.8)
train_data = scaled_data_train[:train_size]
test_data = scaled_data_train[train_size:]
X_train, y_train = create_sequences(train_data, n_steps)
X_test, y_test = create_sequences(test_data, n_steps)


# Training the model
trained_model, history = model(X_train, y_train, X_test, y_test)


# Getting actual values
actual_prices, actual_prices_df = data_scaling(topredict, scaler)
X_actual, y_actual = create_sequences(actual_prices, n_steps)

# Predicting fututre values
prediction(X_actual, number_of_predictions, trained_model, scaler)

# Using model to predict values
predicted_prices = trained_model.predict(X_actual)

# Calculating prediction loss
loss = trained_model.evaluate(X_actual, y_actual)
print("Prediction loss:", loss)

# Calculating error percentage
ms = np.mean(np.square(y_actual))
print("Error percentage = ", loss/ms * 100, "%")

# Rescaling data
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(actual_prices)

# Plotting the graph
graph(actual_prices, predicted_prices, actual_prices_df)
