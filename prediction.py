import matplotlib.pyplot as plt


def prediction(X_actual, number_of_predictions, trained_model, scaler):
    X_input = []
    for i in X_actual[-1]:
        X_input.append(list(i))
    ans = []
    for i in range(number_of_predictions):
        predicted_prices = trained_model.predict([X_input])
        temp = list(X_input[1:])
        temp.append((predicted_prices).tolist()[0])
        X_input = temp
        ans.append((scaler.inverse_transform(predicted_prices)).tolist()[0])
    plt.plot(ans, label='Predicted Value')
    plt.show()
