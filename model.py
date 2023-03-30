from tensorflow import keras
import os

def model(X_train, y_train, X_test, y_test):
    if os.path.exists('models'):
        model = keras.models.load_model('models')
        print('Loaded model from disk')
        history = None
    else:
        model = keras.Sequential([
            keras.layers.LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(64),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(y_train.shape[1])
        ])
        model.compile(optimizer='adam', loss='mse')
        callbacks = [
            keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1),
            keras.callbacks.ModelCheckpoint('models', save_best_only=True, save_weights_only=False),
        ]
        history = model.fit(X_train, y_train, epochs=30, batch_size=15, validation_split=0.1, callbacks=callbacks)
        print('Saved model and history to disk')
    loss = model.evaluate(X_test, y_test)
    print("Test loss:", loss)
    return model, history
