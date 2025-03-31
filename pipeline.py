import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Conv1D, MaxPooling1D, Flatten,
    LSTM, Bidirectional, TimeDistributed, 
    GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

import optuna

'''
I defined this file as a compilation of static functions, at first I wanted to make it a class, but when I realized 
it was maybe too late for that So I keeped it as functions, but the point is that I could recreate the process easily 
doesnt matter the data, and select the better model of all archs, and automatically the best model 
can be pulled by optuna and optimized it without a lot of code on the ipynb, to keep it clean and reproductible
'''

# We load the data
def load_data():
    df_hourly = pd.read_csv("univariate/Hourly-train.csv", index_col=0)
    df_info = pd.read_csv("univariate/m4_info.csv")

    # We select the assigned timeseries
    assign_filter = ['H404', 'H275', 'H64', 'H228', 'H38']
    df_filtered = df_hourly.loc[assign_filter]

    # Transpose and reindex as stepa
    df_filtered = df_filtered.T
    df_filtered.index.name = "Step"
    df_filtered.reset_index(inplace=True)

    # Get info
    df_info_filtered = df_info[df_info['M4id'].isin(assign_filter)]
    print("Info Filtered:")
    print(df_info_filtered)

    # Show first rows
    print("\nFiltered Data (First Rows):")
    print(df_filtered.head())

    def index_date(ts):
        # Get starting date
        starting_date = df_info_filtered[df_info_filtered['M4id'] == ts]['StartingDate'].iloc[0]
        
        # Create a range of datetime values starting from `starting_date`, with hourly frequency
        date_range = pd.date_range(start=starting_date, periods=len(df_filtered[ts]), freq='H')
        
        # Assign the generated date range as an index to the dataframe
        df_filtered[ts].index = date_range
        print(f"\n{ts} After Assigning Dates:")
        print(df_filtered.head())
        
        return df_filtered[ts].dropna()

    # Return the dataframes
    return df_filtered, df_info_filtered

###################################################################################################################

def split_data(ts, horizon=48):
    # define set sizes
    train_size = int(len(ts) * 0.7)
    val_size = int(len(ts) * 0.15)

    # Divide the data
    train_data = ts.iloc[:train_size]
    val_data = ts.iloc[train_size:train_size + val_size]
    test_data = ts.iloc[train_size + val_size:]

    # add prediction horizon
    train_data = train_data.iloc[:-horizon]

    # print size 
    print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

    return train_data, val_data, test_data

###################################################################################################################
def univariate_data_analysis(ts):

    #set the grid
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
    fig.tight_layout(pad=5.0)

    # Plot series
    axes[0, 0].plot(ts, color='dodgerblue', lw=1.5)
    axes[0, 0].set_title('Original Time Series', fontsize=14)
    axes[0, 0].set_xlabel('Date', fontsize=12)
    axes[0, 0].set_ylabel('Value', fontsize=12)

    # Distplot
    sns.histplot(ts, kde=True, ax=axes[0, 1], color='green', stat='density', linewidth=1)
    axes[0, 1].set_title('Distribution of Time Series', fontsize=14)

    # ACF plot
    plot_acf(ts, lags=50, ax=axes[1, 0], color='purple')
    axes[1, 0].set_title('Autocorrelation Function (ACF)', fontsize=14)

    # PACF plot
    plot_pacf(ts, lags=50, ax=axes[1, 1], color='orange')
    axes[1, 1].set_title('Partial Autocorrelation Function (PACF)', fontsize=14)

    plt.tight_layout()
    plt.show()

    # Moving average plot 
    window = 24  # 1 day window
    rolling_mean = ts.rolling(window=window).mean()
    rolling_std = ts.rolling(window=window).std()
    
    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(ts.index, ts, label='Original Series', color='dodgerblue', alpha=0.8)
    plt.plot(ts.index, rolling_mean, label=f'{window}-day Moving Average', color='red', lw=2)
    plt.fill_between(ts.index, 
                     rolling_mean - rolling_std, 
                     rolling_mean + rolling_std, 
                     color='lightcoral', alpha=0.3)
    
    plt.title(f'Time Series with a day Moving Average', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Decomposition
    decomposition = seasonal_decompose(ts, period=24)
    
    plt.figure(figsize=(12, 8), dpi=100)
    plt.subplot(411)
    plt.plot(ts.index, decomposition.observed, color='steelblue')
    plt.title('Observed', fontsize=14)
    plt.subplot(412)
    plt.plot(ts.index, decomposition.trend, color='seagreen')
    plt.title('Trend', fontsize=14)
    plt.subplot(413)
    plt.plot(ts.index, decomposition.seasonal, color='orange')
    plt.title('Seasonal', fontsize=14)
    plt.subplot(414)
    plt.plot(ts.index, decomposition.resid, color='gray')
    plt.title('Residual', fontsize=14)
    plt.tight_layout()
    plt.show()

    # Describe data
    print("Descriptive Statistics:")
    print(ts.describe())
    
    # Normality Test
    _, p_value = stats.shapiro(ts)
    print(f"\nShapiro-Wilk Test p-value: {p_value}")
    print("Interpretation: p < 0.05 suggests the data is not normally distributed")
    
    # Dickey-Fuller Test
    def adf_test(timeseries):
        print("\nAugmented Dickey-Fuller Test for Stationarity:")
        result = adfuller(timeseries)
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        print("Interpretation:")
        print("p < 0.05: Reject null hypothesis (Series is stationary)")
        print("p >= 0.05: Fail to reject null hypothesis (Series might be non-stationary)")
    
    adf_test(ts)

#############################################################

def split_univariate_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # calculate the step end
        end_ix = i + n_steps
        
        # see if we're over the seq
        if end_ix > len(sequence)-1:
            break
        # join input and output parts
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


##############################################


def time_series_forecasting(train_data, val_data, test_data, n_steps=24, forecast_horizon=48):
    def preprocess_data(train_data, val_data, test_data, n_steps):

        #turn data into np arrays
        train_data = np.array(train_data)
        val_data = np.array(val_data)
        test_data = np.array(test_data)
        
        #scale the data so the model will converge faster
        scaler = MinMaxScaler()
        train_data_flat = train_data.reshape(-1, 1)
        scaler.fit(train_data_flat)
        
        train_scaled = scaler.transform(train_data_flat).reshape(train_data.shape)
        val_scaled = scaler.transform(val_data.reshape(-1, 1)).reshape(val_data.shape)
        test_scaled = scaler.transform(test_data.reshape(-1, 1)).reshape(test_data.shape)
        
        # pairs for the scaled data 
        def create_sequences(data, n_steps):
            X, y = [], []
            for i in range(len(data) - n_steps):
                X.append(data[i:(i + n_steps)])
                y.append(data[i + n_steps])
            return np.array(X), np.array(y)
        
        X_train, y_train = create_sequences(train_scaled, n_steps)
        X_val, y_val = create_sequences(val_scaled, n_steps)
        X_test, y_test = create_sequences(test_scaled, n_steps)
        
        # Prepare different data shapes for each model type

        #2D for mlp
        X_train_mlp = X_train.reshape((X_train.shape[0], -1))
        X_val_mlp = X_val.reshape((X_val.shape[0], -1))
        X_test_mlp = X_test.reshape((X_test.shape[0], -1))
        
        #3D for Cnn
        X_train_cnn = X_train.reshape((X_train.shape[0], n_steps, 1))
        X_val_cnn = X_val.reshape((X_val.shape[0], n_steps, 1))
        X_test_cnn = X_test.reshape((X_test.shape[0], n_steps, 1))
        
        #3D for lstm
        X_train_lstm = X_train.reshape((X_train.shape[0], n_steps, 1))
        X_val_lstm = X_val.reshape((X_val.shape[0], n_steps, 1))
        X_test_lstm = X_test.reshape((X_test.shape[0], n_steps, 1))
        
        return {
            'X_train_mlp': X_train_mlp, 'y_train': y_train,
            'X_val_mlp': X_val_mlp, 'y_val': y_val,
            'X_test_mlp': X_test_mlp, 'y_test': y_test,
            'X_train_cnn': X_train_cnn,
            'X_val_cnn': X_val_cnn,
            'X_test_cnn': X_test_cnn,
            'X_train_lstm': X_train_lstm,
            'X_val_lstm': X_val_lstm,
            'X_test_lstm': X_test_lstm,
            'scaler': scaler
        }
#######################################################################################################
#     
    def create_model_and_train(model, X_train, y_train, X_val, y_val, model_name):

        model.compile(
            optimizer=Adam(learning_rate=0.001), #small lr for better performance and adam has shown its one of the better optimizers
            loss='mean_squared_error', # for regression
            metrics=['mae']  # easy to interpret
        )
        
        #early stop if in 30 epochs we dont get better results
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=30, 
            restore_best_weights=True
        )

        #train with 200 epochs
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            verbose=0,
            callbacks=[early_stop]
        )
        
        return model, history
    
#######################################################################################################

    def select_best_model_and_forecast(processed_data, models, model_types, forecast_horizon=48):
        model_performance = {}

        for model, model_type in zip(models, model_types):
            # Use the right data for each model 
            if 'mlp' in model_type.lower():
                X_test = processed_data['X_test_mlp']
            elif 'cnn' in model_type.lower() and 'lstm' not in model_type.lower():
                X_test = processed_data['X_test_cnn']
            else:  # LSTM and CNN-LSTM are the same
                X_test = processed_data['X_test_lstm']
            
            #make predictions and inverse scaling to normal 
            predictions = model.predict(X_test)
            predictions_original = processed_data['scaler'].inverse_transform(predictions)
            y_test_original = processed_data['scaler'].inverse_transform(processed_data['y_test'].reshape(-1, 1))
            
            #calculate MAE
            mae = mean_absolute_error(y_test_original, predictions_original)
            model_performance[model_type] = mae
            print(f"{model_type} MAE: {mae}")
        
        #select the best model
        best_model_type = min(model_performance, key=model_performance.get)
        best_model_index = model_types.index(best_model_type)
        best_model = models[best_model_index]
        
        print(f"\nBest Model: {best_model_type}")
        
        # Reshape the secuence accord to the model that was better
        if 'mlp' in best_model_type.lower():
            initial_sequence = processed_data['X_test_mlp'][-1].reshape(1, -1)
        elif 'cnn' in best_model_type.lower() and 'lstm' not in best_model_type.lower():
            initial_sequence = processed_data['X_test_cnn'][-1].reshape(1, processed_data['X_test_cnn'].shape[1], 1)
        else:  # LSTM and CNN-LSTM
            initial_sequence = processed_data['X_test_lstm'][-1].reshape(1, processed_data['X_test_lstm'].shape[1], 1)
        
        # generate forecast
        forecast = []
        current_sequence = initial_sequence
        
        # advance n(horizon) steps
        for _ in range(forecast_horizon):
            next_prediction = best_model.predict(current_sequence)
            next_prediction_original = processed_data['scaler'].inverse_transform(next_prediction)
            forecast.append(next_prediction_original[0][0])

            #update each models secuence
            
            if 'mlp' in best_model_type.lower():
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[0, -1] = next_prediction[0, 0] 
            else:
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = next_prediction[0, 0]
        
        return best_model, best_model_type, np.array(forecast), model_performance
    
    #######################################################################################################

    
    def plot_forecast(train_data, val_data, test_data, forecast, title, best_model, processed_data):
            # Original full timeline plot
            plt.figure(figsize=(15, 7))
            
            # Plot train data
            plt.plot(range(len(train_data)), train_data, color='green', label='Training Data')
            
            # Plot val data
            train_end = len(train_data)
            val_end = train_end + len(val_data)
            plt.plot(range(train_end, val_end), val_data, color='orange', label='Validation Data')
            
            # Plot test data
            test_start = val_end
            test_end = test_start + len(test_data)
            plt.plot(range(test_start, test_end), test_data, color='red', label='Test Data')
            
            # Plot forecast
            forecast_start = test_end
            forecast_end = forecast_start + len(forecast)
            plt.plot(range(test_end, forecast_end), forecast, color='blue', label='Forecast')
            
            plt.title(f'Time Series Forecast - {title}')
            plt.xlabel('Time Steps')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
            # choose input data for model evaluation
            if 'mlp' in title.lower():
                X_test = processed_data['X_test_mlp']
            elif 'cnn' in title.lower() and 'lstm' not in title.lower():
                X_test = processed_data['X_test_cnn']
            else:  # LSTM and CNN-LSTM
                X_test = processed_data['X_test_lstm']
            
            # Predict on test data
            predictions = best_model.predict(X_test)
            predictions_original = processed_data['scaler'].inverse_transform(predictions)
            y_test_original = processed_data['scaler'].inverse_transform(processed_data['y_test'].reshape(-1, 1))
            
            # Create another plot only for test vs pred
            plt.figure(figsize=(15, 7))
            plt.plot(range(len(y_test_original)), y_test_original, color='red', label='Actual Test Data')
            plt.plot(range(len(predictions_original)), predictions_original, color='blue', label='Model Predictions', linestyle='--')
            
            plt.title(f'Test Data vs Predictions - {title}')
            plt.xlabel('Time Steps (Test Period)')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            
            mse = mean_squared_error(y_test_original, predictions_original)
            mae = mean_absolute_error(y_test_original, predictions_original)
            r2 = r2_score(y_test_original, predictions_original)
            
            metrics_text = f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}"
            plt.annotate(metrics_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
            # Performance metrics table
            performance_data = {
                'Model Type': list(model_performance.keys()),
                'Mean Absolute Error': list(model_performance.values())
            }
            performance_df = pd.DataFrame(performance_data)
            performance_df = performance_df.sort_values('Mean Absolute Error')
            print("\nModel Performance Comparison:")
            print(performance_df.to_string(index=False))
            
            return mse, mae

#######################################################################################################

    # Model Architectures
    # MLP Models

    # 4 dense layers rom 128 to 16 with relu
    mlp_model_1 = Sequential([
        Dense(128, activation='relu', input_shape=(n_steps * 1,)), #flatten seq
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    # like model one but a bigger scale of neurons from 256 to 32, also with Relu
    mlp_model_3 = Sequential([
        Dense(256, activation='relu', input_shape=(n_steps * 1,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # like model one but using dropout to see if the model improoves
    mlp_model_2 = Sequential([
        Dense(128, activation='relu', input_shape=(n_steps * 1,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    # CNN Models

    # 1 conv layer with 64 filters and kernel of 3
    cnn_model_1 = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(n_steps, 1)),
        MaxPooling1D(2), #reduce dimensionality
        Flatten(), #preparing for dense
        Dense(50, activation='relu'),
        Dense(1)
    ])

    # 2 conv layers with kernel 3 and 64 and 128 filters, also more max pooling and with dropout
    cnn_model_2 = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(n_steps, 1)),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dense(1)
    ])

    # using global avg pooling that aggrupates and reduces features into a single vector, also 2 conv layers
    cnn_model_3 = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(n_steps, 1)),
        Conv1D(128, 3, activation='relu'),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # LSTM Models
    # singfle lstm layer with 50 neurons
    lstm_model_1 = Sequential([
        LSTM(50, activation='relu', input_shape=(n_steps, 1)),
        Dense(1)
    ])

    #stacked lstm but with 64 and 32 neurons and a extra dense layer
    lstm_model_2 = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(n_steps, 1)),
        LSTM(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    #bidirectional lst with 64 neurons, another dense and dropout
    lstm_model_3 = Sequential([
        Bidirectional(LSTM(64, activation='relu'), input_shape=(n_steps, 1)),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    # CNN-LSTM Models
    # 1 conv, maxpooling and then timedistributed to apply the dense per sequence as we saw on class, then a lstm layer
    cnnlstm_model_1 = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(n_steps, 1)),
        MaxPooling1D(2),
        TimeDistributed(Dense(25, activation='relu')),
        LSTM(50, activation='relu'),
        Dense(1)
    ])

    # now with two conv layers and more neurons on the dense and lstm layers
    cnnlstm_model_2 = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(n_steps, 1)),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        TimeDistributed(Dense(32, activation='relu')), 
        LSTM(64, activation='relu'),
        Dense(1)
    ])

    # now a bidirectional LSTM, with 1 conv and again Timedistributed dense 
    cnnlstm_model_3 = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(n_steps, 1)),
        MaxPooling1D(2),
        TimeDistributed(Dense(32, activation='relu')),
        Bidirectional(LSTM(64, activation='relu')),
        Dense(1)
    ])

    # Preprocess data
    processed_data = preprocess_data(train_data, val_data, test_data, n_steps)

    # Collect models and types
    models = [
        # MLP
        mlp_model_1, mlp_model_2, mlp_model_3,
        # CNN
        cnn_model_1, cnn_model_2, cnn_model_3,
        # LSTM
        lstm_model_1, lstm_model_2, lstm_model_3,
        # CNN-LSTM
        cnnlstm_model_1, cnnlstm_model_2, cnnlstm_model_3
    ]

    model_types = [
        # MLP 
        'MLP Model 1', 'MLP Model 2', 'MLP Model 3',
        # CNN 
        'CNN Model 1', 'CNN Model 2', 'CNN Model 3',
        # LSTM
        'LSTM Model 1', 'LSTM Model 2', 'LSTM Model 3',
        # CNN-LSTM
        'CNN-LSTM Model 1', 'CNN-LSTM Model 2', 'CNN-LSTM Model 3'
    ]

    # Train all models
    trained_models = []
    for i, model in enumerate(models):
        if i < 3:  # MLP models
            trained_model, _ = create_model_and_train(
                model, 
                processed_data['X_train_mlp'], 
                processed_data['y_train'], 
                processed_data['X_val_mlp'], 
                processed_data['y_val'], 
                model_types[i]
            )
        elif i < 6:  # CNN models
            trained_model, _ = create_model_and_train(
                model, 
                processed_data['X_train_cnn'], 
                processed_data['y_train'], 
                processed_data['X_val_cnn'], 
                processed_data['y_val'], 
                model_types[i]
            )
        elif i < 9:  # LSTM models
            trained_model, _ = create_model_and_train(
                model, 
                processed_data['X_train_lstm'], 
                processed_data['y_train'], 
                processed_data['X_val_lstm'], 
                processed_data['y_val'], 
                model_types[i]
            )
        else:  # CNN-LSTM models
            trained_model, _ = create_model_and_train(
                model, 
                processed_data['X_train_lstm'], 
                processed_data['y_train'], 
                processed_data['X_val_lstm'], 
                processed_data['y_val'], 
                model_types[i]
            )
        trained_models.append(trained_model)

    # Select the best model and generate forecast
    best_model, best_model_type, forecast, model_performance = select_best_model_and_forecast(
        processed_data, 
        trained_models, 
        model_types,
        forecast_horizon
    )

    # Plot forecast and get metrics
    mse, mae = plot_forecast(
        train_data, 
        val_data, 
        test_data, 
        forecast, 
        best_model_type, 
        best_model, 
        processed_data
    )

    # Return
    return {
        'best_model': best_model,
        'best_model_type': best_model_type,
        'forecast': forecast,
        'model_performance': model_performance,
        'mse': mse,
        'mae': mae
    }


##############################################################
#now for optuna 
def prepare_data_for_optimization(train_data, val_data, n_steps, model_type='mlp'):

    # Convert to numpy
    train_data = np.array(train_data) if isinstance(train_data, pd.Series) else train_data
    val_data = np.array(val_data) if isinstance(val_data, pd.Series) else val_data
    
    # Scale the data
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
    val_data_scaled = scaler.transform(val_data.reshape(-1, 1)).flatten()
    
    # Prepare sequences
    def create_sequences(data, n_steps):
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i:i + n_steps])
            y.append(data[i + n_steps])
        return np.array(X), np.array(y)
    
    # make sequences per model type
    if model_type == 'mlp':
        train_X, train_y = create_sequences(train_data_scaled, n_steps)
        val_X, val_y = create_sequences(val_data_scaled, n_steps)
        
        # Reshape for MLP
        train_X = train_X.reshape((train_X.shape[0], -1))
        val_X = val_X.reshape((val_X.shape[0], -1))
    else:
        train_X, train_y = create_sequences(train_data_scaled, n_steps)
        val_X, val_y = create_sequences(val_data_scaled, n_steps)
        
        # Reshape for CNN, LSTM, CNN-LSTM, which recieve the same shape
        train_X = train_X.reshape((train_X.shape[0], n_steps, 1))
        val_X = val_X.reshape((val_X.shape[0], n_steps, 1))
    
    return train_X, train_y, val_X, val_y, scaler

##############################################################

def create_optuna_model(trial, input_shape, model_type='mlp'):

    # Select optimizer, to see if there's one better than adam, and also diferent lr
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate)
    
    # Select the activation function
    activation = trial.suggest_categorical('activation', ['relu', 'elu', 'tanh'])
    
    if model_type == 'mlp':
        # Hyperparameters for MLP
        n_layers = trial.suggest_int('n_layers', 2, 5)
        layers = []
        
        # Input layer
        in_neurons = trial.suggest_int('neurons_0', 32, 256)
        layers.append(Dense(in_neurons, activation=activation, input_shape=input_shape))
        
        # Hidden layers
        for i in range(1, n_layers):
            neurons = trial.suggest_int(f'neurons_{i}', 16, 128)
            dropout_rate = trial.suggest_float(f'dropout_{i}', 0.0, 0.5)
            
            layers.append(Dense(neurons, activation=activation))
            if dropout_rate > 0:
                layers.append(Dropout(dropout_rate))
        
        # Output layer
        layers.append(Dense(1))
        
        model = Sequential(layers)
    
    elif model_type == 'cnn':
        # Hyperparameters for CNN
        n_conv_layers = trial.suggest_int('n_conv_layers', 1, 3)
        layers = []
        
        # First convolutional layer
        first_filters = trial.suggest_int('first_filters', 32, 128)
        first_kernel = trial.suggest_int('first_kernel', 2, 5)
        layers.append(Conv1D(first_filters, first_kernel, activation=activation, input_shape=input_shape))
        
        # Additional convolutional layers
        for i in range(1, n_conv_layers):
            filters = trial.suggest_int(f'filters_{i}', 32, 256)
            kernel = trial.suggest_int(f'kernel_{i}', 2, 5)
            layers.append(Conv1D(filters, kernel, activation=activation))
            
            # Optional pooling
            if trial.suggest_categorical(f'add_pooling_{i}', [True, False]):
                layers.append(MaxPooling1D(2))
        
        layers.append(GlobalAveragePooling1D())
        
        # Dense layers
        n_dense_layers = trial.suggest_int('n_dense_layers', 1, 3)
        for i in range(n_dense_layers):
            neurons = trial.suggest_int(f'dense_neurons_{i}', 16, 128)
            dropout_rate = trial.suggest_float(f'dense_dropout_{i}', 0.0, 0.5)
            
            layers.append(Dense(neurons, activation=activation))
            if dropout_rate > 0:
                layers.append(Dropout(dropout_rate))
        
        layers.append(Dense(1))
        
        model = Sequential(layers)
    
    elif model_type == 'lstm':
        # Hyperparameters for LSTM
        n_lstm_layers = trial.suggest_int('n_lstm_layers', 1, 3)
        layers = []
        
        # First LSTM layer
        first_units = trial.suggest_int('first_lstm_units', 32, 256)
        is_bidirectional = trial.suggest_categorical('is_bidirectional', [True, False])
        
        if is_bidirectional:
            layers.append(Bidirectional(LSTM(first_units, activation=activation, return_sequences=n_lstm_layers > 1), 
                                         input_shape=input_shape))
        else:
            layers.append(LSTM(first_units, activation=activation, 
                               return_sequences=n_lstm_layers > 1, 
                               input_shape=input_shape))
        
        # Additional LSTM layers
        for i in range(1, n_lstm_layers):
            units = trial.suggest_int(f'lstm_units_{i}', 32, 256)
            dropout_rate = trial.suggest_float(f'lstm_dropout_{i}', 0.0, 0.5)
            
            if is_bidirectional:
                layers.append(Bidirectional(LSTM(units, activation=activation, 
                                                 return_sequences=(i < n_lstm_layers - 1))))
            else:
                layers.append(LSTM(units, activation=activation, 
                                   return_sequences=(i < n_lstm_layers - 1)))
            
            if dropout_rate > 0:
                layers.append(Dropout(dropout_rate))
        
        # Dense layers
        n_dense_layers = trial.suggest_int('n_dense_layers', 1, 3)
        for i in range(n_dense_layers):
            neurons = trial.suggest_int(f'dense_neurons_{i}', 16, 128)
            layers.append(Dense(neurons, activation=activation))
        
        layers.append(Dense(1))
        
        model = Sequential(layers)
    
    elif model_type == 'cnnlstm':
        # Hyperparameters for CNN-LSTM
        n_conv_layers = trial.suggest_int('n_conv_layers', 1, 3)
        n_lstm_layers = trial.suggest_int('n_lstm_layers', 1, 3)
        layers = []
        
        # Convolutional layers
        first_filters = trial.suggest_int('first_filters', 32, 128)
        first_kernel = trial.suggest_int('first_kernel', 2, 5)
        layers.append(Conv1D(first_filters, first_kernel, activation=activation, input_shape=input_shape))
        
        for i in range(1, n_conv_layers):
            filters = trial.suggest_int(f'filters_{i}', 32, 256)
            kernel = trial.suggest_int(f'kernel_{i}', 2, 5)
            layers.append(Conv1D(filters, kernel, activation=activation))
            
            if trial.suggest_categorical(f'add_pooling_{i}', [True, False]):
                layers.append(MaxPooling1D(2))
        
        # LSTM layers
        first_lstm_units = trial.suggest_int('first_lstm_units', 32, 256)
        is_bidirectional = trial.suggest_categorical('is_bidirectional', [True, False])
        
        layers.append(TimeDistributed(Dense(first_lstm_units // 2, activation=activation)))
        
        if is_bidirectional:
            layers.append(Bidirectional(LSTM(first_lstm_units, activation=activation, 
                                              return_sequences=n_lstm_layers > 1)))
        else:
            layers.append(LSTM(first_lstm_units, activation=activation, 
                                return_sequences=n_lstm_layers > 1))
        
        # Additional LSTM layers
        for i in range(1, n_lstm_layers):
            units = trial.suggest_int(f'lstm_units_{i}', 32, 256)
            dropout_rate = trial.suggest_float(f'lstm_dropout_{i}', 0.0, 0.5)
            
            if is_bidirectional:
                layers.append(Bidirectional(LSTM(units, activation=activation, 
                                                 return_sequences=(i < n_lstm_layers - 1))))
            else:
                layers.append(LSTM(units, activation=activation, 
                                   return_sequences=(i < n_lstm_layers - 1)))
            
            if dropout_rate > 0:
                layers.append(Dropout(dropout_rate))
        
        # Dense layers
        n_dense_layers = trial.suggest_int('n_dense_layers', 1, 3)
        for i in range(n_dense_layers):
            neurons = trial.suggest_int(f'dense_neurons_{i}', 16, 128)
            layers.append(Dense(neurons, activation=activation))
        
        layers.append(Dense(1))
        
        model = Sequential(layers)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model

##############################################################

def objective(trial, train_X, train_y, val_X, val_y, n_steps, model_type='mlp'):

    # Determine input shape based on the model type
    if model_type == 'mlp':
        input_shape = (n_steps * 1,)
    else:
        input_shape = (n_steps, 1)
    
    # Create and train the model
    model = create_optuna_model(trial, input_shape, model_type)
    
    # Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=20, 
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        train_X, train_y,32
        validation_data=(val_X, val_y),
        epochs=100,
        batch_size=32,
        verbose=0,
        callbacks=[early_stop]
    )
    
    # Return the best validation MAE
    return min(history.history['val_mae'])

##############################################################

def optimize_model(train_data, val_data, test_data, n_steps=24, model_type='mlp', n_trials=100, forecast_horizon=48):
   
    # Prepare data for optimization
    train_X, train_y, val_X, val_y, scaler = prepare_data_for_optimization(
        train_data, val_data, n_steps, model_type
    )
    
    # Test data preparation
    test_data_array = np.array(test_data)
    test_data_scaled = scaler.transform(test_data_array.reshape(-1, 1)).flatten()
    
    # Create test sequences
    test_X, test_y = [], []
    for i in range(len(test_data_scaled) - n_steps):
        test_X.append(test_data_scaled[i:i + n_steps])
        test_y.append(test_data_scaled[i + n_steps])
    
    test_X = np.array(test_X)
    test_y = np.array(test_y)
    
    # Prepare data for the specific model type
    if model_type == 'mlp':
        train_X_model = train_X.reshape((train_X.shape[0], -1))
        val_X_model = val_X.reshape((val_X.shape[0], -1))
        test_X_model = test_X.reshape((test_X.shape[0], -1))
    else:
        train_X_model = train_X.reshape((train_X.shape[0], n_steps, 1))
        val_X_model = val_X.reshape((val_X.shape[0], n_steps, 1))
        test_X_model = test_X.reshape((test_X.shape[0], n_steps, 1))
    
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, train_X_model, train_y, val_X_model, val_y, n_steps, model_type), 
        n_trials=n_trials
    )
    
    # Print best trial information
    print(f"Best {model_type.upper()} Model Optimization Results:")
    print(f"  Number of trials: {len(study.trials)}")
    print(f"  Best trial value (MAE): {study.best_value}")
    print("  Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    # Determine input shape based on model type
    input_shape = (n_steps * 1,) if model_type == 'mlp' else (n_steps, 1)
    
    # Create the best model with optimal hyperparameters and use the best trial directly
    best_model = create_optuna_model(
        study.best_trial,  
        input_shape, 
        model_type
    )
    
    # Combine train and validation data
    full_train_X = np.concatenate([train_X_model, val_X_model], axis=0)
    full_train_y = np.concatenate([train_y, val_y], axis=0)
    
    # Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=20, 
        restore_best_weights=True
    )
    
    # Train on full data
    best_model.fit(
        full_train_X, full_train_y,
        epochs=100,
        verbose=0,
        validation_split=0.2,
        callbacks=[early_stop]
    )
    
    # predictions on test data
    test_predictions = best_model.predict(test_X_model)
    test_predictions_original = scaler.inverse_transform(test_predictions)
    test_true_original = scaler.inverse_transform(test_y.reshape(-1, 1))
    
    # Calculate  metrics
    mae = mean_absolute_error(test_true_original, test_predictions_original)
    mse = mean_squared_error(test_true_original, test_predictions_original)
    
    #generate forecast for the model
    def generate_forecast(model, initial_sequence, forecast_horizon):
        forecast = []
        current_sequence = initial_sequence
        
        for _ in range(forecast_horizon):
            next_prediction = model.predict(current_sequence)
            next_prediction_original = scaler.inverse_transform(next_prediction)
            forecast.append(next_prediction_original[0][0])
            
            # Update sequence
            if model_type == 'mlp':
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[0, -1] = next_prediction[0, 0] 
            else:
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = next_prediction[0, 0]
        
        return np.array(forecast)
    
    # Use the last sequence of test data as initial sequence for forecasting
    initial_forecast_sequence = test_X_model[-1].reshape(1, *test_X_model[-1].shape)
    forecast = generate_forecast(best_model, initial_forecast_sequence, forecast_horizon)
    
    # Plotting
    plt.figure(figsize=(15, 7))
    
    # Plot training and val data
    plt.plot(range(len(train_data)), train_data, color='green', label='Training Data')
    train_end = len(train_data)
    val_end = train_end + len(val_data)
    plt.plot(range(train_end, val_end), val_data, color='orange', label='Validation Data')
    
    # Plot test data
    test_start = val_end
    test_end = test_start + len(test_data)
    plt.plot(range(test_start, test_end), test_data, color='red', label='Test Data')
    
    # Plot forecast
    forecast_start = test_end
    forecast_end = forecast_start + len(forecast)
    plt.plot(range(test_end, forecast_end), forecast, color='blue', label='Forecast')
    
    plt.title(f'Time Series Forecast - {model_type.upper()} Optimized Model')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Print prediction comparison
    print("\nTest Data Prediction Analysis:")
    plt.figure(figsize=(15, 6))
    plt.plot(test_true_original, label='Actual Test Data', color='red')
    plt.plot(test_predictions_original, label='Predicted Test Data', color='blue')
    plt.title(f'{model_type.upper()} Model: Actual vs Predicted Test Data')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Performance metrics
    print(f"\nModel Performance Metrics:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    
    return {
        'best_model': best_model,
        'best_params': study.best_params,
        'study': study,
        'scaler': scaler,
        'forecast': forecast,
        'mae': mae,
        'mse': mse,
        'test_predictions': test_predictions_original,
        'test_true': test_true_original
    }


