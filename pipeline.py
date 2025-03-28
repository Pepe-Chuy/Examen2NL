import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
from tensorflow.keras.optimizers import Adam


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
    # Definir proporciones de los conjuntos
    train_size = int(len(ts) * 0.7)
    val_size = int(len(ts) * 0.15)

    # Dividir los datos
    train_data = ts.iloc[:train_size]
    val_data = ts.iloc[train_size:train_size + val_size]
    test_data = ts.iloc[train_size + val_size:]

    # Ajustar horizonte de predicción
    train_data = train_data.iloc[:-horizon]  # Se elimina el horizonte del final para evitar fuga de datos

    # Imprimir tamaños
    print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

    return train_data, val_data, test_data

###################################################################################################################
def univariate_data_analysis(ts):
    # Set a more compact style
    sns.set(style="whitegrid")
    
    # Set up figure for the 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
    fig.tight_layout(pad=5.0)

    # Time Series Plot
    axes[0, 0].plot(ts, color='dodgerblue', lw=1.5)
    axes[0, 0].set_title('Original Time Series', fontsize=14)
    axes[0, 0].set_xlabel('Date', fontsize=12)
    axes[0, 0].set_ylabel('Value', fontsize=12)

    # Distribution Plot
    sns.histplot(ts, kde=True, ax=axes[0, 1], color='green', stat='density', linewidth=1)
    axes[0, 1].set_title('Distribution of Time Series', fontsize=14)

    # ACF Plot
    plot_acf(ts, lags=50, ax=axes[1, 0], color='purple')
    axes[1, 0].set_title('Autocorrelation Function (ACF)', fontsize=14)

    # PACF Plot
    plot_pacf(ts, lags=50, ax=axes[1, 1], color='orange')
    axes[1, 1].set_title('Partial Autocorrelation Function (PACF)', fontsize=14)

    plt.tight_layout()
    plt.show()

    # Rolling Statistics Plot (Moving Average and Standard Deviation)
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
    plt.title(f'Time Series with a day Moving Average and Std. Dev.', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Decomposition for the Time Series
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

    # Statistical Tests and Descriptive Statistics
    print("Descriptive Statistics:")
    print(ts.describe())
    
    # Shapiro-Wilk Normality Test
    _, p_value = stats.shapiro(ts)
    print(f"\nShapiro-Wilk Test p-value: {p_value}")
    print("Interpretation: p < 0.05 suggests the data is not normally distributed")
    
    # Augmented Dickey-Fuller Test for Stationarity
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
        # encontrar el final de este patrón
        end_ix = i + n_steps
        
        # comprobar si estamos más allá de la secuencia
        if end_ix > len(sequence)-1:
            break
        # reunir partes de entrada y salida del patrón
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


##############################################


def time_series_forecasting(train_data, val_data, test_data, n_steps=24, forecast_horizon=48):
    def preprocess_data(train_data, val_data, test_data, n_steps):
        train_data = np.array(train_data)
        val_data = np.array(val_data)
        test_data = np.array(test_data)
        
        scaler = MinMaxScaler()
        train_data_flat = train_data.reshape(-1, 1)
        scaler.fit(train_data_flat)
        
        train_scaled = scaler.transform(train_data_flat).reshape(train_data.shape)
        val_scaled = scaler.transform(val_data.reshape(-1, 1)).reshape(val_data.shape)
        test_scaled = scaler.transform(test_data.reshape(-1, 1)).reshape(test_data.shape)
        
        def create_sequences(data, n_steps):
            X, y = [], []
            for i in range(len(data) - n_steps):
                X.append(data[i:(i + n_steps)])
                y.append(data[i + n_steps])
            return np.array(X), np.array(y)
        
        X_train, y_train = create_sequences(train_scaled, n_steps)
        X_val, y_val = create_sequences(val_scaled, n_steps)
        X_test, y_test = create_sequences(test_scaled, n_steps)
        
        # Prepare different data shapes for various model types
        X_train_mlp = X_train.reshape((X_train.shape[0], -1))
        X_val_mlp = X_val.reshape((X_val.shape[0], -1))
        X_test_mlp = X_test.reshape((X_test.shape[0], -1))
        
        X_train_cnn = X_train.reshape((X_train.shape[0], n_steps, 1))
        X_val_cnn = X_val.reshape((X_val.shape[0], n_steps, 1))
        X_test_cnn = X_test.reshape((X_test.shape[0], n_steps, 1))
        
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
    
    def create_model_and_train(model, X_train, y_train, X_val, y_val, model_name):
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=30, 
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            verbose=0,
            callbacks=[early_stop]
        )
        
        return model, history
    
    def select_best_model_and_forecast(processed_data, models, model_types, forecast_horizon=48):
        model_performance = {}

        for model, model_type in zip(models, model_types):
            # Determine appropriate input data based on model type
            if 'mlp' in model_type.lower():
                X_test = processed_data['X_test_mlp']
            elif 'cnn' in model_type.lower() and 'lstm' not in model_type.lower():
                X_test = processed_data['X_test_cnn']
            else:  # LSTM and CNN-LSTM
                X_test = processed_data['X_test_lstm']
            
            predictions = model.predict(X_test)
            predictions_original = processed_data['scaler'].inverse_transform(predictions)
            y_test_original = processed_data['scaler'].inverse_transform(processed_data['y_test'].reshape(-1, 1))
            
            mae = mean_absolute_error(y_test_original, predictions_original)
            model_performance[model_type] = mae
            print(f"{model_type} MAE: {mae}")
        
        best_model_type = min(model_performance, key=model_performance.get)
        best_model_index = model_types.index(best_model_type)
        best_model = models[best_model_index]
        
        print(f"\nBest Model: {best_model_type}")
        
        # Prepare initial sequence for forecasting
        if 'mlp' in best_model_type.lower():
            initial_sequence = processed_data['X_test_mlp'][-1].reshape(1, -1)
        elif 'cnn' in best_model_type.lower() and 'lstm' not in best_model_type.lower():
            initial_sequence = processed_data['X_test_cnn'][-1].reshape(1, processed_data['X_test_cnn'].shape[1], 1)
        else:  # LSTM and CNN-LSTM
            initial_sequence = processed_data['X_test_lstm'][-1].reshape(1, processed_data['X_test_lstm'].shape[1], 1)
        
        # Generate forecast
        forecast = []
        current_sequence = initial_sequence
        
        for _ in range(forecast_horizon):
            next_prediction = best_model.predict(current_sequence)
            next_prediction_original = processed_data['scaler'].inverse_transform(next_prediction)
            forecast.append(next_prediction_original[0][0])
            
            if 'mlp' in best_model_type.lower():
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[0, -1] = next_prediction[0, 0] 
            else:
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = next_prediction[0, 0]
        
        return best_model, best_model_type, np.array(forecast), model_performance
    
    def plot_forecast(train_data, val_data, test_data, forecast, title, best_model, processed_data):
        plt.figure(figsize=(15, 7))
        
        # Plot training data
        plt.plot(range(len(train_data)), train_data, color='green', label='Training Data')
        
        # Plot validation data
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
        
        # Determine appropriate input data for model evaluation
        if 'mlp' in title.lower():
            X_test = processed_data['X_test_mlp']
        elif 'cnn' in title.lower() and 'lstm' not in title.lower():
            X_test = processed_data['X_test_cnn']
        else:  # LSTM and CNN-LSTM
            X_test = processed_data['X_test_lstm']
        
        predictions = best_model.predict(X_test)
        predictions_original = processed_data['scaler'].inverse_transform(predictions)
        y_test_original = processed_data['scaler'].inverse_transform(processed_data['y_test'].reshape(-1, 1))
        
        mse = mean_squared_error(y_test_original, predictions_original)
        mae = mean_absolute_error(y_test_original, predictions_original)
        
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
    
    # Model Architectures
    # MLP Models
    mlp_model_1 = Sequential([
        Dense(128, activation='relu', input_shape=(n_steps * 1,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    mlp_model_3 = Sequential([
        Dense(256, activation='relu', input_shape=(n_steps * 1,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])

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
    cnn_model_1 = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(n_steps, 1)),
        MaxPooling1D(2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])

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

    cnn_model_3 = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(n_steps, 1)),
        Conv1D(128, 3, activation='relu'),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # LSTM Models
    lstm_model_1 = Sequential([
        LSTM(50, activation='relu', input_shape=(n_steps, 1)),
        Dense(1)
    ])

    lstm_model_2 = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(n_steps, 1)),
        LSTM(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    lstm_model_3 = Sequential([
        Bidirectional(LSTM(64, activation='relu'), input_shape=(n_steps, 1)),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    # CNN-LSTM Models
    cnnlstm_model_1 = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(n_steps, 1)),
        MaxPooling1D(2),
        TimeDistributed(Dense(25, activation='relu')),
        LSTM(50, activation='relu'),
        Dense(1)
    ])

    cnnlstm_model_2 = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(n_steps, 1)),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        TimeDistributed(Dense(32, activation='relu')), 
        LSTM(64, activation='relu'),
        Dense(1)
    ])

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
        # MLP Models
        mlp_model_1, mlp_model_2, mlp_model_3,
        # CNN Models
        cnn_model_1, cnn_model_2, cnn_model_3,
        # LSTM Models
        lstm_model_1, lstm_model_2, lstm_model_3,
        # CNN-LSTM Models
        cnnlstm_model_1, cnnlstm_model_2, cnnlstm_model_3
    ]

    model_types = [
        # MLP Models
        'MLP Model 1', 'MLP Model 2', 'MLP Model 3',
        # CNN Models
        'CNN Model 1', 'CNN Model 2', 'CNN Model 3',
        # LSTM Models
        'LSTM Model 1', 'LSTM Model 2', 'LSTM Model 3',
        # CNN-LSTM Models
        'CNN-LSTM Model 1', 'CNN-LSTM Model 2', 'CNN-LSTM Model 3'
    ]

    # Train models
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

    # Select best model and generate forecast
    best_model, best_model_type, forecast, model_performance = select_best_model_and_forecast(
        processed_data, 
        trained_models, 
        model_types,
        forecast_horizon
    )

    # Plot forecast and get performance metrics
    mse, mae = plot_forecast(
        train_data, 
        val_data, 
        test_data, 
        forecast, 
        best_model_type, 
        best_model, 
        processed_data
    )

    # Return comprehensive results
    return {
        'best_model': best_model,
        'best_model_type': best_model_type,
        'forecast': forecast,
        'model_performance': model_performance,
        'mse': mse,
        'mae': mae
    }