import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

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
