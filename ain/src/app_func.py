def handle_missing_data(df, columns) :  
# Subset the DataFrame to only include the specified columns
    print(df.columns)
    subset_df = df[columns]

    # Calculate the percentage of null values for each column in the subset
    null_percentages = subset_df.isnull().mean() * 100

    # List the columns with more than 18% null values
    columns_with_high_nulls = null_percentages[null_percentages > 18].index.tolist()
    # print(columns_with_high_nulls)

    # Drop the columns with high null values from the subset DataFrame
    modified_df = subset_df.drop(columns=columns_with_high_nulls)

    # Forward fill na values in the 'class' column
    if 'class' in modified_df.columns:
        modified_df['class'] = modified_df['class'].fillna(method='ffill')
    
    # modified_df = modified_df[modified_df['class'].notnull()]

    # Smooth the DataFrame using a moving average 
    window_size = 1800
    smoothed_df = modified_df.copy()
    sensor_columns = modified_df.columns.difference(['class'])
    smoothed_df[sensor_columns] = modified_df[sensor_columns].rolling(window=window_size, min_periods=1).mean()

    return(modified_df, smoothed_df)


def time_series_chart(df, columns, time_scale):
    # Section: Time Series Visualization
    import plotly.graph_objects as go
    import plotly.express as px
    

    # Sample data preparation (ensure 'class' column exists)
    sensor_data = df[columns]
    df_resampled = sensor_data.resample(f'{time_scale}T').ffill()
    df_resampled_ffill = df_resampled.ffill()

    # Determine min and max timestamps
    min_timestamp = df_resampled.index.min()
    max_timestamp = df_resampled.index.max()

    # Plotting with Plotly
    fig = go.Figure()

    # Add traces for each column in your time series data
    for column in df_resampled.columns:
        if column != 'class':  # Skip the 'class' column
            fig.add_trace(go.Scatter(x=df_resampled.index, y=df_resampled[column], mode='lines', name=column))


    # Add background color shading based on 'class'
    unique_classes = df_resampled['class'].unique()
    colors = px.colors.qualitative.Plotly  # Use Plotly color palette

    for i, cls in enumerate(unique_classes):
        class_df = df_resampled[df_resampled['class'] == cls]
        fig.add_vrect(
            x0=class_df.index.min(), x1=class_df.index.max(),
            fillcolor=colors[i % len(colors)], opacity=0.2,
            layer="below", line_width=0,
            name=f'Class: {cls}'
        )

    # Update layout for better readability
    fig.update_layout(
        title=f'Time Series Data Visualization (Resampled to {time_scale} Minutes)',
        xaxis_title='Timestamp',
        yaxis_title='Value',
        legend_title='Legend',
        legend=dict(x=0, y=-0.2, orientation='h'),
        xaxis_rangeslider=dict(
            visible=True,
            range=[min_timestamp, max_timestamp]  # Set slider range from min to max timestamp
        )
    )

    fig.show()


    return fig



def sensor_line_graph(df, columns_to_plot, unique_classes, window_size=180):
    """
    Generate line graphs for specified columns in the DataFrame, differentiating by 'y_pred' values.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the sensor data with 'y_pred' column and a datetime index.
    - columns_to_plot (list): List of columns to plot.
    - unique_classes (list): Unique values in 'y_pred' column to differentiate lines.
    - window_size (int): Window size for the rolling average.
    
    Returns:
    - fig (plotly.graph_objects.Figure): Plotly figure object with line plots.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Exclude 'class' from the columns_to_plot if it is present
    sensor_columns = [col for col in columns_to_plot if col != 'class']
    num_columns = len(sensor_columns)

    # Create subplots
    fig = make_subplots(
        rows=num_columns, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,  # Reduced vertical spacing
        subplot_titles=sensor_columns
    )

    # Define colors for each class
    colors = {cls: f'rgba({cls*100}, {255-cls*100}, 0, 0.8)' for cls in unique_classes}

    # Add traces for each column
    for i, column in enumerate(sensor_columns, 1):
        for unique_class in unique_classes:
            subset = df[df['y_pred'] == unique_class]
            
            if not subset.empty:
                # Apply rolling average
                smoothed_data = subset[column].rolling(window=window_size, min_periods=1).mean()

                fig.add_trace(
                    go.Scatter(
                        x=subset.index,
                        y=smoothed_data,
                        mode='lines',
                        name=f'y_pred {unique_class}',
                        line=dict(color=colors.get(unique_class, 'grey'), width=2),
                        legendgroup=column
                    ),
                    row=i, col=1
                )

        fig.update_yaxes(title_text=column, row=i, col=1)
        fig.update_xaxes(title_text='Datetime', row=i, col=1)

    # Update layout for the whole figure
    fig.update_layout(
        title='Sensor Data Line Graph by y_pred',
        showlegend=True,  
        height=300 * num_columns,  # Adjust height based on number of columns
        template='plotly_dark',
        legend=dict(x=0, y=-0.2, orientation='h')  # Adjust legend position
    )

    return fig




def z_score_outlier(df, columns):
    import numpy as np
    from scipy.stats import zscore

    # Calculate Z-scores for each column
    df_zscores = df[columns].apply(zscore)

    # Set a threshold for Z-scores to identify outliers
    threshold = 3

    # Identify outliers
    outliers = (np.abs(df_zscores) > threshold).any(axis=1)

    # Replace outliers with rolling average
    window_size = 3  # Set window size for rolling average
    z_score_df = df.copy()

    for col in columns:
        rolling_avg = df[col].rolling(window=window_size, min_periods=1).mean()
        z_score_df.loc[outliers, col] = rolling_avg[outliers]

    return(z_score_df)

def time_windowing(df, window_size=180, step_size=15):
    import pandas as pd
    windows = [df.iloc[i:i + window_size] for i in range(0, len(df), window_size)]
    # Select every 'step_size' window
    selected_windows = windows[::step_size]
    
     # Add window ID
    for window_id, window in enumerate(windows, start=1):
        window['id'] = window_id

    # Combine the selected windows into a single DataFrame
    result_df = pd.concat(selected_windows)

    return result_df

def scale_data(resampled_df, cols_to_check):
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    # Initialize the StandardScaler
    scaler = StandardScaler()

    features_value = resampled_df[cols_to_check]
    target_values = resampled_df['class']
    window_id = resampled_df['id']


    # Fit the scaler on the training data and transform the training data
    features_scaled = scaler.fit_transform(features_value)

    # Combine scaled features with the labels
    resampled_df_scaled = pd.DataFrame(features_scaled, index=features_value.index, columns=features_value.columns)
    resampled_df_scaled['class'] = target_values.values
    resampled_df_scaled['id'] = window_id.values

    return(resampled_df_scaled)


def extract_and_impute_features(data, id_column='id', timestamp_column='timestamp', drop_columns=['class'], custom_fc_parameters=None):
    from tsfresh import extract_features, select_features
    from tsfresh.utilities.dataframe_functions import impute
    
    if custom_fc_parameters is None:
        custom_fc_parameters = {
            'mean': None,
            'median': None,
            'standard_deviation': None,
            'variance': None,
            'maximum': None,
            'minimum': None
        }

    # Rename the timestamp column
    data = data.rename(columns={timestamp_column: 'time'})

    # Extract features
    extracted_features = extract_features(
        data.drop(columns=drop_columns),
        column_id=id_column,
        column_sort='time',
        default_fc_parameters=custom_fc_parameters
    )

    # Impute missing values
    selected_features = impute(extracted_features)

    return selected_features

