import streamlit as st
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from sklearn.ensemble import RandomForestClassifier
import altair as alt
import streamlit.components.v1 as components
from app_func import get_data, time_series_chart, z_score_outlier, time_windowing, scale_data, extract_and_impute_features, handle_missing_data
import shap
import streamlit as st
from PIL import Image
from selenium import webdriver
import os

print(os.getcwd())

instance_list_1 = ['WELL-00006_20170731180930','WELL-00006_20170731220432', 'WELL-00006_20180617200257']
instance_list_2 = ['WELL-00002_20131104014101','WELL-00009_20170313160804']
instance_list_5 = ['WELL-00017_20140319031616','WELL-00017_20140314135248']
instance_list_6 = ['WELL-00002_20140301151700', 'WELL-00002_20140212170333']
instance_list_7 = ['WELL-00001_20170226140146','WELL-00006_20180617181315', 'WELL-00018_20190403023307', 'WELL-00006_20180620155728']

# Using relative path for maintainability
base_directory = './dataset'

# Mapping of instance lists to their corresponding directories
instance_directories = {
    os.path.join(base_directory, "1"): instance_list_1,
    os.path.join(base_directory, "2"): instance_list_2,
    os.path.join(base_directory, "5"): instance_list_5,
    os.path.join(base_directory, "6"): instance_list_6,
    os.path.join(base_directory, "7"): instance_list_7
}

# Function to list and filter dataset files based on instance lists and directories
def list_dataset_files(instance_directories):
    filtered_files = []
    for directory, instance_list in instance_directories.items():
        if os.path.exists(directory):  # Ensure the directory exists
            # List all files in the directory that match the instance list
            filtered_files += [os.path.join(directory, f) for f in os.listdir(directory)
                               if f.endswith('.csv') and any(instance in f for instance in instance_list)]
        else:
            st.warning(f"Directory {directory} does not exist")
    return filtered_files

# Assuming instance_directories has been defined with appropriate lists and directories
# List dataset files filtered by instance names and directories
data_files = list_dataset_files(instance_directories)

# Create a dropdown in Streamlit to select a dataset
selected_data = st.selectbox("Select a dataset: ", data_files)
print(selected_data)
columns = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP', 'P-JUS-CKGL', 'T-JUS-CKGL','QGL', 'class']
sensor_columns = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP', 'P-JUS-CKGL', 'T-JUS-CKGL','QGL']

def update_class_label(instance_n, class_column):
    class_mappings = {
        1: {0: 0, 101: 1, 1: 1},
        2: {0: 0, 102: 1, 2: 1},
        3: {0: 0, 103: 1, 3: 1},
        4: {0: 0, 104: 1, 4: 1},
        5: {0: 0, 105: 1, 5: 1},
        6: {0: 0, 106: 1, 6: 1},
        7: {0: 0, 107: 1, 7: 1},
        8: {0: 0, 108: 1, 8: 1}
    }
    
    if instance_n in class_mappings:
        return class_column.replace(class_mappings[instance_n])
    # else:
    #     print("instance_number is out of range")

    
instance_n = selected_data[10]
print(instance_n)

# Only proceed if a dataset has been selected
if selected_data:
    # Load the selected dataset into a DataFrame
    df = pd.read_csv(selected_data)
    
    # Display the DataFrame and its head in the Streamlit app
    st.write("Selected Dataset:")
    # st.dataframe(df)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index() 

    clean_vdf, smooth_vdf = handle_missing_data(df, df.columns)
    # print(smooth_vdf.head())
    smooth_vdf = smooth_vdf[smooth_vdf['class'].notnull()]
    # print(smooth_vdf.head())

    cols_to_check = smooth_vdf.columns.difference(['class']) 
    z_score_vdf = z_score_outlier(smooth_vdf, cols_to_check)
    resample_vdf = time_windowing(smooth_vdf)
    print(resample_vdf.head())



    if 'class' in resample_vdf.columns:
        X_vdf = resample_vdf[cols_to_check]
        y_vdf = resample_vdf['class']
        window = resample_vdf['id']

        scale_vdf = scale_data(resample_vdf, cols_to_check )
        scale_vdf = scale_vdf.reset_index()
        scale_vdf.rename(columns={'timestamp':'time'}, inplace = True)
        # print(scale_vdf.head())

        X_val = extract_and_impute_features(scale_vdf)
        y_val = scale_vdf.groupby('id')['class'].first()
        y_val = update_class_label(instance_n, y_val)
        # Optionally, print the first few rows in the console for debugging (if running locally)
        print(X_val.head())


    # Make prediction
    # Load the trained Random Forest model
    model_path = f"./ain/predictionModel/{instance_n}/best_random_forest_model.pkl"
    print(model_path)
    best_random_forest_model = joblib.load(model_path)
    y_pred = best_random_forest_model.predict(X_val)
    # print(y_pred)

    # Add predictions to the DataFrame
    output_df = X_val.copy()
    output_df['y_pred'] = y_pred
    output_df['class'] = y_val
    # print(output_df)

    # add predictions to sensor dataframe
    # Ensure that the index of model_sample_data is named as 'id' for easy merging
    output_df = output_df.rename_axis('id')

    # Reset the index on df2 to align 'timestamp' as a column
    resample_vdf = resample_vdf.reset_index()

    # Perform the left join on the 'timestamp' column
    display_df = smooth_vdf.merge(resample_vdf[['timestamp', 'id']], on='timestamp', how='left')

    # Set the 'timestamp' back as the index
    display_df.set_index('timestamp', inplace=True)

    output_df = output_df.reset_index()
    display_df = display_df.merge(output_df[['id', 'y_pred']], on='id', how='left')
    display_df['y_pred'] = display_df['y_pred'].fillna(method='ffill')

    # print(len(df),len(display_df))
    # print(df.head())
    # print(display_df.head())

    # Title of the app
    st.title('3W Data')
    st.dataframe(df)
    # st.dataframe(display_df[display_df.columns.difference(['class','id'])])


    # Set the title of the Streamlit app
    st.title('Sensor Data Visualization')

    # Reset the index for df to expose timestamp
    df_reset = df.reset_index()
    print(df_reset.head())

    # Generate separate charts for each sensor column
    for sensor in cols_to_check:
        # Create figure
        fig = go.Figure()

        # Plot sensor data from df
        fig.add_trace(go.Scatter(x=df_reset['timestamp'], y=df[sensor], mode='lines', name=f'{sensor} (raw)'))

        # Plot sensor data from df_display
        fig.add_trace(go.Scatter(x=df_reset['timestamp'], y=display_df[sensor], mode='lines', name=f'{sensor} (resampled)', line=dict(dash='dash')))

        # Plot y_pred on secondary y-axis
        fig.add_trace(go.Scatter(x=df_reset['timestamp'], y=display_df['y_pred'], mode='lines', name='well status prediction (0: normal | 1: faulty)', yaxis='y2', line=dict(color='red')))

        # Update layout for dual y-axis
        fig.update_layout(
            title=f"{sensor} with well status prediction",
            xaxis_title="Timestamp",
            yaxis_title=f"{sensor} Values",
            yaxis2=dict(
                title="y_pred",
                overlaying='y',
                side='right'
            ),
            legend_title="Legend"
        )

        # Display each figure in Streamlit
        st.plotly_chart(fig)


    # explainable AI


    # Prepare the data for SHAP
    X_val = extract_and_impute_features(scale_vdf)
    explainer = shap.TreeExplainer(best_random_forest_model)
    shap_values = explainer.shap_values(X_val)

    # Save SHAP summary plot as an image
    summary_plot_path = '/tmp/shap_summary_plot.png'
    shap.summary_plot(shap_values, X_val, show=False)  # Generate plot
    plt.savefig(summary_plot_path)  # Save plot as PNG
    plt.close()  # Close plot to free memory

    # Display the saved image in Streamlit
    st.title('SHAP Summary Plot')
    image = Image.open(summary_plot_path)
    st.image(image, caption='SHAP Summary Plot', use_column_width=True)


        
else:
    st.warning("Please select a dataset to proceed.")



