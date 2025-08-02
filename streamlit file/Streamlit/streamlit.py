import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import networkx as nx
import os
import seaborn as sns
import io
import sklearn
import plotly.express as px



logo_path = 'logo.png'  # logo path
cover_image_path = 'cover_image.png'  # Cover
svg_path = 'Methodology.svg'  # SVG Methodology
df=pd.read_csv("../cl_JUIN_2013-complet3.csv",encoding='ISO-8859-1', sep=';')
df_metrics=pd.read_csv("df_metrics.csv",encoding='ISO-8859-1', sep=';')
df_missing_data=pd.read_csv("df_missing_data.csv",encoding='ISO-8859-1', sep=';')
dt_outliers=pd.read_csv("outliers.csv",encoding='ISO-8859-1', sep=';')

# Chapter titles
chapters = {
    "Introduction": "📖",
    "Data mining and Visualization": "📊",
    "Pre-processing": "🔧",
    "Feature Engineering": "🛠️",
    "Modeling": "📈",
    # "Interpretation of results": "🔍",
    "Conclusion": "🏁",
    "Application": "🚗"
}

# Sidebar with logo and table of contents
st.sidebar.image(logo_path, use_column_width=True)
st.sidebar.title("Agenda")
selected_chapter = st.sidebar.radio("", ["🏠 Home"] + [f"{icon} {chapter}" for chapter, icon in chapters.items()])

# Display the cover page by default
if selected_chapter == "🏠 Home":
    st.title("CO2 Emissions Prediction Project")
    st.image(cover_image_path, use_column_width=True)
    st.write("### Team Members")
    st.write("""
        👨🏻‍🎓 **Abd Akdim**\n
        👨🏻‍🎓 **Halimeh Agh**\n
        👩‍🎓 **Azangue Pavel**
    """)
    st.write("### Supervisor")
    st.write("""
        👩‍🏫 **Sarah Lenet**
    """)
    st.write("### University")
    st.write("""
        🏫 **DataScientest Paris**
    """)

else:
    st.title(selected_chapter.split(' ')[-1])
    if selected_chapter == "📖 Introduction":
        st.write("""
        
        - :red[**Objective:**]
        The goal of this project is to develop a model to predict the CO2 emissions of cars based on vehicle characteristics.\n
        """)
        
        st.markdown("""
        ##### Methodology:
        - :red[**Data Mining and Visualization:**] Data is collected and exploratively analyzed to gain insights.
        - :red[**Pre-processing:**] Involves cleaning, transforming, and preparing data for further analysis.
        - :red[**Feature Engineering:**] Extracts and combines features to enhance the predictive models' performance.
        - :red[**Modeling:**] Applies machine learning algorithms to identify patterns and make predictions based on the prepared data.
        - :red[**Interpretation of Results:**] Analyzes the model outcomes to derive conclusions and actionable insights.
        - :red[**Conclusion:**] Summarizes the findings and draws final conclusions from the entire analysis process, providing a clear overview of the insights gained.
        """)
        # SVG
        st.image(svg_path, width=800)
        # st.write("joblib version:", joblib.__version__)
        # st.write("scikit-learn version:", sklearn.__version__)
        
    elif selected_chapter == "📊 Data mining and Visualization":
        st.write(""" 
        The dataset contains 44,850 entries and 26 columns, representing various properties of cars recorded in France in 2013. 
        Key characteristics include fuel type, vehicle model name, fuel consumption, CO2 emissions, and other technical details.
        The table below shows the first 5 rows of our dataset.
        
                   """)
    
        st.dataframe(df.head()) 
        st.write(""" the table below displays all data for each column, including those with missing data""")
        st.dataframe(df_missing_data.head(26))
        st.write(""" Columns such as HC (g/km), HC+NOX (g/km), Particles (g/km), show the highest percentages of missing data. 
        Identifying such columns allows for prioritizing data cleaning to ensure the accuracy and usability of the information.""")
        st.write("""
        #### Numerical Variables:          
        Numerical variables in data analysis represent numeric values. They are used for quantitative measurements such as:\n
        - :red[**Mean:**] $$ \overline{x} $$ is the value obtained by summing all data points $$ {x_i} $$ divided by the number of data points $$ {n} $$
        - :red[Standard deviation $\sigma$:] (Std or $\sigma$) measures the average deviation of data points $$ {x_i} $$ from the mean
        - :red[Minimum and Maximum:] the smallest Min. and the largest value in the dataset.
            
        These values are essential for understanding central tendency, variability, and distribution
        of data, which are fundamental for statistical analysis and modeling
       
                 
         """)
        st.dataframe(dt_outliers.head(26))
        
        # List of attributes for the boxplots
        attributes_emission = ['CO type I (g/km)', 'HC (g/km)', 'NOX (g/km)', 'HC+NOX (g/km)', 'Particules (g/km)']

        attributes_masse = ['masse vide euro min (kg)', 'masse vide euro max (kg)']

        # Create subplots with smaller figure size
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 6)) 

        # Plot Emission attributes
        sns.boxplot(data=df[attributes_emission], ax=axs[0], palette='Set2', width=0.5) 
        axs[0].set_title('Boxplots for Emission Attributes', fontsize=14)  
        axs[0].set_xlabel('Attribute', fontsize=12)  
        axs[0].set_ylabel('Value', fontsize=12)  
        axs[0].tick_params(axis='x', rotation=45, labelsize=10)  
        axs[0].tick_params(axis='both', which='major', labelsize=10)  

        # Plot Masse attributes
        sns.boxplot(data=df[attributes_masse], ax=axs[1], palette='Set2', width=0.5)  
        axs[1].set_title('Boxplots for Vehicle Mass Attributes', fontsize=14)  
        axs[1].set_xlabel('Attribute', fontsize=12)  
        axs[1].set_ylabel('Value', fontsize=12) 
        axs[1].tick_params(axis='x', rotation=45, labelsize=10) 
        axs[1].tick_params(axis='both', which='major', labelsize=10)  

        # Adjust layout and save the plot as PNG
        plt.tight_layout()
        st.pyplot(fig)

        st.write(""" #### Categorical Variables:
                 
         """)


    elif selected_chapter == "📊 Data mining and Visualization":
        st.write("""
        
        """)

    elif selected_chapter == "🔧 Pre-processing":
        st.write("""
       
        """)

    elif selected_chapter == "🛠️ Feature Engineering":
        st.write("""
     
        """)

    elif selected_chapter == "📈 Modeling":
        st.write("""
                 """)

# Data for the metrics
        metrics_data = {
            "Mean Squared Error (MSE)": {
                "Equation": r"$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$",
                "Purpose": "Measures the average squared difference between actual and predicted values. Lower values indicate more accurate predictions."
            },
            "R² (R-Squared)": {
                "Equation": r"$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$",
                "Purpose": "Measures the proportion of variance in the dependent variable explained by the model. Higher values are better."
            },
            "Mean Absolute Error (MAE)": {
                "Equation": r"$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$",
                "Purpose": "Measures the average magnitude of errors between actual and predicted values. Lower values indicate more accurate predictions."
            },
            "Root Mean Squared Error (RMSE)": {
                "Equation": r"$\text{RMSE} = \sqrt{\text{MSE}}$",
                "Purpose": "Provides errors in the same units as the original data and is sensitive to large errors. Lower values are better."
            },
            "Mean Squared Logarithmic Error (MSLE)": {
                "Equation": r"$\text{MSLE} = \frac{1}{n} \sum_{i=1}^{n} (\log(1 + y_i) - \log(1 + \hat{y}_i))^2$",
                "Purpose": "Measures the squared error between the logarithms of actual and predicted values. Lower values are better."
            },
            "Median Absolute Error (MedAE)": {
                "Equation": r"$\text{MedAE} = \text{Median}(|y_i - \hat{y}_i|)$",
                "Purpose": "Measures the median of the absolute errors and is robust to outliers. Lower values indicate more accurate predictions."
            },
            "Max Error (Max Err)": {
                "Equation": r"$\text{Max Err} = \max(|y_i - \hat{y}_i|)$",
                "Purpose": "Measures the largest absolute error between actual and predicted values. Lower values are better."
            },
            "Explained Variance Score (EVS)": {
                "Equation": r"$\text{EVS} = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}$",
                "Purpose": "Measures how well the variance of the actual data is explained by the model. Values close to 1 indicate better performance."
            },
            "Mean Absolute Percentage Error (MAPE)": {
                "Equation": r"$\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$",
                "Purpose": "Measures the average percentage error between actual and predicted values. Lower values are better."
            },
            "Adjusted R² (R² Adjusted)": {
                "Equation": r"$\text{Adjusted R}^2 = 1 - \left(1 - R^2\right) \frac{n - 1}{n - p - 1}$",
                "Purpose": "Accounts for the number of predictors in the model and penalizes adding irrelevant predictors. Higher values are better."
            }
        }

        # Create a sidebar with explanations
        st.sidebar.write("### Metric Explanations")
        st.sidebar.write("""
        - $$ \hat{y}_i $$ : The predicted values.
        - $$ {y}_i $$ :  The actual values.
        - $$ \overline{y} $$ : The mean of the actual values.
                 """)
        metric_names = list(metrics_data.keys())
        selected_metric = st.sidebar.selectbox("Select a metric to see its details:", metric_names)

        # Display the selected metric's details
        if selected_metric:
            st.sidebar.subheader(selected_metric)
            st.sidebar.markdown(f"**Equation:** {metrics_data[selected_metric]['Equation']}")
            st.sidebar.markdown(f"**Purpose:** {metrics_data[selected_metric]['Purpose']}")

        # Main content
        st.markdown("""
                    #### Model Evaluation Metrics
                    """)
        
        # Text with clickable metric names
        text = (
            "Metrics in data science are standards used to evaluate the performance and"
            "quality of models or algorithms. They are crucial for objectively assessing"
            "how well a model makes predictions and for comparing different approaches"
            "In our project, we used several metrics to evaluate the quality of the models. For example,"
            "the **Mean Squared Error (MSE)**, **R² (R-Squared)**, and **Mean Absolute Percentage Error (MAPE)** Etc... "
        )
        st.markdown(text)

        st.markdown("""
                    #### Applied Machine Learning Methods
                    In this chapter, we evaluate various machine learning models for predicting CO2 emissions (g/km) 
                    of different car types, examining their implementation and effectiveness.
                    """)

#--------------------------------------------------------
        def plot_metrics(df, selected_models, metrics, metric_group_name):
            df_filtered = df[df['Model'].isin(selected_models)]
            df_filtered = df_filtered[['Model'] + metrics]

            # Melt the DataFrame for Plotly
            df_melted = pd.melt(df_filtered, id_vars='Model', var_name='Metric', value_name='Value')

            # Create Plotly bar plot
            fig = px.bar(df_melted, x='Metric', y='Value', color='Model', barmode='group',
                        title=f'Comparison of Metrics for Selected Models ({metric_group_name})',
                        labels={'Value': 'Metric Value', 'Metric': 'Metric'},
                        color_discrete_sequence=px.colors.qualitative.Plotly)

            # Update layout
            fig.update_layout(
                xaxis_title='Metric',
                yaxis_title='Value',
                legend_title='Model',
                title_x=0.5,
                title_y=0.95
            )

            return fig


        # Section 1: Model Comparison - Regression Methods
        st.markdown("""
            ##### :red[Regression Methods:] \n
            Regression methods are essential in machine learning for predicting continuous outcomes based on input features. 
            These methods model the relationship between dependent and independent variables to estimate future values or trends.
            For applying these models, we divided our data into training and testing sets, 80% of the data was allocated for training, while 20% was reserved for testing.\n 
            - The following diagrams show the results of our regression models:
        """)

        # List of models to compare
        models_to_compare_reg = [
            'Linear Regression',
            'Linear Regression with PCA',
            'Ridge Regression',
            'Ridge Regression with PCA',
            'Lasso Regression',
            'ElasticNet Regression',
            'Decision Trees',
            'Decision Trees with PCA'
        ]

        # Metric groups
        metrics1 = ['MSE', 'MAE', 'RMSE', 'MSLE']
        metrics2 = ['R2', 'R2 Adjusted', 'Explained Variance Score']

        # Select models from the available DataFrame
        selected_models_reg = st.multiselect('Select regression models to compare', models_to_compare_reg, default=models_to_compare_reg, key='reg_model_selector')

        # Select metric group for regression methods
        metric_group_reg = st.radio('Select metric group for regression models', ['Group 1', 'Group 2'], key='reg_metric_selector')

        # Filter metrics based on the selected group for regression methods
        metrics_reg = metrics1 if metric_group_reg == 'Group 1' else metrics2

        # Plot and display
        fig_reg = plot_metrics(df_metrics, selected_models_reg, metrics_reg, metric_group_reg)
        st.plotly_chart(fig_reg)

        # Section 2: Model Comparison - Ensemble Methods
        st.markdown("""
            ##### :red[Ensemble Methods:] \n
            Ensemble methods combine the predictions of multiple models to improve performance. The following ensemble models are compared:
        """)

        # List of models to compare
        models_to_compare_ensemble = [
            'Random Forest',
            'Bagged Decision Trees',
            'Bagged SVM',
            'AdaBoost',
            'Gradient Boost',
            'XGBoost'
        ]

        # Select models for ensemble methods
        selected_models_ensemble = st.multiselect('Select ensemble models to compare', models_to_compare_ensemble, default=models_to_compare_ensemble, key='ensemble_model_selector')

        # Select metric group for ensemble methods
        metric_group_ensemble = st.radio('Select metric group for ensemble models', ['Group 1', 'Group 2'], key='ensemble_metric_selector')

        # Filter metrics based on the selected group for ensemble methods
        metrics_ensemble = metrics1 if metric_group_ensemble == 'Group 1' else metrics2

        # Plot and display
        fig_ensemble = plot_metrics(df_metrics, selected_models_ensemble, metrics_ensemble, metric_group_ensemble)
        st.plotly_chart(fig_ensemble)

        # Model Comparison - Deep Learning Techniques
        st.markdown("""
            ##### :red[Deep Learning Techniques:] \n
            Deep learning techniques use neural networks with many layers to analyze and predict outcomes based on input data.
            - :red[Epochs:]	Number of complete passes through the training dataset.
            - :red[Batch Size:]	Number of training examples utilized in one iteration. 
            - :red[Learning Rate:]	Controls how much to change the model in response to the estimated error.
            - :red[Loss Function:]	Measure of how well the model's predictions match the true data.
            - :red[Activation Fuction:]	Function that determines the output of a neural network node.
            - :red[Optimizer:]	Algorithm for updating the model's parameters to minimize the loss.
            - :red[Early Stopping:]	Technique to stop training when the model's performance stops improving.
                    
                    
            The following table shows the parameters for deep learning models:
        """)
        parameters = {
            'Parameter': [
                'Epochs', 
                'Batch Size', 
                'Learning Rate', 
                'Loss Function', 
                'Activation Function', 
                'Optimizer', 
                'Early Stopping'
            ],
            'Value': [
                '100', 
                '32', 
                'Default (adam)', 
                'mse (Mean Squared Error)', 
                'relu (Rectified Linear Unit)', 
                'adam', 
                "Monitor='val loss', Patience=10, Restore best weights=True"
            ]
        }

        # Create a DataFrame
        df_parameters = pd.DataFrame(parameters)

        st.table(df_parameters)

        # List of models to compare
        models_to_compare_dl = [
            'MLP',
            'MLP with early stopping',
            'CNN',
            'CNN with early stopping',
            'RNN',
            'RNN with early stopping'
        ]

        # Select models for deep learning techniques
        selected_models_dl = st.multiselect('Select deep learning models to compare', models_to_compare_dl, default=models_to_compare_dl, key='dl_model_selector')

        # Select metric group for deep learning
        metric_group_dl = st.radio('Select metric group for deep learning models', ['Group 1', 'Group 2'], key='dl_metric_selector')

        # Filter metrics based on the selected group for deep learning
        metrics_dl = metrics1 if metric_group_dl == 'Group 1' else metrics2

        # Plot and display
        fig_dl = plot_metrics(df_metrics, selected_models_dl, metrics_dl, metric_group_dl)
        st.plotly_chart(fig_dl)

#========================================================================================================================================================================
    elif selected_chapter == "🏁 Conclusion":
        st.write("""
        - :red[**Model Results:**] Five models excelled: :red[**Randomforest**], :red[**XGBoost**], :red[**Gradient Boost**], :red[**Bagged Decision Trees**], :red[**Desision Trees**]. 

        - :red[**Recommendations:**] Use Decision Trees for small datasets; Random Forest for balanced performance; Gradient Boosting and XGBoost for high accuracy.

        - :red[**Key Predictors:**] Fuel consumption is the top predictor; emission factors are also important. Focus on efficient technologies and reducing emissions.

        - :red[**Limitations:**] Some features have low impact, requiring data reevaluation. Use regularization and cross-validation to avoid overfitting.

        - :red[**Strategic Focus:**] Invest in advanced engine tech and lightweight materials. Highlight fuel efficiency and low emissions in marketing to attract eco-conscious customers.
        """)

        # Display the performance metrics table
        with st.expander("Performance Metrics"):
            st.dataframe(df_metrics)

        # Create a container for comparisons
        st.write("#### Compare Models and Metrics")

        # Dropdown for selecting models
        models = st.multiselect("Select models to compare:", df_metrics['Model'].unique())

        # Dropdown for selecting metrics
        metrics = st.multiselect("Select metrics to compare:", df_metrics.columns[1:])

        if models and metrics:
            # Filter the DataFrame based on selected models
            df_filtered = df_metrics[df_metrics['Model'].isin(models)]

            # Create a plot for the selected metrics
            fig, ax = plt.subplots(figsize=(12, 6))

            for metric in metrics:
                sns.lineplot(data=df_filtered, x='Model', y=metric, marker='o', ax=ax, label=metric)

            ax.set_title('Model Comparison by Metrics')
            ax.set_xlabel('Model')
            ax.set_ylabel('Value')
            ax.legend(title='Metrics')
            plt.xticks(rotation=45)

            st.pyplot(fig)
        else:
            st.write("Please select at least one model and one metric.")

            

    elif selected_chapter == "🚗 Application":
        st.write("""
        
        """)

        # Input form in the sidebar
        with st.sidebar:
            st.header("Input Data")
            Input_Data = {
                "Consommation extra-urbaine (l/100km)": st.number_input("Consommation extra-urbaine (l/100km)", min_value=0, max_value=100),
                "Consommation mixte (l/100km)": st.number_input("Consommation mixte (l/100km)", min_value=0, max_value=100),
                "NOX (g/km)": st.number_input("NOX (g/km)", min_value=0, max_value=100),
                "Consommation urbaine (l/100km)": st.number_input("Consommation urbaine (l/100km)", min_value=0, max_value=100),
                "Carburant_GO": 1 if st.toggle("Carburant_GO - Yes/No", key="carburant_go") else 0,
                "Carburant_ES": 1 if st.toggle("Carburant_ES - Yes/No", key="carburant_es") else 0,
                "Puissance maximale (kW)": st.number_input("Puissance maximale (kW)", min_value=0, max_value=500),
                "Puissance administrative": st.number_input("Puissance administrative", min_value=0, max_value=500),
                "Mode UTAC_CADDY": 1 if st.toggle("Mode UTAC_CADDY - Yes/No", key="mode_utac_caddy") else 0,
                "CO type I (g/km)": st.number_input("CO type I (g/km)", min_value=0, max_value=100),
            }

            st.subheader("Your Input Data")
            st.write(pd.DataFrame.from_dict(Input_Data, orient='index', columns=['Value']).style.set_table_attributes('class="full-width-table"'))

        # Load the saved model
        model = joblib.load('rf_model.pkl')

        # Retrieve the feature names used during training
        model_features = model.feature_names_in_

        # Ensure the input data has the same features as the model expects
        input_data = pd.DataFrame([Input_Data], columns=model_features).reindex(columns=model_features, fill_value=0)

        # Prediction
        if st.button("Predict"):
            prediction = model.predict(input_data)
            co2_value = prediction[0]
            
            # Display the prediction in a styled box
            st.markdown(
                f"""
                <div style="background-color: rgb(255, 86, 68); padding: 20px; border-radius: 5px;">
                    <h2 style="color: white; text-align: center;">Predicted CO2 Emissions: {co2_value:.2f} g/km</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
