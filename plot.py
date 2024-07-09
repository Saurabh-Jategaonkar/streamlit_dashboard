import streamlit as st
import pandas as pd
import duckdb
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import plotly.express as px
from pygwalker.api.streamlit import StreamlitRenderer


st.set_page_config(
    page_title="Research",
    page_icon=":bar_chart:",
    layout="wide")

st.title("Dashboard")
st.markdown("Prototype v0.1")

@st.cache_data
def load_data(file):
    file_extension = os.path.splitext(file.name)[1].lower()
    
    if file_extension == '.csv':
        data = pd.read_csv(file)
    elif file_extension in ['.xls', '.xlsx']:
        data = pd.read_excel(file)
    else:
        st.error(f"Unsupported file format: {file_extension}")
        return None
    
    ## light cleaning can be added here if needed
    return data

@st.cache_data
def plot_correlation_matrix(df):
    # Select only numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Calculate the correlation matrix
    correlation_matrix = df[numeric_columns].corr()

    # Create a heatmap using seaborn
    fig, ax = plt.subplots(figsize=(len(numeric_columns), len(numeric_columns)))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
    plt.title("Correlation Matrix")

    return fig

@st.cache_data
def plot_scatter_matrix(df):
    # Select only numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Create a scatter plot matrix
    fig, ax = plt.subplots(figsize=(len(numeric_columns) * 3, len(numeric_columns) * 3))
    pd.plotting.scatter_matrix(df[numeric_columns], ax=ax)
    plt.title("Scatter Plot Matrix")

    return fig

@st.cache_data
def plot_correlation_circle(df):
    # Select only numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Drop rows with NaN values
    df = df[numeric_columns].dropna(axis=0)

    # Create a correlation circle
    fig, ax = plt.subplots(figsize=(len(numeric_columns), len(numeric_columns)))
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df[numeric_columns])
    circle = plt.Circle((0, 0), 1, fill=False)
    ax.add_artist(circle)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
    for i, feature in enumerate(numeric_columns):
        ax.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], head_width=0.1, head_length=0.1, fc='r', ec='r')
        # ax.text(pca.components_[0, i] * 1.2, pca.components_[1, i] * 1.2, feature, color='r')
    plt.title("Correlation Circle")
    plt.axis('equal')

    return fig

@st.cache_resource
def get_pyg_renderer(df) -> "StreamlitRenderer":
    # If you want to use feature of saving chart config, set `spec_io_mode="rw"`
    return StreamlitRenderer(df, spec="./gw_config.json", spec_io_mode="rw")

@st.cache_data
def plotly_plot(df, x_field: str, y_field: str):
    fig = px.scatter(df, x=x_field, y=y_field)
    return fig
    

with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xls', 'xlsx'])

if uploaded_file is None:
    st.info("Upload a file through config")
    st.stop()

df = load_data(uploaded_file)


if df is not None:
    with st.expander("Data Preview"):
        st.dataframe(df)

    renderer = get_pyg_renderer(df)
    renderer.explorer()

    # Get the column names from the DataFrame
    column_names = df.select_dtypes(include=['float64', 'int64']).columns

    # Add second plot
    st.header("Scatter plot")
    # Allow the user to select the X and Y variables
    col1, col2 = st.columns(2)
    with col1:
        x_field = st.selectbox("Select the X variable", column_names)
    with col2:
        y_field = st.selectbox("Select the Y variable", column_names)

    st.plotly_chart(plotly_plot(df,x_field, y_field), on_select="rerun")
    

    # Add correlation matrix
    st.header("Correlation Matrix")
    col1, col2, col3 = st.columns(3, vertical_alignment="center")
    with col1:
        st.pyplot(plot_correlation_matrix(df))
    with col2:
        st.pyplot(plot_scatter_matrix(df))
    with col3:
        st.pyplot(plot_correlation_circle(df))
    
    
else:
    st.error("Failed to load the file. Please make sure it's a valid CSV or Excel file.")

