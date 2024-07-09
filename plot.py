import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

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

with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is None:
    st.info("Upload a file through config")
    st.stop()

df = load_data(uploaded_file)

with st.expander("Data Preview"):
    st.dataframe(df)

# Add correlation matrix
st.header("Correlation Matrix")

# Select only numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

print(int(len(numeric_columns)))

# Calculate the correlation matrix
correlation_matrix = df[numeric_columns].corr()

# Create a heatmap using seaborn
fig, ax = plt.subplots(figsize=(len(numeric_columns), len(numeric_columns)))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
plt.title("Correlation Matrix")

# Display the plot in Streamlit
st.pyplot(fig)