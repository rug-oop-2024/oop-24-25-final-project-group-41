import streamlit as st
import pandas as pd


from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

st.title("Datasets")

st.markdown("""
- Manage datasets (upload, view, and manage CSV files)
""")


def create_dataset():
    st.header("Create Dataset")
    """create datset as one tab, using pandas to read in the csv file and display it in a dataframe
    """
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        
        # Dataset details 
        with st.form("dataset_details"):
            dataset_name = st.text_input("Dataset Name")
            dataset_version = st.text_input("Version", value="1.0.0")
            
            if st.form_submit_button("Create Dataset"):
                try:
                    # Create and save dataset
                    system = AutoMLSystem.get_instance()
                    asset_path = f"datasets/{dataset_name}"
                    dataset = Dataset.from_dataframe(
                        data=df,
                        name=dataset_name,
                        asset_path=asset_path,
                        version=dataset_version
                    )
                    system.registry.register.save(dataset)
                    st.success("Dataset created successfully!")
                except Exception as e:
                    st.error(f"Error creating dataset: {str(e)}")

    """list data sets as another tab, using the artifact registry to list the datasets
    """

def list_datasets():
    st.header("Existing Datasets")
    system = AutoMLSystem.get_instance()
    datasets = system.registry.list(type="dataset")
    
    if datasets:
        for dataset in datasets:
            st.write(f"Name: {dataset.name}, Version: {dataset.version}")
    else:
        st.info("No datasets found")

st.title("Dataset Management")
tab1, tab2 = st.tabs(["Create Dataset", "View Datasets"])

with tab1:
    create_dataset()
with tab2:
    list_datasets()