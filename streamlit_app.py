import streamlit as st
from datasets import load_dataset
from PIL import Image
import os
import requests
from collections import defaultdict
import pandas as pd
import sys
from io import StringIO

API_TOKEN = st.secrets.HF_API_KEY

def check_datasets(datasets):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    API_URL = f"https://datasets-server.huggingface.co/splits?dataset={datasets}"  
    response = requests.get(API_URL, headers=headers)
    data = response.json()

    return data

def shows_dataset_configs(data):
    datasets = list(set(item["config"] for item in data["splits"]))

    dataset_splits_dict = defaultdict(list)
    for item in data["splits"]:
        dataset_splits_dict[item["config"]].append(item["split"])

    return datasets, dataset_splits_dict


def shows_dataset_info(dataset, config, split):
    print(dataset, config, split)
    datasets = load_dataset(dataset, config, split=split+"[:5]")
    df = pd.DataFrame(datasets)
    st.write("Data: ")
    st.dataframe(df)

    datasets = load_dataset(dataset, config, split=split)
    df = pd.DataFrame(datasets)
    st.write("Total Rows:")
    st.write(datasets.num_rows)

    return df

st.set_page_config(
    page_title="HF Downloader",
    page_icon="🦜",
    menu_items={
        'About': "# This is a simple webapp to help you download datasets directly from Huggingface Datasets.",
        'Get help':  "https://huggingface.co/datasets"
    },
)
st.title('🦜Huggingface Datasets Downloader')
col_capt_1, col_capt_2 = st.columns([1,1])
with col_capt_1:
    st.caption('by: Rio Audino/AI Engineer Intern @GLAIR')
with col_capt_2:
    st.caption('Find your datasets here: https://huggingface.co/datasets')

if "run_process" not in st.session_state:
    st.session_state["run_process"] = False
    st.session_state["download-csv"] = False
    st.session_state["config"] = ""

def run_process():
    st.session_state["run_process"] = True

col1, col2 = st.columns([1,2])
option_config = None
option_split = None

with col1:   
    image = Image.open('hf-logo.png')
    st.image(image, caption='Huggingface')

with col2:
    dataset = st.text_input('Datasets Name', placeholder='truthful_qa')
    
    if dataset:
        data = check_datasets(dataset)
        if not "error" in data:
            configs, splits = shows_dataset_configs(data)
            col_datasets, col_split = st.columns(2)

            with col_datasets: 
                option_config = st.selectbox(
                    'config',
                    configs
                    )
            
            if option_config:
                split = splits.get(option_config)
            else:
                split = splits.get(configs[0])

            with col_split:
                option_split = st.selectbox(
                    'split',
                    split
                    )   

            if st.session_state["config"] != f"({dataset}, {option_config}, {option_split})":
                st.session_state["run_process"] = False
                st.session_state["download-csv"] = False
                st.session_state["config"] = f"({dataset}, {option_config}, {option_split})"
        else:
            st.error("Dataset not found! ❌")

st.button("Process Data", on_click=run_process)

if  st.session_state["run_process"] and not st.session_state["download-csv"]:
    if option_config and option_split:
        st.info("Wait for the process to finish...↺↺↺", icon="⏳")
        df = shows_dataset_info(dataset, option_config, option_split)
        csv = df.to_csv(index=False).encode('utf-8')

        with col2:
            filename=st.text_input(
                'Filename: ', 
                f'data_{dataset}')

            if filename:
                col_btn_1, col_btn_2, col_btn_3 = st.columns([2,2,1])
                with col_btn_2:
                    st.download_button(
                        ":green[Download]",
                        csv,
                        f"{filename}.csv",
                        "text/csv",
                        key='download-csv',
                    )
        
    else:
        st.warning("Please fill the data correctly", icon="⚠️")

if st.session_state["download-csv"]:
    st.success("Downloaded! ✔️")
    st.balloons()