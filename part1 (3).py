import streamlit as st
import json
import os
import pandas as pd
import numpy as np
import subprocess
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import plotly.graph_objects as go

# Set up paths
REPO_NAME = "edgar-crawler"
REPO_URL = "https://github.com/nlpaueb/edgar-crawler.git"
REPO_PATH = os.path.join(os.getcwd(), REPO_NAME)
EXTRACTED_FOLDER = os.path.join(REPO_PATH, "datasets", "EXTRACTED_FILINGS", "10-K")

# Load FinBERT model and tokenizer
@st.cache_resource
def load_finbert():
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_finbert()

def analyze_sentiment(text):
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.detach().numpy()[0]
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits)
    sentiment_labels = ["negative", "neutral", "positive"]
    return dict(zip(sentiment_labels, probabilities))

def extract_all_json_content(folder_path):
    extracted_content = []
    if not os.path.exists(folder_path):
        st.error(f"Error: The folder '{folder_path}' does not exist.")
        return extracted_content

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            try:
                parts = file_name.replace(".json", "").split("_")[:3]
                if len(parts) < 3:
                    st.warning(f"Skipping invalid filename: {file_name}")
                    continue
                cik, filing_type, year = parts
                with open(file_path, 'r') as f:
                    content = json.load(f)
                content.update({"cik": cik, "filing_type": filing_type, "year": year})
                extracted_content.append(content)
            except Exception as e:
                st.error(f"Error reading {file_name}: {e}")
    return extracted_content

st.title("EDGAR 10-K Filings Sentiment Analysis")
ticker = st.text_input("Enter Ticker Symbol:")
year = st.number_input("Enter Start Year:", min_value=2000, max_value=2025, step=1)

if st.button("Analyze"):
    if ticker and year:
        with st.spinner("Extracting filings..."):
            data = extract_all_json_content(EXTRACTED_FOLDER)

        company_dfs = {}
        for report in data:
            company_name = report.get('company', 'Unknown')
            report_year = report.get('year', 'Unknown')
            if company_name not in company_dfs:
                company_dfs[company_name] = pd.DataFrame(columns=['year'] + [f'item_{i}' for i in range(1, 17)])
            row = {'year': report_year}
            for item in range(1, 17):
                item_key = f'item_{item}'
                if item_key in report:
                    sentiment_scores = analyze_sentiment(report[item_key])
                    row[item_key] = sentiment_scores["positive"]
                else:
                    row[item_key] = None
            company_dfs[company_name] = pd.concat([company_dfs[company_name], pd.DataFrame([row])], ignore_index=True)

        for company, df in company_dfs.items():
            st.subheader(f"Sentiment Scores for {company}")
            st.dataframe(df)
            df['Average'] = df.mean(axis=1)
            fig = px.line(df, x='year', y='Average', title=f"Sentiment Over Time for {company}")
            st.plotly_chart(fig)

            st.subheader("Descriptive Statistics")
            st.write(df.describe())

            st.subheader("Correlation Matrix")
            corr_matrix = df.corr()
            fig_corr = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='Viridis'))
            st.plotly_chart(fig_corr)

        csv = df.to_csv(index=False)
        st.download_button("Download Data", csv, "sentiment_analysis.csv", "text/csv")
    else:
        st.error("Please enter both Ticker Symbol and Start Year.")
