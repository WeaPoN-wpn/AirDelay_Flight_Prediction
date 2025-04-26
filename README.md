# ✈️ Flight Delay Prediction & LLM-Based Chatbot
This project predicts flight delay categories using historical and real-time data. It also features an interactive chatbot powered by a local LLM to support natural language flight queries. The app is built with Streamlit, integrates batch and streaming pipelines via Azure, and offers explainable model outputs.

## ‼️Prerequisite
1. Ollama needs to be installed and configured locally in advance.

    - **Reference Link**: [Ollama Installation Tutorial](https://www.cnblogs.com/obullxl/p/18295202/NTopic2024071001)

2. If you want to change the model, please change `def query_ollama_chat()` in **llm_dialogue_2.py** accordingly.
    
    [library (ollama.com)](https://ollama.com/library) is the ollama model library, please search for the model you need and launch it before running this project.

    - **Reference Link**: [Ollama GitHub](https://github.com/ollama/ollama)

## 🚀 Features Overview

### Part 1: Prediction Interface
- User inputs flight details (airline, airports, datetime, etc.)
- Model returns delay category and probability distribution in real-time

![Prediction Interface](figs/Prediction_Interface.png)

### Part 2: Exploratory Data Analysis (EDA)
- Delay category distribution
- Histograms of numerical features
- Average arrival delays per airline
- Most delayed airports
- Delay trends by time-related features

![EDA of Training Data](figs/Traindata_EDA.png)

### Part 3: LLM-Based Flight Chatbot
- Users can describe flights in natural language
- The system extracts structured information (slots) through LLM
- If information is incomplete, follow-up questions are generated
- The final prediction is explained to the user

![Chatbot](figs/Chatbot.png)


