# ✈️ Flight Delay Prediction & LLM-Based Chatbot
This project predicts flight delay categories using historical and real-time data. It also features an interactive chatbot powered by a local LLM to support natural language flight queries. The app is built with Streamlit, integrates batch and streaming pipelines via Azure, and offers explainable model outputs.

## ‼️Prerequisite
1. Ollama needs to be installed and configured locally in advance.

    - [Reference Link](https://www.cnblogs.com/obullxl/p/18295202/NTopic2024071001)

2. If you want to change the model, please change `def query_ollama_chat()` in **llm_dialogue_2.py** accordingly.
    
    [Ollama Model Library](https://ollama.com/library) - Please search for the model you need and launch it before running this project.

    - [Reference Link](https://github.com/ollama/ollama)
