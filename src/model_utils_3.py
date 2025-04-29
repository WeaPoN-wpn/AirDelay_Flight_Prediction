import torch
import torch.nn as nn
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.feature_engineering_1 import map_categorical_to_ids
import os
import requests

BASE_DIR_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

def query_summary_llm(prompt, model_name="llama3"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }
        )
        return response.json().get("response", "[No response from local LLM]")
    except Exception as e:
        print("❌ Local summary LLM error:", str(e))
        return "[Local LLM failed]"

def load_label_encoders(path=None):
    if path is None:
        path = os.path.join(BASE_DIR_2, "label_encoders.json")
    with open(path, "r") as f:
        enc = json.load(f)
    encoders = {}
    for col, data in enc.items():
        le = LabelEncoder()
        le.classes_ = np.array(data["classes"])
        encoders[col] = le
    return encoders

class DelayPredictor(nn.Module):
    def __init__(self, embedding_dims, num_cont_features, num_classes):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat, emb_dim) for num_cat, emb_dim in embedding_dims
        ])
        emb_total_dim = sum([e for _, e in embedding_dims])
        self.mlp = nn.Sequential(
            nn.Linear(emb_total_dim + num_cont_features, 128),
            nn.ReLU(), nn.Linear(128, 64),
            nn.ReLU(), nn.Linear(64, num_classes)
        )

    def forward(self, x_cat, x_cont):
        x = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        return self.mlp(torch.cat([x, x_cont], dim=1))

def predict_and_explain(slots, features, embedding_dims, label_encoders, display_fn=print):
    cat_ids = map_categorical_to_ids(slots, label_encoders)
    x_cat = torch.tensor([[cat_ids[k] for k in ["Airline", "Origin", "Dest", "RushHour", "Season", "IsWeekend", "IsStartOrEndOfMonth"]]], dtype=torch.long)
    x_cont = torch.tensor([[features[k] for k in ["DepDelay", "ActualElapsedTime", "Airline_AvgDepDelay", "Airline_AvgArrDelay", "Origin_Avg_DepDelay", "Dest_Avg_ArrDelay"]]], dtype=torch.float32)

    model = DelayPredictor(embedding_dims, 6, 4)
    model_path = os.path.join(BASE_DIR_2, "best_model.pt")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        logits = model(x_cat, x_cont)
        probs_tensor = torch.softmax(logits, dim=1).numpy()[0]

    labels = ["Early(<0min)", "Minor Delay(0-30min)", "Moderate Delay(31-60min)", "Severe Delay(60+min)"]
    pred_label = labels[probs_tensor.argmax()]
    probs = {label: f"{float(prob)*100:.2f}%" for label, prob in zip(labels, probs_tensor)}

    summary_prompt = f"""
Based on the flight info below, explain the prediction of arrival delay simply and clearly:

Flight:
- Airline: {features['Airline']}
- From: {features['Origin']} → To: {features['Dest']}
- Departure Delay: {features['DepDelay']} minutes

Prediction:
Most likely outcome: {pred_label}
Class probabilities: {probs}
"""
    llm_response = query_summary_llm(
        f"You are a helpful assistant who summarizes predictions clearly and concisely.\n{summary_prompt}"
    )

    display_fn("✅ Prediction:", pred_label)
    display_fn("📊 Probabilities:", probs)
    display_fn("🧠 Explanation:", llm_response)
