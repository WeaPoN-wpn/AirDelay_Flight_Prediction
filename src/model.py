import torch
import torch.nn as nn
import json
from src.utils import encode_categorical_columns

class DelayPredictor(nn.Module):
    def __init__(self, embedding_dims, num_cont_features, num_classes):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat, emb_dim) for num_cat, emb_dim in embedding_dims
        ])
        emb_total_dim = sum([e for _, e in embedding_dims])
        self.mlp = nn.Sequential(
            nn.Linear(emb_total_dim + num_cont_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_cat, x_cont):
        x = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        x = torch.cat([x, x_cont], dim=1)
        return self.mlp(x)

def load_model(model_path='models/best_model.pt', embedding_dims_path='models/embedding_dims.json'):
    with open(embedding_dims_path, 'r') as f:
        embedding_dims = json.load(f)

    num_cont_features = 6  # number of continuous features
    num_classes = 4  # number of output classes

    model = DelayPredictor(embedding_dims, num_cont_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(model, encoders, df):
    columns_to_encode = ["Airline", "Origin", "Dest", "RushHour", "Season", "IsWeekend", "IsStartOrEndOfMonth"]
    df = encode_categorical_columns(df, encoders, columns_to_encode)
    X_cat = df[[col + "_ID" for col in columns_to_encode]].values.astype('int64')
    X_cont = df[[
        "DepDelay",
        "ActualElapsedTime",
        "Airline_AvgDepDelay",
        "Airline_AvgArrDelay",
        "Origin_Avg_DepDelay",
        "Dest_Avg_ArrDelay"
    ]].values.astype('float32')

    X_cat = torch.tensor(X_cat, dtype=torch.int64)
    X_cont = torch.tensor(X_cont, dtype=torch.float32)

    with torch.no_grad():
        preds = model(X_cat, X_cont)
        prob = torch.softmax(preds, dim=1)
        category = preds.argmax(dim=1).cpu().numpy()

    return category, prob.cpu().numpy()