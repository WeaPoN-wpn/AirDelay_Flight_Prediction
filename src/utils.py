import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
import numpy as np

# 加载处理后的数据
def load_csv_with_check(filepath, expected_columns):
    df = pd.read_csv(filepath)
    for col in expected_columns:
        if col not in df.columns:
            raise KeyError(f'Expected column "{col}" not found in {filepath}')
    return df

airline_avg_dep_delay = load_csv_with_check('mapping/airline_avg_dep_delay.csv', ['Airline', 'Airline_AvgDepDelay'])
airline_avg_arr_delay = load_csv_with_check('mapping/airline_avg_arr_delay.csv', ['Airline', 'Airline_AvgArrDelay'])
origin_avg_dep_delay = load_csv_with_check('mapping/origin_avg_dep_delay.csv', ['Origin', 'Origin_Avg_DepDelay'])
dest_avg_arr_delay = load_csv_with_check('mapping/dest_avg_arr_delay.csv', ['Dest', 'Dest_Avg_ArrDelay'])

def find_avg_delays(airline, origin, dest):
    airline_dep_delay = airline_avg_dep_delay[airline_avg_dep_delay['Airline'] == airline]['Airline_AvgDepDelay'].values
    airline_arr_delay = airline_avg_arr_delay[airline_avg_arr_delay['Airline'] == airline]['Airline_AvgArrDelay'].values
    origin_dep_delay = origin_avg_dep_delay[origin_avg_dep_delay['Origin'] == origin]['Origin_Avg_DepDelay'].values
    dest_arr_delay = dest_avg_arr_delay[dest_avg_arr_delay['Dest'] == dest]['Dest_Avg_ArrDelay'].values
    
    return {
        "Airline_AvgDepDelay": airline_dep_delay[0] if len(airline_dep_delay) > 0 else None,
        "Airline_AvgArrDelay": airline_arr_delay[0] if len(airline_arr_delay) > 0 else None,
        "Origin_Avg_DepDelay": origin_dep_delay[0] if len(origin_dep_delay) > 0 else None,
        "Dest_Avg_ArrDelay": dest_arr_delay[0] if len(dest_arr_delay) > 0 else None,
    }

def classify_rush_hour(hour):
    try:
        hour = int(hour)
        if 6 <= hour < 10:
            return "Morning Rush"
        elif 12 <= hour < 14:
            return "Midday Busy"
        elif 16 <= hour < 20:
            return "Evening Rush"
        else:
            return "Non-Rush Hour"
    except:
        return None

def preprocess_date(df, date, dep_time):
    df["Month"] = date.dt.month
    df["DayofMonth"] = date.dt.day
    df["DayOfWeek"] = date.dt.weekday + 1  # Monday=0, Sunday=6
    df["Season"] = df["Month"].apply(
        lambda x: "Winter" if x in [12, 1, 2] else 
                  "Spring" if x in [3, 4, 5] else 
                  "Summer" if x in [6, 7, 8] else 
                  "Fall"
    )
    df["DepHour"] = dep_time.str[:2].astype(int)
    df["RushHour"] = df["DepHour"].apply(classify_rush_hour)
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7])
    df["IsStartOrEndOfMonth"] = df["DayofMonth"].apply(lambda x: x <= 3 or x >= 28)
    return df

def create_features(df, airline, origin, dest, dep_delay, elapsed_time):
    avg_delays = find_avg_delays(airline, origin, dest)
    df["Airline"] = airline
    df["Origin"] = origin
    df["Dest"] = dest
    df["DepDelay"] = dep_delay
    df["ActualElapsedTime"] = elapsed_time
    df["Airline_AvgDepDelay"] = avg_delays["Airline_AvgDepDelay"]
    df["Airline_AvgArrDelay"] = avg_delays["Airline_AvgArrDelay"]
    df["Origin_Avg_DepDelay"] = avg_delays["Origin_Avg_DepDelay"]
    df["Dest_Avg_ArrDelay"] = avg_delays["Dest_Avg_ArrDelay"]
    return df

def encode_categorical_columns(df, encoders, columns_to_encode):
    for col in columns_to_encode:
        if col in df.columns:
            df[col + '_ID'] = encoders[col].transform(df[col])
    return df

def load_label_encoders(filepath='models/label_encoders.json'):
    with open(filepath, 'r') as f:
        json_encoders = json.load(f)
    encoders = {}
    for col, value in json_encoders.items():
        le = LabelEncoder()
        le.classes_ = np.array(value['classes'])
        encoders[col] = le
    return encoders