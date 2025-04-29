import pandas as pd
import numpy as np
import pickle
import json
import pytz
import os
from dateutil import parser as date_parser

BASE_DIR_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
BASE_DIR_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
with open(os.path.join(BASE_DIR_1, "lookup_dicts.pkl"), "rb") as f:
    lookup = pickle.load(f)

with open(os.path.join(BASE_DIR_2, "embedding_dims.json"), "r") as f:
    embedding_dims = json.load(f)

airports_df = pd.read_csv(os.path.join(BASE_DIR_1, "airports_timezone.csv"))
airports_df = airports_df[airports_df["iata_code"].notnull()]
airports_df["name_lower"] = airports_df["name"].str.lower()
airports_df["city_lower"] = airports_df["municipality"].str.lower()

airport_timezones = dict(zip(airports_df["iata_code"], airports_df["timezone"]))

def find_iata_code(user_input):
    user_input = user_input.strip().lower()
    match = airports_df[
        (airports_df["iata_code"].str.lower() == user_input) |
        (airports_df["city_lower"] == user_input) |
        (airports_df["name_lower"].str.contains(user_input))
    ]
    if not match.empty:
        return match.iloc[0]["iata_code"]
    return None

def infer_season(month):
    return (
        "Winter" if month in [12, 1, 2] else
        "Spring" if month in [3, 4, 5] else
        "Summer" if month in [6, 7, 8] else
        "Fall"
    )

def get_rushhour(schedule_dep_time):
    try:
        dep_time = date_parser.parse(schedule_dep_time)
        dep_hour = dep_time.hour
        if dep_hour in [6, 7, 8, 9]:
            return "Morning Rush"
        elif dep_hour in [12, 13]:
            return "Midday Busy"
        elif dep_hour in [16, 17, 18, 19]:
            return "Evening Rush"
        else:
            return "Non-Rush Hour"
    except:
        return -1

def infer_weekend(day_of_week):
    return day_of_week in [6, 7]

def infer_start_or_end_of_month(day_of_month):
    return day_of_month <= 3 or day_of_month >= 28

def calculate_departure_delay(actual_departure, scheduled_departure):
    try:
        actual = date_parser.parse(actual_departure)
        scheduled = date_parser.parse(scheduled_departure)
        return int((actual - scheduled).total_seconds() / 60)
    except:
        return -1

def compute_elapsed_with_timezone(dep_str, arr_str, origin_code, dest_code):
    try:
        dep_zone = pytz.timezone(airport_timezones.get(origin_code))
        arr_zone = pytz.timezone(airport_timezones.get(dest_code))
        dep_time = dep_zone.localize(date_parser.parse(dep_str))
        arr_time = arr_zone.localize(date_parser.parse(arr_str))
        return int((arr_time.astimezone(pytz.utc) - dep_time.astimezone(pytz.utc)).total_seconds() / 60)
    except:
        return 0

def map_categorical_to_ids(slots, encoders):
    def safe_encode(col, value):
        try:
            return int(encoders[col].transform([value])[0])
        except:
            return 0
    return {
        "Airline": safe_encode("Airline", slots["Airline"]),
        "Origin": safe_encode("Origin", slots["Origin"]),
        "Dest": safe_encode("Dest", slots["Destination"]),
        "RushHour": safe_encode("RushHour", get_rushhour(slots["Schedule Departure Time"])),
        "Season": safe_encode("Season", infer_season(slots["Month"])),
        "IsWeekend": safe_encode("IsWeekend", infer_weekend(slots["Day of Week"])),
        "IsStartOrEndOfMonth": safe_encode("IsStartOrEndOfMonth", infer_start_or_end_of_month(slots["Day of Month"]))
    }

feature_required_for_model = [
    "Airline", "Origin", "Dest", "DepDelay", "ActualElapsedTime",
    "IsWeekend", "IsStartOrEndOfMonth", "RushHour", "Season",
    "Airline_AvgDepDelay", "Airline_AvgArrDelay",
    "Origin_Avg_DepDelay", "Dest_Avg_ArrDelay"
]
