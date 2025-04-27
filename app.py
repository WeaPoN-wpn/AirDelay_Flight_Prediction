import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import pytz
import os
from dateutil import parser as date_parser
from src.utils import preprocess_date, create_features, load_label_encoders
from src.model import load_model, predict
from src.llm_dialogue_2 import extract_slots_loop
from src.feature_engineering_1 import *
from src.model_utils_3 import load_label_encoders, predict_and_explain

BASE_DIR_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "data")) # path of visualization_dataset.parquet
BASE_DIR_MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))# path of model

st.set_page_config(page_title="AIRDelayWise", layout="wide")

# Sidebar navigation
section = st.sidebar.radio("Select Application Section", [
    "🔮 Flight Delay Prediction",
    "📊 Exploratory Data Analysis",
    "🧠 LLM-based Flight Prediction Assistant"
])


# 🔮 Part 1: Prediction Module
if section == "🔮 Flight Delay Prediction":
    st.title('AIRDelayWise - Flight Delay Predictor')
    st.markdown("""
        Please enter the flight information below. Our model will predict whether the flight is delayed, and if so, the delay category.
    """)

    # 用户输入
    embedding_cols = ["Airline", "Origin", "Dest"]
    dummy_cols = ["RushHour", "Season", "IsWeekend", "IsStartOrEndOfMonth"]

    airline_input = st.text_input('✈️ Enter Airline IATA Code:', '9E')
    origin_input = st.text_input('🛫 Enter Origin Airport IATA Code:', 'BGM')
    dest_input = st.text_input('🛬 Enter Destination Airport IATA Code:', 'DTW')
    date_input = st.date_input('📅 Select Flight Date:', pd.Timestamp(2023, 1, 1))
    dep_time_input = st.text_input('⏰ Enter Departure Time (HHMM):', '1600')
    dep_delay_input = st.text_input('⏳ Enter Departure Delay (minutes):', '0')
    elapsed_time_input = st.text_input('🕒 Enter Flight Duration (minutes):', '60')

    # 加载模型和编码器
    model = load_model()
    encoders = load_label_encoders()

    # 创建 DataFrame
    df_inputs = pd.DataFrame({
        'date': [pd.to_datetime(date_input)],
        'DepTime': [dep_time_input]
    })

    # 预处理日期并生成特征
    df_inputs = preprocess_date(df_inputs, df_inputs['date'], df_inputs['DepTime'])
    df_inputs = create_features(df_inputs, airline_input, origin_input, dest_input, float(dep_delay_input), float(elapsed_time_input))

    # 预测
    category, prob = predict(model, encoders, df_inputs)

    # 显示输入的数据
    st.json(df_inputs.to_dict(orient='records')[0])

    # 显示预测概率
    st.subheader('📊 Prediction:')
    label_enc_dict = encoders['DelayCategory']
    classes = label_enc_dict.classes_

    prob_percent = [f"{p * 100:.2f}%" for p in prob[0]]
    for cls, prob in zip(classes, prob_percent):
        st.markdown(f"The probability of **{cls}** is : {prob}")

    # 显示最可能的延误类别和表情
    # Display the most likely delay category and emoji
    st.markdown(f"### 🎯 Predicted Delay Category: **{classes[category[0]]}**")

    if category[0] == 0:
        st.markdown("### 🟢 Flight will be on time or early!")
    elif category[0] == 1:
        st.markdown("### 🟡 Flight will have a minor delay.")
    elif category[0] == 2:
        st.markdown("### 🟠 Flight will have a moderate delay.")
    else:
        st.markdown("### 🔴 Flight will have a severe delay.")

## 📊 Part 2: EDA Dashboard
elif section == "📊 Exploratory Data Analysis":
    st.title("✈️ AIRDelayWise - Exploratory Data Analysis")

    @st.cache_data
    def load_data():
        file_path = os.path.join(BASE_DIR_DATA, "visualization_dataset.parquet")
        return pd.read_parquet(file_path)
    df = load_data()

    # ========== Feature engineering ==========
    df["DelayCategory"] = pd.cut(df["ArrDelay"], bins=[-np.inf, 0, 30, 60, np.inf],
                                 labels=["Early(<0min)", "Minor Delay(0-30min)", "Moderate Delay(30-60min)", "Severe Delay(60+min)"])
    df["Season"] = df["Month"].apply(lambda x: "Winter" if x in [12, 1, 2] else "Spring" if x in [3, 4, 5] else "Summer" if x in [6, 7, 8] else "Fall")
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7])
    df["IsStartOrEndOfMonth"] = ((df["DayofMonth"] <= 3) | (df["DayofMonth"] >= 28))
    df["DepHour"] = df["CRSDepTime"].astype(str).str.zfill(4).str[:2].astype(int)

    def classify_rush_hour(hour):
        if 6 <= hour < 10:
            return "Delay Valley"
        elif 12 <= hour < 14:
            return "Flat Midday"
        elif 16 <= hour < 20:
            return "Evening Delay Peak"
        else:
            return "Unstable Off-Peak"

    df["RushHour"] = df["DepHour"].apply(classify_rush_hour)
    df["Origin_Avg_DepDelay"] = df["Origin"].map(df.groupby("Origin")["DepDelay"].mean())
    df["Dest_Avg_ArrDelay"] = df["Dest"].map(df.groupby("Dest")["ArrDelay"].mean())
    df["Airline_AvgDelay"] = df["IATA_Code_Marketing_Airline"].map(df.groupby("IATA_Code_Marketing_Airline")["ArrDelay"].mean())
    
    # ========== Section 1: Target ========== #
    with st.expander("🎯 Target Variable (DelayCategory)", expanded=True):
        # 📌 Section header
        st.markdown("<h4>🎯 Target Delay Categories</h4>", unsafe_allow_html=True)

        # 📌 Define target category mapping (for display)
        df["DelayCategory_Short"] = df["DelayCategory"].map({
            "Early(<0min)": "Early",
            "Minor Delay(0-30min)": "Minor Delay",
            "Moderate Delay(30-60min)": "Moderate",
            "Severe Delay(60+min)": "Severe"
        })

        # 📌 Display category definitions
        st.markdown("#### 📋 Delay Category Definitions")
        st.table(pd.DataFrame({
            "Delay Category": ["Early", "Minor Delay", "Moderate", "Severe"],
            "Definition": [
                "Arrival earlier than scheduled (< 0 min)",
                "0 to 30 minutes delay",
                "30 to 60 minutes delay",
                "Over 60 minutes delay"
            ]
        }))

        # 📌 Add explanation
        st.markdown("""
        - The classification helps us group arrival delays into meaningful brackets for better interpretation.
        - As shown above, the majority of flights arrived early or with minor delays.
        """)

        # 📌 Plot bar chart of delay categories
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x="DelayCategory_Short", data=df, order=["Early", "Minor Delay", "Moderate", "Severe"], palette="Blues_r", ax=ax)
        ax.set_title("Flight Delay Categories Distribution", fontsize=14)
        ax.set_xlabel("Delay Category", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.tick_params(axis='x', labelsize=11)
        st.pyplot(fig)

    # ========== Section 2: Numerical Features ========== #
    with st.expander("📏 Numerical Features Summary", expanded=True):
        st.markdown("<h4>📏 Summary of Key Numeric Variables</h4>", unsafe_allow_html=True)
        
        # 📌 Select numeric columns
        num_cols = ["DepDelay", "ArrDelay", "ActualElapsedTime"]

        # 📌 Show descriptive statistics table
        st.markdown("**Descriptive Statistics:**")
        st.dataframe(df[num_cols].describe().T.style.format("{:.2f}"))

        # 📌 Add explanation
        st.markdown("""
        These are the core numerical variables influencing flight delay. 
        - **DepDelay** and **ArrDelay** indicate delays at departure and arrival respectively.
        - **ActualElapsedTime** captures the real airborne time.
        Use the histogram to observe skewness or outliers.
        """)

        # 📌 Histogram of numeric features
        st.markdown("**Histogram of Numeric Features:**")
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        for i, col in enumerate(num_cols):
            axs[i].hist(df[col], bins=50, edgecolor='black', alpha=0.7)
            axs[i].set_title(f"{col}", fontsize=12)
        st.pyplot(fig)

    # ========== Section 3: Airline ========== #
    with st.expander("🛩️ Average Arrival Delay by Airline", expanded=True):
        st.markdown("<h4>🛩️ Top Airlines by Arrival Delay</h4>", unsafe_allow_html=True)

        # 📌 Airline average delay calculation
        avg_delay = df.groupby("IATA_Code_Marketing_Airline")["ArrDelay"].mean().sort_values()
        
        # 📌 Add insights
        st.markdown("""
        This chart shows the average arrival delay by airline.
        - Airlines like **Delta Air Lines Inc. (DL)** and **Hawaiian Airlines INC (HA)** generally arrive earlier.
        - Airlines like **Allegiant Air LLC (G4)** and **JetBlue Airways Corporation (B6)** tend to have higher delays.
        """)

        # 📌 Plot airline delay bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(y=avg_delay.index, x=avg_delay.values, palette="Set2", ax=ax)
        ax.set_title("Average Arrival Delay by Airline (Sorted)", fontsize=14)
        ax.set_xlabel("Average Delay (minutes)", fontsize=12)
        ax.set_ylabel("Airline", fontsize=12)
        ax.tick_params(labelsize=10)
        st.pyplot(fig)

    # ========== Section 4: Airport ========== #
    with st.expander("🛫 Top Airports by Delay", expanded=True):
        st.markdown("<h4>🛫 Airports with Highest Delay</h4>", unsafe_allow_html=True)
        
        # 📌 Explanation
        st.markdown("""
        This section highlights airports with the worst average delays.
        - Departure delays are often influenced by local congestion or operational inefficiencies.
        - Arrival delays can relate to airspace restrictions, or traffic at destination.
        """)

        # 📌 Side-by-side airport charts
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top 10 Departure Delay Airports**")
            top_dep = df.groupby("Origin")["DepDelay"].mean().nlargest(10)
            fig, ax = plt.subplots()
            sns.barplot(x=top_dep.index, y=top_dep.values, palette="Reds_r", ax=ax)
            st.pyplot(fig)

        with col2:
            st.markdown("**Top 10 Arrival Delay Airports**")
            top_arr = df.groupby("Dest")["ArrDelay"].mean().nlargest(10)
            fig, ax = plt.subplots()
            sns.barplot(x=top_arr.index, y=top_arr.values, palette="Reds_r", ax=ax)
            st.pyplot(fig)

        # 📌 Density plot of engineered delay features
        st.markdown("**Histogram of Engineered Delay Features**")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.kdeplot(df["Origin_Avg_DepDelay"], label="Origin_Avg_DepDelay", fill=True)
        sns.kdeplot(df["Dest_Avg_ArrDelay"], label="Dest_Avg_ArrDelay", fill=True)
        ax.legend()
        st.pyplot(fig)

    # ========== Section 5: Time Features ========== #
    with st.expander("⏰ Time-related Delay Trends", expanded=True):
            st.markdown("<h4>⏰ Delay Patterns by Time Features</h4>", unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs([
                "By RushHour", "By Season", "Weekend vs MonthEnd"
            ])

            with tab1:
                st.markdown("""
                We defined custom rush hour segments:
                - **Evening Delay Peak** clearly shows highest average delays.
                - **Delay Valley** (early morning) has lowest delays.
                """)

                rush_order = ["Unstable Off-Peak", "Delay Valley", "Flat Midday", "Evening Delay Peak"]
                rush_avg = df.groupby("RushHour")["ArrDelay"].mean().reindex(rush_order)
                fig, ax = plt.subplots()
                sns.barplot(x=rush_avg.index, y=rush_avg.values, palette="coolwarm", ax=ax)
                ax.set_title("Average Delay by Rush Hour Category")
                st.pyplot(fig)

            with tab2:
                st.markdown("""
                Delays vary slightly by season, possibly due to weather and traffic patterns.
                Use this to compare performance across quarters.
                """)

                season_avg = df.groupby("Season")["ArrDelay"].mean().reindex(["Winter", "Spring", "Summer", "Fall"])
                fig, ax = plt.subplots()
                sns.barplot(x=season_avg.index, y=season_avg.values, palette="coolwarm", ax=ax)
                ax.set_title("Average Arrival Delay by Season")
                st.pyplot(fig)

            with tab3:
                st.markdown("""
                Weekend flights and flights near the start/end of the month appear to have slightly higher delays.
                This could reflect scheduling density or passenger surges.
                """)

                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                sns.barplot(x="IsWeekend", y="ArrDelay", data=df, palette="Set2", ax=ax[0])
                ax[0].set_title("Weekend vs Weekday", fontsize=13)
                sns.barplot(x="IsStartOrEndOfMonth", y="ArrDelay", data=df, palette="Set2", ax=ax[1])
                ax[1].set_title("Start/End vs Mid-Month", fontsize=13)
                st.pyplot(fig)


#  Part 3: LLM-based Flight Assistant
elif section == "🧠 LLM-based Flight Prediction Assistant":
    st.title("🧠 AirDelay Chat – Flight Delay Prediction")
    
    # 初始化 Session 状态
    if "conversation" not in st.session_state:
        st.session_state.conversation = [{"role": "assistant", "content":"Hi! Tell me about your flight and I’ll help predict delay."}]
    if "collected_slots" not in st.session_state:
        st.session_state.collected_slots = {}
    if "missing_fields" not in st.session_state:
        st.session_state.missing_fields = []

    # Session Reset 按钮
    if st.button("🔁 Reset Session"):
        st.session_state.conversation = [{"role": "assistant", "content": "Hi! Tell me about your flight and I’ll help predict delay."}]
        st.session_state.collected_slots = {}
        st.session_state.missing_fields = []
        st.rerun()

    printed_user_messages = set()

    for msg in st.session_state.conversation:
        if msg["role"] == "system":
            continue
        elif msg["role"] == "user":
            if msg["content"] in printed_user_messages:
                continue  # 跳过已打印的用户输入
            printed_user_messages.add(msg["content"])
        st.chat_message(msg["role"]).markdown(msg["content"])


    # 聊天输入框
    user_input = st.chat_input("✍️ Describe your flight...")

    if user_input:
        st.session_state.conversation.append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)

        # 调用 LLM 提取字段
        slots, conversation, missing_fields = extract_slots_loop(
            user_input,
            conversation=st.session_state.conversation,
            collected_slots=st.session_state.collected_slots
        )
        st.session_state.conversation = conversation
        st.session_state.collected_slots = slots
        st.session_state.missing_fields = missing_fields

        # ✅ 展示 LLM 的真实回复（而不是 summary）
        llm_reply = conversation[-1]["content"]
        st.chat_message("assistant").markdown(llm_reply)

        # ✅ 如果还有缺失字段，可额外提示（可选）
        if missing_fields:
#            followup_msg = f"🔍Still need more info to predict"
#            # 只有在最后一条 assistant 消息不是相同内容时，才添加
#            if not (st.session_state.conversation and st.session_state.conversation[-1]["role"] == "assistant" and st.session_state.conversation[-1]["content"] == followup_msg):
#                st.chat_message("assistant").markdown(followup_msg)
#                st.session_state.conversation.append({"role": "assistant", "content": followup_msg})
            pass
        else:
            # 开始做预测（用你的模型）
            with open(os.path.join(BASE_DIR_DATA, "lookup_dicts.pkl"), "rb") as f:
                lookup = pickle.load(f)
            with open(os.path.join(BASE_DIR_MODELS, "embedding_dims.json"), "r") as f:
                embedding_dims = json.load(f)
            label_encoders = load_label_encoders(os.path.join(BASE_DIR_MODELS, "label_encoders.json"))

            slots["Origin"] = find_iata_code(slots["Origin"])
            slots["Destination"] = find_iata_code(slots["Destination"])
            slots["Airline_AvgDepDelay"] = lookup["airline_avg_dep"].get(slots["Airline"], 10.0)
            slots["Airline_AvgArrDelay"] = lookup["airline_avg_arr"].get(slots["Airline"], 12.0)
            slots["Origin_Avg_DepDelay"] = lookup["origin_avg_dep"].get(slots["Origin"], 8.0)
            slots["Dest_Avg_ArrDelay"] = lookup["dest_avg_arr"].get(slots["Destination"], 9.0)

            if "Schedule Departure Time" in slots and "Schedule Arrival Time" in slots:
                slots["ActualElapsedTime"] = compute_elapsed_with_timezone(
                    slots["Schedule Departure Time"],
                    slots["Schedule Arrival Time"],
                    slots["Origin"],
                    slots["Destination"]
                )

            features = {
                "Airline": slots["Airline"],
                "Origin": slots["Origin"],
                "Dest": slots["Destination"],
                "DepDelay": calculate_departure_delay(slots["Actual Departure Time"], slots["Schedule Departure Time"]),
                "ActualElapsedTime": slots["ActualElapsedTime"],
                "IsWeekend": infer_weekend(slots["Day of Week"]),
                "IsStartOrEndOfMonth": infer_start_or_end_of_month(slots["Day of Month"]),
                "RushHour": get_rushhour(slots["Schedule Departure Time"]),
                "Season": infer_season(slots["Month"]),
                "Airline_AvgDepDelay": slots["Airline_AvgDepDelay"],
                "Airline_AvgArrDelay": slots["Airline_AvgArrDelay"],
                "Origin_Avg_DepDelay": slots["Origin_Avg_DepDelay"],
                "Dest_Avg_ArrDelay": slots["Dest_Avg_ArrDelay"],
            }

            st.chat_message("assistant").markdown("✅ All info collected! Making prediction...")
            try:
                predict_and_explain(slots, features, embedding_dims, label_encoders, display_fn=st.write)
            except Exception as e:
                st.chat_message("assistant").markdown(f"❌ Error during prediction: {e}")
        
