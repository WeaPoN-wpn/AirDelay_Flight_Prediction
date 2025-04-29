import json
import re
import requests

ENDPOINT = "http://localhost:11434/api/chat" # 本地模型endpoint


def query_ollama_chat(conversation, model_name="llama3"):
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model_name,
                "messages": conversation,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except Exception as e:
        print("❌ Exception in query_ollama_chat:", str(e))
        return "[Local inference failed]"

#Create Prompt
system_prompt = """
You are a flight assistant. Your main job is to extract structured flight details from the user's message.

✈️ You must extract the following fields:
- Airline
- Origin
- Destination
- Month (1–12)
- Day of Month (1–31)
- Day of Week (1=Monday)
- Schedule Departure Time
- Schedule Arrival Time
- Actual Departure Time

🎯 INSTRUCTIONS:
1. DO NOT output anything in JSON format until all fields above are collected.
2. DO ask follow-up questions conversationally to get missing fields. Don't ask more than one question at a time.
3. Ask one question to gain one field at a time
4. DO NOT guess or invent any information.
5. If the user asks irrelevant questions, politely ask them to describe their flight instead.
6. Once ALL required fields are available, DON'T ask any more questions and just respond with ONLY a valid JSON object. No explanation, no extra text, no markdown.

✅ Final response format (only when complete and strictly follow this format):

{
  "Airline": "Delta Airlines",
  "Origin": "Los Angeles",
  "Destination": "Boston",
  "Month": 3,
  "Day of Month": 26,
  "Day of Week": 2,
  "Schedule Departure Time": "8:15am",
  "Schedule Arrival Time": "4:45pm",
  "Actual Departure Time": "8:35am"
}
"""

# create function to extract info

def extract_slots_loop(initial_input, conversation=None, collected_slots=None):
    required_fields = [
        "Airline", "Origin", "Destination",
        "Actual Departure Time", "Month", "Day of Month", "Day of Week",
        "Schedule Departure Time", "Schedule Arrival Time"
    ]
    
    # ✅ 初始化对话历史，确保 system prompt 存在
    if conversation is None:
        conversation = []
    if not any(m["role"] == "system" for m in conversation):
        conversation.insert(0, {"role": "system", "content": system_prompt})
    # ✅ 初始化已提取槽位信息
    if collected_slots is None:
        collected_slots = {}

    # ✅ 添加用户输入
    conversation.append({"role": "user", "content": initial_input})

    # ✅ 获取 LLM 回复
    reply = query_ollama_chat(conversation).strip()
    print("🧠 Raw LLM Reply:\n", reply)

    # ✅ 添加 assistant 回复
    conversation.append({"role": "assistant", "content": reply})
    print("🧠 Current Collected Slots:", collected_slots)
    print("🧠 Full Assistant Reply:", reply)
    # ✅ 尝试提取 JSON
    # 尝试提取 JSON 块（完整字段集）
    match = re.search(r'\{[\s\S]*?\}', reply)
    if match:
        try:
            slots = json.loads(match.group(0))
            for k, v in slots.items():
                if k in required_fields and v not in [None, "", 0]:
                    collected_slots[k] = v
        except json.JSONDecodeError:
            print("❌ JSON parse error – invalid JSON block")

    # 否则使用关键词匹配进行增量提取（fallback 模式）
    else:
        for field in required_fields:
            pattern = rf'{field}:\s*["\']?([\w\s:]+)["\']?'  # 简单正则，匹配字段
            found = re.search(pattern, reply, re.IGNORECASE)
            if found:
                value = found.group(1).strip()
                if value and field not in collected_slots:
                    collected_slots[field] = value

    # ✅ 检查缺失字段
    missing_fields = [f for f in required_fields if collected_slots.get(f) in [None, "", 0]]
    return collected_slots, conversation, missing_fields
