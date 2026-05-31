import json
import os
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

MODEL_NAME = "gemini-2.5-flash"

def get_streamlit_secret(name: str) -> Optional[str]:
    try:
        return st.secrets.get(name)
    except Exception:
        return None

@st.cache_data
def load_config()-> dict:
    with open("config.json", 'r', encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_menu()-> pd.DataFrame:
    return pd.read_csv("menu.csv", index_col=0)

def load_api_key()->Optional[str]:
    load_dotenv()
    return (
        os.getenv("GEMINI_API_KEY")
        # or get_streamlit_secret("GEMINI_API_KEY")
        # or os.getenv("GOOGLE_API_KEY")
        # or get_streamlit_secret("GOOGLE_API_KEY")
    )
    
    
@st.cache_resource
def create_client(api_key: Optional[str]):
    if not api_key or genai is None: 
        return None
    return genai.Client(api_key=api_key)

def build_menu_context(menu_df: pd.DataFrame) -> str:
    def clean_text(value)->str:
        if pd.isna(value):
            return ""
        return str(value).strip()
    
    def add_detail(item_text:str, label:str, value) -> str: 
        value = clean_text()
        if not value:
            return item_text
        
        suffix = "" if value.endswith((".", "!", "?")) else "."
        return f"{item_text} {label}: {value}{suffix}"
    
    menu_lines = []




def ask_bot(prompt:str, messages: list[dict], client, menu_df: pd.DataFrame, config: dict, use_mock: bool) -> str:
    if use_mock or client is None:
        return mock_response(prompt, menu_df, config) # chưa có hàm này, bổ sun sau
    
    system_instruction = buld_system_intruction(config)
    menu_context = build_menu_context(menu_df)
    history_text = build_history_text(messages)
    user_content= (
        "Dữ liệu nhà hàng:\n" + menu_context + 
        "\n\nLich sử trò chuyện gần đây:\n" + history_text + 
        "\n\nCâu hỏi mới của khách hàng:\n" + prompt
    )
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4
            ),
        )
        return response.text or "Xin lỗi, tôi chưa tạo được câu trả lời phù hợp."
    except Exception as exc: 
        return(
            "Xin lỗi, hiện tại tôi chưa kết nối được với GEMINI API"
            "Bạn có thể kiểm tra lại API key, kết nối mạng hoặc dùng chế độ Mock để tiếp tục"
            f"Chi tiết lỗi: `{exc}`"
        )