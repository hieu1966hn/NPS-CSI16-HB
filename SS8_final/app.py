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