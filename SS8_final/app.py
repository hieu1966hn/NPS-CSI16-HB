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
def load_config() -> dict:
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_menu() -> pd.DataFrame:
    return pd.read_csv("menu.csv", index_col=0)


def load_api_key() -> Optional[str]:
    load_dotenv()
    return (
        os.getenv("GEMINI_API_KEY")
        or get_streamlit_secret("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or get_streamlit_secret("GOOGLE_API_KEY")
    )


@st.cache_resource
def create_client(api_key: Optional[str]):
    if not api_key or genai is None:
        return None
    return genai.Client(api_key=api_key)


def build_menu_context(menu_df: pd.DataFrame) -> str:
    def clean_text(value) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip()

    def add_detail(item_text: str, label: str, value) -> str:
        value = clean_text(value)
        if not value:
            return item_text

        suffix = "" if value.endswith((".", "!", "?")) else "."
        return f"{item_text} {label}: {value}{suffix}"

    menu_lines = []
    for _, row in menu_df.iterrows():
        item_text = f"- {clean_text(row['name'])}: {clean_text(row['description'])}"
        item_text = add_detail(item_text, "Nguyên liệu", row.get("ingredients", ""))
        item_text = add_detail(item_text, "Ghi chú", row.get("notes", ""))
        menu_lines.append(item_text)
    return "\n".join(menu_lines)


def build_system_instruction(config: dict) -> str:
    functions = ", ".join(config.get("functions", []))
    restaurant_name = config.get("restaurant_name", "Viet Cuisine")
    restaurant_address = config.get("restaurant_address", "329 Scottmouth, Georgia, USA")
    out_of_scope_message = config.get("out_of_scope_message", "Xin liên hệ nhân viên nhà hàng để được trợ giúp.")
    return "\n".join([
        f"Bạn tên là PhoBot, một trợ lý AI hỗ trợ khách hàng của nhà hàng {restaurant_name}.",
        f"Địa chỉ nhà hàng: {restaurant_address}.",
        f"Các chức năng được hỗ trợ: {functions}.",
        "",
        "Nguyên tắc trả lời:",
        "1. Trả lời ngắn gọn, thân thiện, lịch sự và dễ hiểu.",
        "2. Chỉ trả lời các câu hỏi liên quan đến nhà hàng, menu hoặc món ăn trong dữ liệu được cung cấp.",
        f"3. Nếu câu hỏi nằm ngoài phạm vi hỗ trợ, trả lời đúng câu sau: \"{out_of_scope_message}\"",
        "4. Không bịa thông tin nếu dữ liệu menu không có câu trả lời.",
    ])


def build_history_text(messages: list[dict], max_messages: int = 6) -> str:
    recent_messages = messages[-max_messages:]
    lines = []
    for message in recent_messages:
        role = "Khách hàng" if message["role"] == "user" else "PhoBot"
        lines.append(f"{role}: {message['content']}")
    return "\n".join(lines)


def mock_response(prompt: str, menu_df: pd.DataFrame, config: dict) -> str:
    prompt_lower = prompt.lower()
    if any(keyword in prompt_lower for keyword in ["menu", "món", "mon", "ăn", "an", "food"]):
        return "\n\n".join(
            f"**{row['name']}**: {row['description']}" for _, row in menu_df.iterrows()
        )
    if any(keyword in prompt_lower for keyword in ["địa chỉ", "dia chi", "ở đâu", "address"]):
        return f"Nhà hàng {config.get('restaurant_name')} nằm tại {config.get('restaurant_address')}."
    if any(keyword in prompt_lower for keyword in ["xin chào", "hello", "hi", "chào"]):
        return "Chào bạn! Tôi là PhoBot. Bạn có thể hỏi tôi về nhà hàng hoặc các món trong menu nhé."
    return config.get("out_of_scope_message")


def ask_bot(prompt: str, messages: list[dict], client, menu_df: pd.DataFrame, config: dict, use_mock: bool) -> str:
    if use_mock or client is None:
        return mock_response(prompt, menu_df, config)

    system_instruction = build_system_instruction(config)
    menu_context = build_menu_context(menu_df)
    history_text = build_history_text(messages)
    user_content = (
        "Dữ liệu menu nhà hàng:\n" + menu_context +
        "\n\nLịch sử trò chuyện gần đây:\n" + history_text +
        "\n\nCâu hỏi mới của khách hàng:\n" + prompt
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4,
            ),
        )
        return response.text or "Xin lỗi, tôi chưa tạo được câu trả lời phù hợp."
    except Exception as exc:
        return (
            "Xin lỗi, hiện tại tôi chưa kết nối được với Gemini API. "
            "Bạn có thể kiểm tra lại API key, kết nối mạng hoặc dùng chế độ Mock để tiếp tục demo.\n\n"
            f"Chi tiết lỗi: `{exc}`"
        )


def restaurant_chatbot():
    config = load_config()
    menu_df = load_menu()
    api_key = load_api_key()
    client = create_client(api_key)

    st.set_page_config(page_title="PhoBot - Restaurant Assistant", page_icon="🍜")
    st.title("🍜 PhoBot - Restaurant Assistant")
    st.write("Trợ lý ảo hỗ trợ khách hàng tìm hiểu nhà hàng Viet Cuisine và menu món Việt.")

    with st.sidebar:
        st.header("Cấu hình")
        st.caption("Dùng cho buổi 8: Streamlit Chat UI + Gemini API")
        has_sdk = genai is not None
        st.write("Google GenAI SDK:", "✅ Đã sẵn sàng" if has_sdk else "⚠️ Chưa cài `google-genai`")
        st.write("API key:", "✅ Đã tìm thấy" if api_key else "⚠️ Chưa có")
        use_mock = st.toggle(
            "Dùng Mock Response",
            value=not bool(api_key and has_sdk),
            help="Bật chế độ này nếu học viên chưa tạo được API key hoặc lớp cần tránh lỗi API.",
        )
        st.markdown("---")
        st.markdown("**Gợi ý câu hỏi:**")
        st.markdown("- Nhà hàng ở đâu?")
        st.markdown("- Menu có những món gì?")
        st.markdown("- Phở Việt Nam gồm nguyên liệu gì?")

    if "conversation_log" not in st.session_state:
        st.session_state.conversation_log = [
            {"role": "assistant", "content": config.get("initial_bot_message", "Xin chào! Bạn cần hỗ trợ gì?")}
        ]

    for message in st.session_state.conversation_log:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Nhập yêu cầu của bạn tại đây...")
    if prompt:
        st.session_state.conversation_log.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        bot_reply = ask_bot(
            prompt=prompt,
            messages=st.session_state.conversation_log,
            client=client,
            menu_df=menu_df,
            config=config,
            use_mock=use_mock,
        )

        st.session_state.conversation_log.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.write(bot_reply)

    with st.expander("Checklist bảo mật API key"):
        st.markdown(
            """
- Không dán API key trực tiếp vào `app.py`.
- Local: lưu key trong `.env` với tên `GEMINI_API_KEY`.
- Deploy Streamlit Cloud: lưu key trong **Secrets**.
- Không commit file `.env` lên GitHub.
- Nếu thiếu key, app vẫn có thể chạy bằng Mock Response để demo giao diện.
"""
        )


if __name__ == "__main__":
    restaurant_chatbot()
