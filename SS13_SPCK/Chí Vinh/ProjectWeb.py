import streamlit as st
from pathlib import Path
import json
import os
from typing import Optional
import pandas as pd
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

def load_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    load_dotenv()
    return os.getenv("GEMINI_API_KEY")

@st.cache_resource
def create_client(api_key):
    if not api_key or genai is None:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None

# Sidebar navigation
st.sidebar.title("Menu")
page = st.sidebar.radio("Chọn trang", ["Xem Xét", "BotChat"])

if page == "Xem Xét":
    st.title("Bảng Thất Nghiệp Toàn Cầu Từ Năm 2014 đến năm 2024")
    st.markdown("")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Giới tính", "Trước và Sau Covid", "Toàn cầu"])
    
    with tab1:
        st.write("Biểu đồ về sự thất nghiệp giữa nam và nữ từ năm 2014 đến năm 2024")
        img_path1 = Path(__file__).parent / "SSProjectPics" / "output.png"
        if img_path1.exists():
            st.image(str(img_path1), caption="Output", width=700)
        else:
            st.error(f"Image not found: {img_path1}")
    
    with tab2:
        st.write("Biểu đồ về sự thay đổi về thất nghiệp trước và sau khi dịch bệnh Covid-19")
        img_path2 = Path(__file__).parent / "SSProjectPics" / "outputbox.png"
        if img_path2.exists():
            st.image(str(img_path2), width=700)
        else:
            st.error(f"Image not found: {img_path2}")
        
        img_path5 = Path(__file__).parent/ "SSProjectPics"/ "outputViolin.png"
        if img_path5.exists():
            st.image(str(img_path5), caption="Output", width=700)
        else:
            st.error(f"Image not found: {img_path5}")
        
    
    with tab3:
        st.write("Biểu đồ về 3 quốc gia có tổng thất nghiệp nhiều nhất")
        img_path3 = Path(__file__).parent / "SSProjectPics" / "outputCircle.png"
        if img_path3.exists():
            st.image(str(img_path3), caption="Output", width=700)
        else:
            st.error(f"Image not found: {img_path3}")

        img_path4 = Path(__file__).parent/ "SSProjectPics"/ "outputMap"
        if img_path4.exists():
            st.image(str(img_path4), caption="Output", width=700)
        else:
            st.error(f"Image not found: {img_path4}")

        img_path6 = Path(__file__).parent/ "SSProjectPics"/ "outputMultipleBars.png"
        if img_path6.exists():
            st.image(str(img_path6), caption="Output", width=700)
        else:
            st.error(f"Image not found: {img_path6}")

        img_path7 = Path(__file__).parent/ "SSProjectPics"/ "outputLineAggr.png"
        if img_path7.exists():
            st.image(str(img_path7), caption="Output", width=700)
        else:
            st.error(f"Image not found: {img_path7}")

        img_path8 = Path(__file__).parent/ "SSProjectPics"/ "outputHistogram.png"
        if img_path8.exists():
            st.image(str(img_path8), caption="Output", width=700)
        else:
            st.error(f"Image not found: {img_path8}")

elif page == "BotChat":
    st.title("🤖 Trợ Lý Phân Tích Thất Nghiệp AI")
    
    # Load config and key
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {
            "initial_bot_message": "Chào bạn, tôi là BotChat. Tôi có thể giúp bạn tìm hiểu thông tin về sự thất nghiệp toàn cầu.",
            "out_of_scope_message": "Tôi đang không hỗ trợ chức năng này. Xin liên hệ qua hotline để được trợ giúp."
        }
        
    api_key = load_api_key()
    client = create_client(api_key)
    
    # Show status
    with st.sidebar:
        st.write("---")
        st.caption("🤖 Trạng thái kết nối:")
        if client:
            st.success("🟢 Gemini Connected")
        else:
            st.error("🔴 Gemini API key not found. Add GEMINI_API_KEY to secrets or .env.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": config.get("initial_bot_message")}
        ]
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input("Nhập yêu cầu phân tích thất nghiệp..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        if client is None:
            reply = "❌ **Lỗi:** API Key chưa được cấu hình. Vui lòng thêm GEMINI_API_KEY vào secrets hoặc file `.env`."
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            try:
                system_instruction = (
                    "Bạn là UnemploymentBot, một trợ lý AI chuyên gia phân tích dữ liệu thất nghiệp toàn cầu từ năm 2014 đến 2024.\n"
                    "Nhiệm vụ của bạn là giải đáp các thắc mắc về tỉ lệ thất nghiệp giữa nam và nữ, sự ảnh hưởng trước và sau Covid, hoặc các quốc gia có tỉ lệ thất nghiệp nhiều nhất.\n"
                    "Hãy trả lời thân thiện, khoa học, khách quan bằng Tiếng Việt dựa trên dữ liệu. Không được bịa số liệu. "
                    "Nếu câu hỏi nằm ngoài phạm vi phân tích kinh tế - xã hội hoặc thất nghiệp, vui lòng trả lời đúng câu sau: "
                    f"\"{config.get('out_of_scope_message')}\""
                )
                
                # Format history for Gemini SDK
                contents = []
                for msg in st.session_state.messages[-8:]:
                    role_val = "user" if msg["role"] == "user" else "model"
                    contents.append(types.Content(
                        role=role_val,
                        parts=[types.Part.from_text(text=msg["content"])]
                    ))
                
                with st.spinner("Đang kết nối dữ liệu và phân tích..."):
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.3
                        )
                    )
                    reply = response.text or "Tôi không thể xử lý yêu cầu lúc này."
                    
                with st.chat_message("assistant"):
                    st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
                st.info("Kiểm tra lại cấu hình hoặc kết nối mạng.")