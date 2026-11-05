import os
import json
import pickle

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from google import genai
from google.genai import types

# ===========================================
# CONFIG
# ===========================================

MODEL_NAME = "gemini-2.5-flash"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ===========================================
# LOAD FILE
# ===========================================

@st.cache_data
def load_config():

    path = os.path.join(BASE_DIR, "config.json")

    with open(path, "r", encoding="utf-8") as f:

        return json.load(f)


@st.cache_data
def load_knowledge():

    path = os.path.join(
        BASE_DIR,
        "admission_info.csv"
    )

    return pd.read_csv(path)


@st.cache_data
def load_admission_data():

    path = os.path.join(
        BASE_DIR,
        "Admission_Predict.csv"
    )

    if os.path.exists(path):
        return pd.read_csv(path)

    return None


# ===========================================
# GEMINI
# ===========================================

def load_api_key():

    try:

        return st.secrets["GEMINI_API_KEY"]

    except:

        return None


@st.cache_resource
def create_client(api_key):

    if api_key is None:

        return None

    try:

        return genai.Client(api_key=api_key)

    except:

        return None


# ===========================================
# MODEL
# ===========================================

@st.cache_resource
def load_model():

    path = os.path.join(
        BASE_DIR,
        "model.pkl"
    )

    if not os.path.exists(path):

        return None

    try:

        with open(path, "rb") as f:

            model = pickle.load(f)

        return model

    except Exception as e:

        st.error(f"Không load được model\n\n{e}")

        return None


# ===========================================
# CHATBOT
# ===========================================

def build_context(df):

    text = ""

    for _, row in df.iterrows():

        text += f"""

Chủ đề:

{row['topic']}

Thông tin:

{row['description']}

"""

    return text


def build_system_instruction(config):

    return f"""
Bạn là AdmissionBot.

Bạn là trợ lý AI tư vấn tuyển sinh.

Bạn chỉ được trả lời các câu hỏi liên quan đến:

- GRE
- TOEFL
- IELTS
- SOP
- LOR
- Research
- CGPA
- GPA
- Chance of Admit
- Đại học
- Du học
- Hồ sơ

Nguyên tắc:

1. Trả lời bằng tiếng Việt.

2. Không bịa thông tin.

3. Nếu ngoài phạm vi thì trả lời đúng:

"{config['out_of_scope_message']}"

4. Ưu tiên sử dụng dữ liệu đã được cung cấp.
"""


def ask_bot(
    prompt,
    history,
    client,
    knowledge_df,
    config
):

    if client is None:

        return "❌ Gemini API chưa kết nối."

    history_text = ""

    for msg in history[-8:]:

        role = "User"

        if msg["role"] == "assistant":

            role = "Assistant"

        history_text += f"{role}: {msg['content']}\n"

    context = build_context(
        knowledge_df
    )

    user_content = f"""
DỮ LIỆU

{context}

=====================

LỊCH SỬ CHAT

{history_text}

=====================

CÂU HỎI

{prompt}
"""

    try:

        response = client.models.generate_content(

            model=MODEL_NAME,

            contents=user_content,

            config=types.GenerateContentConfig(

                system_instruction=build_system_instruction(config),

                temperature=0.3

            )

        )

        return response.text

    except Exception as e:

        return str(e)

# ===========================================
# MAIN
# ===========================================

def main():

    st.set_page_config(
        page_title="Admission AI",
        page_icon="🎓",
        layout="wide"
    )

    config = load_config()

    knowledge_df = load_knowledge()

    api_key = load_api_key()

    client = create_client(api_key)

    model = load_model()

    # -------------------------

    st.title("🎓 Admission AI")

    st.caption(
        "Chatbot tư vấn tuyển sinh và dự đoán khả năng đỗ đại học"
    )

    with st.sidebar:

        st.header("Trạng thái hệ thống")

        if client:

            st.success("✅ Gemini Connected")

        else:

            st.error("❌ Gemini Disconnected")

        if model:

            st.success("✅ Model Loaded")

        else:

            st.warning("⚠ Model chưa sẵn sàng")

    tab_chat, tab_predict, tab_viz = st.tabs(
        [
            "🤖 Chatbot",
            "🔮 Prediction",
            "📊 Data Visualization"
        ]
    )

    # ======================================
    # CHATBOT
    # ======================================

    with tab_chat:

        if "messages" not in st.session_state:

            st.session_state.messages = [

                {
                    "role": "assistant",
                    "content":
                    config["initial_bot_message"]
                }

            ]

        for message in st.session_state.messages:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )

        prompt = st.chat_input(
            "Nhập câu hỏi..."
        )

        if prompt:

            st.session_state.messages.append(

                {
                    "role": "user",
                    "content": prompt
                }

            )

            with st.chat_message("user"):

                st.markdown(prompt)

            with st.spinner("Đang trả lời..."):

                answer = ask_bot(

                    prompt=prompt,

                    history=st.session_state.messages,

                    client=client,

                    knowledge_df=knowledge_df,

                    config=config

                )

            with st.chat_message("assistant"):

                st.markdown(answer)

            st.session_state.messages.append(

                {
                    "role": "assistant",
                    "content": answer
                }

            )

    # ======================================
    # PREDICTION
    # ======================================

    with tab_predict:

        st.subheader(
            "Chance of Admit Prediction"
        )

        if model is None:

            st.error(
                "Không load được model."
            )

            st.stop()

        col1, col2 = st.columns(2)

        with col1:

            gre = st.number_input(
                "GRE Score",
                260,
                340,
                320
            )

            toefl = st.number_input(
                "TOEFL Score",
                0,
                120,
                105
            )

            university_rating = st.slider(
                "University Rating",
                1,
                5,
                3
            )

            sop = st.slider(
                "SOP",
                1.0,
                5.0,
                4.0,
                step=0.5
            )

        with col2:

            lor = st.slider(
                "LOR",
                1.0,
                5.0,
                4.0,
                step=0.5
            )

            cgpa = st.number_input(
                "CGPA",
                0.0,
                10.0,
                8.5
            )

            research = st.selectbox(
                "Research",
                [0, 1]
            )

        if st.button(
            "Predict Chance of Admit",
            use_container_width=True
        ):

            input_data = pd.DataFrame(
                {
                    "GRE Score": [gre],
                    "TOEFL Score": [toefl],
                    "University Rating": [university_rating],
                    "SOP": [sop],
                    "LOR": [lor],
                    "CGPA": [cgpa],
                    "Research": [research]
                }
            )

            try:

                prediction = model.predict(
                    input_data
                )

                chance = float(prediction[0]) * 100

                st.metric(
                    "Chance of Admit",
                    f"{chance:.2f}%"
                )

                st.progress(
                    min(int(chance), 100)
                )

                if chance >= 85:

                    st.success(
                        "🎉 Khả năng đỗ rất cao."
                    )

                elif chance >= 70:

                    st.info(
                        "👍 Hồ sơ khá cạnh tranh."
                    )

                elif chance >= 50:

                    st.warning(
                        "⚠ Nên cải thiện hồ sơ."
                    )

                else:

                    st.error(
                        "❌ Khả năng đỗ thấp."
                    )

                st.subheader(
                    "Đánh giá hồ sơ"
                )

                if gre < 310:
                    st.write("• GRE còn thấp.")

                if toefl < 100:
                    st.write("• TOEFL nên trên 100.")

                if cgpa < 8.5:
                    st.write("• CGPA nên tăng.")

                if research == 0:
                    st.write("• Có Research sẽ tăng khả năng đỗ.")

                if sop < 4:
                    st.write("• SOP nên đầu tư hơn.")

                if lor < 4:
                    st.write("• LOR có thể cải thiện.")

            except Exception as e:

                st.error(e)

    # ======================================
    # DATA VISUALIZATION
    # ======================================

    with tab_viz:

        st.subheader("📊 Data Visualization & Exploratory Data Analysis")

        df_viz = load_admission_data()

        if df_viz is None:

            st.error("Không tìm thấy tệp Admission_Predict.csv để hiển thị biểu đồ.")

        else:

            col_v1, col_v2 = st.columns(2)

            with col_v1:

                st.write("**Biểu đồ 1: Phân bố điểm GRE (GRE Score Distribution)**")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.histplot(df_viz["GRE Score"], kde=True, color="skyblue", ax=ax)
                st.pyplot(fig)
                plt.close(fig)

                st.write("**Biểu đồ 2: Điểm số CGPA vs Cơ hội trúng tuyển (Chance of Admit)**")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(x="CGPA", y="Chance of Admit", data=df_viz, hue="Research", palette="coolwarm", ax=ax)
                st.pyplot(fig)
                plt.close(fig)

                st.write("**Biểu đồ 3: Điểm số GRE vs Cơ hội trúng tuyển (Chance of Admit)**")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(x="GRE Score", y="Chance of Admit", data=df_viz, color="green", ax=ax)
                st.pyplot(fig)
                plt.close(fig)

                st.write("**Biểu đồ 4: Điểm số TOEFL vs Cơ hội trúng tuyển (Chance of Admit)**")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(x="TOEFL Score", y="Chance of Admit", data=df_viz, color="orange", ax=ax)
                st.pyplot(fig)
                plt.close(fig)

            with col_v2:

                st.write("**Biểu đồ 5: Kinh nghiệm Nghiên cứu vs Cơ hội trúng tuyển**")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(x="Research", y="Chance of Admit", data=df_viz, palette="Set2", ax=ax)
                st.pyplot(fig)
                plt.close(fig)

                st.write("**Biểu đồ 6: Ma trận tương quan giữa các chỉ số (Correlation Heatmap)**")
                fig, ax = plt.subplots(figsize=(8, 5))
                corr = df_viz.corr(numeric_only=True)
                sns.heatmap(corr, annot=True, cmap="YlOrRd", ax=ax)
                st.pyplot(fig)
                plt.close(fig)

                st.write("**Biểu đồ 7: Phân bố số lượng đơn ứng tuyển theo xếp hạng Đại học**")
                fig, ax = plt.subplots(figsize=(8, 5))
                df_viz["University Rating"].value_counts().sort_index().plot(kind="bar", color="coral", ax=ax)
                plt.title("Applications by University Rating")
                plt.ylabel("Quantity")
                plt.xticks(rotation=0)
                st.pyplot(fig)
                plt.close(fig)

                st.write("**Biểu đồ 8: Phân bố điểm CGPA theo kinh nghiệm Nghiên cứu**")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(x="Research", y="CGPA", data=df_viz, palette="Pastel1", ax=ax)
                st.pyplot(fig)
                plt.close(fig)


# ===========================================
# RUN APP
# ===========================================

if __name__ == "__main__":

    main()