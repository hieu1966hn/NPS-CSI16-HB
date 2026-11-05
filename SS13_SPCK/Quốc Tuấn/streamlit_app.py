import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.compose._column_transformer as _ct
if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    _ct._RemainderColsList = _RemainderColsList

from google import genai


@st.cache_data
def load_student_data():
    if os.path.exists("StudentPerformanceFactors.csv"):
        return pd.read_csv("StudentPerformanceFactors.csv")
    return None

st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="💀",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "student_score_model.keras"
    )

    preprocessor = joblib.load(
        "preprocessor.pkl"
    )

    return model, preprocessor

model, preprocessor = load_model()

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

client = genai.Client(api_key=GEMINI_API_KEY)

with st.sidebar:
    menu = st.radio(
        "Chức năng chính:",
        ["🔮 Dự Đoán Điểm Số", "📊 Biểu Đồ Thống Kê (EDA)"]
    )

if menu == "🔮 Dự Đoán Điểm Số":
    st.title("Student Performance Predictor")

    st.write(
        """
        Enter your information below.
        
        The AI model will:
        - Predict your exam score
        - Analyze your strengths and weaknesses
        - Generate personal study recommendations using Gemini AI
        """
    )

col1, col2 = st.columns(2)

with col1:

    hours_studied = st.slider(
        "Hours Studied",
        0,
        40,
        15
    )

    attendance = st.slider(
        "Attendance (%)",
        0,
        100,
        85
    )

    parental_involvement = st.selectbox(
        "Parental Involvement",
        ["Low","Medium","High"]
    )

    access_to_resources = st.selectbox(
        "Access to Financial/Study Resources",
        ["Low","Medium","High"]
    )

    extracurricular = st.selectbox(
        "Extracurricular Activities",
        ["Yes","No"]
    )

    sleep_hours = st.slider(
        "Sleep Hours",
        0,
        12,
        7
    )

    previous_scores = st.slider(
        "Previous Scores",
        0,
        100,
        75
    )

    motivation_level = st.selectbox(
        "Motivation Level",
        ["Low","Medium","High"]
    )

    internet_access = st.selectbox(
        "Internet Access",
        ["Yes","No"]
    )

    tutoring_sessions = st.slider(
        "Tutoring Sessions",
        0,
        10,
        2
    )

with col2:

    family_income = st.selectbox(
        "Family Income",
        ["Low","Medium","High"]
    )

    teacher_quality = st.selectbox(
        "Teacher Quality",
        ["Low","Medium","High"]
    )

    school_type = st.selectbox(
        "School Type",
        ["Public","Private"]
    )

    peer_influence = st.selectbox(
        "Peer Influence",
        ["Positive","Neutral","Negative"]
    )

    physical_activity = st.slider(
        "Physical Activity Hours",
        0,
        20,
        5
    )

    learning_disabilities = st.selectbox(
        "Learning Disabilities",
        ["Yes","No"]
    )

    parental_education = st.selectbox(
        "Parental Education",
        ["High School","College","Postgraduate"]
    )

    distance_from_home = st.selectbox(
        "Distance From Home",
        ["Near","Moderate","Far"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

if st.button("Predict Exam Score"):

    student_df = pd.DataFrame({

        "Hours_Studied":[hours_studied],
        "Attendance":[attendance],
        "Parental_Involvement":[parental_involvement],
        "Access_to_Resources":[access_to_resources],
        "Extracurricular_Activities":[extracurricular],
        "Sleep_Hours":[sleep_hours],
        "Previous_Scores":[previous_scores],
        "Motivation_Level":[motivation_level],
        "Internet_Access":[internet_access],
        "Tutoring_Sessions":[tutoring_sessions],
        "Family_Income":[family_income],
        "Teacher_Quality":[teacher_quality],
        "School_Type":[school_type],
        "Peer_Influence":[peer_influence],
        "Physical_Activity":[physical_activity],
        "Learning_Disabilities":[learning_disabilities],
        "Parental_Education_Level":[parental_education],
        "Distance_from_Home":[distance_from_home],
        "Gender":[gender]

    })

    X = preprocessor.transform(student_df)

    prediction = model.predict(X)

    predicted_score = float(prediction[0][0])

    predicted_score = round(predicted_score, 2)

    st.success(
        f"Predicted Exam Score: {predicted_score}"
    )

    prompt = f"""
You are an educational advisor.

Student Profile:

Hours Studied: {hours_studied}
Attendance: {attendance}
Parental Involvement: {parental_involvement}
Access to Resources: {access_to_resources}
Extracurricular Activities: {extracurricular}
Sleep Hours: {sleep_hours}
Previous Scores: {previous_scores}
Motivation Level: {motivation_level}
Internet Access: {internet_access}
Tutoring Sessions: {tutoring_sessions}
Family Income: {family_income}
Teacher Quality: {teacher_quality}
School Type: {school_type}
Peer Influence: {peer_influence}
Physical Activity: {physical_activity}
Learning Disabilities: {learning_disabilities}
Parental Education Level: {parental_education}
Distance from Home: {distance_from_home}
Gender: {gender}

Predicted Exam Score:
{predicted_score}

Provide:

1. Performance Summary
2. Strengths
3. Weaknesses
4. Study Recommendations
5. Weekly Improvement Plan
6. Motivation Advice

Keep response student-friendly.
"""

    with st.spinner("Generating AI recommendations..."):

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        st.subheader("AI Educational Advisor")

        st.write(response.text)

else:
    st.title("📊 Student Performance Data Analysis (EDA)")
    df_viz = load_student_data()
    
    if df_viz is None:
        st.error("Không tìm thấy tệp StudentPerformanceFactors.csv để vẽ biểu đồ.")
    else:
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.write("**Biểu đồ 1: Ma trận tương quan giữa các biến (Correlation Heatmap)**")
            fig, ax = plt.subplots(figsize=(8, 6))
            numeric_df = df_viz.select_dtypes(include=[np.number])
            corr = numeric_df.corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
            plt.close(fig)

            st.write("**Biểu đồ 2: Phân bố điểm thi của học sinh (Exam Score Distribution)**")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(df_viz["Exam_Score"], bins=20, kde=True, ax=ax)
            ax.set_title("Phân bố điểm thi")
            ax.set_xlabel("Điểm thi")
            ax.set_ylabel("Số lượng học sinh")
            st.pyplot(fig)
            plt.close(fig)

            st.write("**Biểu đồ 3: Giờ học mỗi tuần vs Điểm thi (Hours Studied vs Exam Score)**")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                data=df_viz,
                x="Hours_Studied",
                y="Exam_Score",
                hue="Sleep_Hours",
                palette="viridis",
                s=60,
                alpha=0.8,
                ax=ax
            )
            ax.set_title("Giờ học, giấc ngủ và điểm thi")
            ax.set_xlabel("Số giờ học mỗi tuần")
            ax.set_ylabel("Điểm thi")
            st.pyplot(fig)
            plt.close(fig)

            st.write("**Biểu đồ 4: Ảnh hưởng kết hợp giữa số giờ học và tỷ lệ chuyên cần**")
            df_viz["Nhóm_Giờ_Học"] = pd.cut(
                df_viz["Hours_Studied"],
                bins=[0, 10, 20, 30, 40],
                labels=["0-10 giờ", "11-20 giờ", "21-30 giờ", "31-40 giờ"]
            )
            df_viz["Nhóm_Chuyên_Cần"] = pd.cut(
                df_viz["Attendance"],
                bins=[0, 70, 85, 100],
                labels=["Thấp", "Trung bình", "Cao"]
            )
            pivot = df_viz.pivot_table(
                values="Exam_Score",
                index="Nhóm_Giờ_Học",
                columns="Nhóm_Chuyên_Cần",
                aggfunc="mean",
                observed=False
            )
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                pivot,
                cmap="YlGnBu",
                annot=True,
                fmt=".1f",
                ax=ax
            )
            ax.set_title("Ảnh hưởng điểm số kết hợp giữa giờ học và chuyên cần")
            ax.set_xlabel("Mức chuyên cần")
            ax.set_ylabel("Số giờ học")
            st.pyplot(fig)
            plt.close(fig)

        with col_v2:
            st.write("**Biểu đồ 5: Thu nhập gia đình & Khả năng tiếp cận tài nguyên ảnh hưởng điểm số**")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(
                data=df_viz,
                x="Family_Income",
                y="Exam_Score",
                hue="Access_to_Resources",
                ax=ax
            )
            ax.set_title("Thu nhập gia đình và khả năng tiếp cận tài nguyên")
            ax.set_xlabel("Thu nhập gia đình")
            ax.set_ylabel("Điểm thi")
            st.pyplot(fig)
            plt.close(fig)

            st.write("**Biểu đồ 6: Mức độ quan tâm của phụ huynh (Parental Involvement)**")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(
                data=df_viz,
                x="Parental_Involvement",
                y="Exam_Score",
                ax=ax
            )
            ax.set_title("Mức độ quan tâm của phụ huynh")
            ax.set_xlabel("Mức độ quan tâm")
            ax.set_ylabel("Điểm thi")
            st.pyplot(fig)
            plt.close(fig)

            st.write("**Biểu đồ 7: Động lực học tập & Chất lượng giáo viên tác động điểm thi**")
            plot_df = df_viz.copy()
            plot_df["Motivation_Level"] = plot_df["Motivation_Level"].replace({
                "Low": "Thấp", "Medium": "Trung bình", "High": "Cao"
            })
            plot_df["Teacher_Quality"] = plot_df["Teacher_Quality"].replace({
                "Low": "Thấp", "Medium": "Trung bình", "High": "Cao"
            })
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.pointplot(
                data=plot_df,
                x="Motivation_Level",
                y="Exam_Score",
                hue="Teacher_Quality",
                order=["Thấp", "Trung bình", "Cao"],
                hue_order=["Thấp", "Trung bình", "Cao"],
                markers=["o", "s", "D"],
                linestyles="-",
                errorbar=None,
                ax=ax
            )
            ax.set_title("Ảnh hưởng của động lực học tập và chất lượng giáo viên")
            ax.set_xlabel("Mức độ động lực học tập")
            ax.set_ylabel("Điểm thi trung bình")
            st.pyplot(fig)
            plt.close(fig)

            st.write("**Biểu đồ 8: Nhiệt đồ chi tiết điểm thi trung bình theo giờ học & chuyên cần**")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                linewidths=0.5,
                cbar_kws={"label":"Điểm thi trung bình"},
                ax=ax
            )
            ax.set_title("Ảnh hưởng kết hợp giữa giờ học và chuyên cần (Chi tiết)")
            ax.set_xlabel("Mức độ chuyên cần")
            ax.set_ylabel("Số giờ học mỗi tuần")
            st.pyplot(fig)
            plt.close(fig)