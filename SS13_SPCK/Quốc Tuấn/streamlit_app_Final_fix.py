import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    page_title="Dự đoán Điểm Thi Học Sinh",
    page_icon="🎓",
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
    menu = st.radio("Chức năng chính:", ["🔮 Dự Đoán Điểm Số", "📊 Biểu Đồ Thống Kê (EDA)"])
if menu == "🔮 Dự Đoán Điểm Số":
    st.title("Hệ thống Dự đoán Kết quả Học tập")

    st.write(
        """
        Vui lòng nhập thông tin của học sinh bên dưới.

        Hệ thống AI sẽ:
        - Dự đoán điểm thi cuối kỳ
        - Phân tích điểm mạnh và điểm cần cải thiện
        - Đưa ra lộ trình và lời khuyên học tập cá nhân hóa bằng Gemini AI
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        hours_studied = st.slider(
            "Số giờ học mỗi tuần",
            0,
            40,
            15
        )

        attendance = st.slider(
            "Tỷ lệ chuyên cần (%)",
            0,
            100,
            85
        )

        parental_involvement = st.selectbox(
            "Mức độ quan tâm của phụ huynh",
            options=["Low", "Medium", "High"],
            format_func=lambda x: {
                "Low": "Thấp",
                "Medium": "Trung bình",
                "High": "Cao"
            }[x]
        )

        access_to_resources = st.selectbox(
            "Mức độ tiếp cận tài nguyên học tập",
            options=["Low", "Medium", "High"],
            format_func=lambda x: {
                "Low": "Thấp",
                "Medium": "Trung bình",
                "High": "Cao"
            }[x]
        )

        extracurricular = st.selectbox(
            "Tham gia hoạt động ngoại khóa",
            options=["Yes", "No"],
            format_func=lambda x: {
                "Yes": "Có",
                "No": "Không"
            }[x]
        )

        sleep_hours = st.slider(
            "Số giờ ngủ mỗi ngày",
            0,
            12,
            7
        )

        previous_scores = st.slider(
            "Điểm trung bình các bài kiểm tra trước",
            0,
            100,
            75
        )

        motivation_level = st.selectbox(
            "Mức độ động lực học tập",
            options=["Low", "Medium", "High"],
            format_func=lambda x: {
                "Low": "Thấp",
                "Medium": "Trung bình",
                "High": "Cao"
            }[x]
        )

        internet_access = st.selectbox(
            "Có kết nối Internet và kĩ năng sử dụng Internet",
            options=["Yes", "No"],
            format_func=lambda x: {
                "Yes": "Có",
                "No": "Không"
            }[x]
        )

        tutoring_sessions = st.slider(
            "Số buổi học thêm mỗi tháng",
            0,
            10,
            2
        )

    with col2:

        family_income = st.selectbox(
            "Mức thu nhập gia đình",
            options=["Low", "Medium", "High"],
            format_func=lambda x: {
                "Low": "Thấp",
                "Medium": "Trung bình",
                "High": "Cao"
            }[x]
        )

        teacher_quality = st.selectbox(
            "Chất lượng giáo viên",
            options=["Low", "Medium", "High"],
            format_func=lambda x: {
                "Low": "Thấp",
                "Medium": "Trung bình",
                "High": "Cao"
            }[x]
        )

        school_type = st.selectbox(
            "Loại hình trường học",
            options=["Public", "Private"],
            format_func=lambda x: {
                "Public": "Công lập",
                "Private": "Tư thục"
            }[x]
        )

        peer_influence = st.selectbox(
            "Ảnh hưởng từ bạn bè",
            options=["Positive", "Neutral", "Negative"],
            format_func=lambda x: {
                "Positive": "Tích cực",
                "Neutral": "Trung lập",
                "Negative": "Tiêu cực"
            }[x]
        )

        physical_activity = st.slider(
            "Số giờ vận động mỗi tuần",
            0,
            20,
            5
        )

        learning_disabilities = st.selectbox(
            "Có khó khăn trong học tập",
            options=["Yes", "No"],
            format_func=lambda x: {
                "Yes": "Có",
                "No": "Không"
            }[x]
        )

        parental_education = st.selectbox(
            "Trình độ học vấn của phụ huynh",
            options=["High School", "College", "Postgraduate"],
            format_func=lambda x: {
                "High School": "THPT",
                "College": "Cao đẳng/Đại học",
                "Postgraduate": "Sau đại học"
            }[x]
        )

        distance_from_home = st.selectbox(
            "Khoảng cách từ nhà đến trường",
            options=["Near", "Moderate", "Far"],
            format_func=lambda x: {
                "Near": "Gần",
                "Moderate": "Trung bình",
                "Far": "Xa"
            }[x]
        )

        gender = st.selectbox(
            "Giới tính",
            options=["Male", "Female"],
            format_func=lambda x: {
                "Male": "Nam",
                "Female": "Nữ"
            }[x]
        )

    level_vi = {
        "Low": "Thấp",
        "Medium": "Trung bình",
        "High": "Cao"
    }

    yes_no_vi = {
        "Yes": "Có",
        "No": "Không"
    }

    school_vi = {
        "Public": "Công lập",
        "Private": "Tư thục"
    }

    peer_vi = {
        "Positive": "Tích cực",
        "Neutral": "Trung lập",
        "Negative": "Tiêu cực"
    }

    education_vi = {
        "High School": "THPT",
        "College": "Cao đẳng/Đại học",
        "Postgraduate": "Sau đại học"
    }

    distance_vi = {
        "Near": "Gần",
        "Moderate": "Trung bình",
        "Far": "Xa"
    }

    gender_vi = {
        "Male": "Nam",
        "Female": "Nữ"
    }

    if st.button("Dự đoán điểm thi"):
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
        predicted_score = max(0, min(10, predicted_score))


        st.success(
            f"Điểm dự đoán: {predicted_score}"
        )

        prompt = f"""
    Bạn là một cố vấn học tập sử dụng AI.
    Thông tin học sinh:
    • Số giờ học mỗi tuần: {hours_studied} giờ
    • Tỷ lệ chuyên cần: {attendance}%
    • Mức độ quan tâm của phụ huynh: {level_vi[parental_involvement]}
    • Mức độ tiếp cận tài nguyên học tập: {level_vi[access_to_resources]}
    • Tham gia hoạt động ngoại khóa: {yes_no_vi[extracurricular]}
    • Số giờ ngủ trung bình mỗi ngày: {sleep_hours} giờ
    • Điểm trung bình các bài kiểm tra trước: {previous_scores}/100
    • Mức độ động lực học tập: {level_vi[motivation_level]}
    • Có kết nối Internet: {yes_no_vi[internet_access]}
    • Số buổi học thêm mỗi tháng: {tutoring_sessions}
    • Thu nhập gia đình: {level_vi[family_income]}
    • Chất lượng giáo viên: {level_vi[teacher_quality]}
    • Loại hình trường học: {school_vi[school_type]}
    • Ảnh hưởng từ bạn bè: {peer_vi[peer_influence]}
    • Số giờ vận động mỗi tuần: {physical_activity} giờ
    • Có khó khăn trong học tập: {yes_no_vi[learning_disabilities]}
    • Trình độ học vấn của phụ huynh: {education_vi[parental_education]}
    • Khoảng cách từ nhà đến trường: {distance_vi[distance_from_home]}
    • Giới tính: {gender_vi[gender]}

    Điểm dự đoán:
    {predicted_score}

    Hãy đưa ra:
    1. Đánh giá tổng quan về kết quả học tập
    2. Điểm mạnh của học sinh
    3. Những điểm cần cải thiện
    4. Các lời khuyên học tập phù hợp
    5. Kế hoạch học tập trong 7 ngày

    Viết bằng tiếng Việt, rõ ràng, thân thiện và dễ hiểu.
    """

        with st.spinner("AI đang phân tích dữ liệu..."):

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            st.subheader("Cố vấn học tập AI")

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