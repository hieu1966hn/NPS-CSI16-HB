from translation import *
import streamlit as st
from ml_model import MacroIPIModel
from charts_module import MacroCharts
from ai_module import MacroAI

st.set_page_config(
    page_title="Macro Investment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Macro Investment Dashboard")
st.caption("Hệ thống phân tích trạng thái kinh tế vĩ mô bằng Investment Pressure Index")
st.divider()

@st.cache_resource
def load_model():
    model=MacroIPIModel()
    model.initialize()
    return model

model=load_model()
charts=MacroCharts(model)
ai=MacroAI(model)

df=model.get_dataframe()
latest=model.get_latest_row()
dashboard=model.get_latest_dashboard()
st.info(f"""
📊 Investment Pressure Index: {dashboard['IPI']:.2f}

📈 Signal: {dashboard['Signal']}

🌍 Regime: {dashboard['Regime']}

⚠️ Risk Score: {dashboard['Risk_Score']:.2f}

🔄 Reversal Score: {dashboard['Reversal_Score']:.2f}
""")
feature_ranges=model.get_feature_ranges()
years=model.get_available_years()

st.sidebar.title("📂 Điều hướng")

page=st.sidebar.radio(
    "Chọn chức năng",
    [
        "Dashboard",
        "Biểu đồ",
        "Dự báo",
        "AI Assistant",
        "Dataset"
    ]
)

st.sidebar.divider()

st.sidebar.metric(
    "Investment Pressure Index",
    f"{dashboard['IPI']:.2f}"
)

st.sidebar.metric(
    "Signal",
    dashboard["Signal"]
)

st.sidebar.metric(
    "Regime",
    dashboard["Regime"]
)

st.sidebar.divider()

st.sidebar.write("### Thông tin nhanh")

st.sidebar.write(
f"""
Risk Score : **{dashboard['Risk_Score']:.2f}**

Confidence Score : **{dashboard['Confidence_Score']:.2f}**

Macro Cycle : **{dashboard['Macro_Cycle']}**

Dominant Core : **{dashboard['Dominant_Core']}**
"""
)

if page=="Dashboard":

    st.header("📊 Dashboard")

    summary=ai.generate_market_summary()
    
    st.markdown(
    f"""
    <div style="
    background:#eef8ee;
    padding:12px;
    border-radius:10px;
    border-left:6px solid #3aa655;
    font-size:16px;
    line-height:1.8;
    ">
    {summary}
    </div>
    """,
    unsafe_allow_html=True
    )

    c1,c2,c3=st.columns(3)

    with c1:

        st.metric(
            "Investment Pressure Index",
            f"{dashboard['IPI']:.2f}"
        )

        st.metric(
            "Signal",
            dashboard["Signal"]
        )

    with c2:

        st.metric(
            "Regime",
            dashboard["Regime"]
        )

        st.metric(
            "Dominant Core",
            dashboard["Dominant_Core"]
        )

    with c3:

        st.metric(
            "Risk Score",
            f"{dashboard['Risk_Score']:.2f}"
        )

        st.metric(
            "Reversal Score",
            f"{dashboard['Reversal_Score']:.2f}"
        )

    st.divider()

    left,right=st.columns(2)

    with left:

        st.subheader("Investment Pressure Index")

        st.plotly_chart(
            charts.plot_ipi_gauge(dashboard["IPI"]),
            width="stretch"
        )

        st.info(
            charts.get_chart_insight()
        )

    with right:

        st.subheader("Risk Score")

        st.plotly_chart(
            charts.plot_risk_score(),
            width="stretch"
        )

        st.warning(
            charts.get_risk_insight()
        )

    st.divider()

    st.subheader("Macro Dashboard")

    st.plotly_chart(
        charts.plot_macro_dashboard(),
        width="stretch"
    )

    st.divider()

    c1,c2,c3=st.columns(3)

    with c1:

        st.subheader("Confidence Score")

        st.plotly_chart(
            charts.plot_confidence_score(),
            width="stretch"
        )

    with c2:

        st.subheader("Macro Cycle")

        st.plotly_chart(
            charts.plot_macro_cycle(),
            width="stretch"
        )

    with c3:

        st.subheader("Early Warning Index")

        st.plotly_chart(
            charts.plot_early_warning(),
            width="stretch"
        )

    st.divider()

    st.subheader("Macro Core Score")

    c1,c2,c3=st.columns(3)

    with c1:

        st.plotly_chart(
            charts.plot_core_gauge(
                dashboard["Liquidity_FX_Core"],
                "Liquidity"
            ),
            width="stretch"
        )

    with c2:

        st.plotly_chart(
            charts.plot_core_gauge(
                dashboard["Inflation_Yield_Core"],
                "Inflation"
            ),
            width="stretch"
        )

    with c3:

        st.plotly_chart(
            charts.plot_core_gauge(
                dashboard["Valuation_Core"],
                "Valuation"
            ),
            width="stretch"
        )

    st.info(
        charts.get_core_insight()
    )

    st.divider()

    st.subheader("AI Market Summary")

    st.success(
        ai.generate_market_summary()
    )

    st.info(
        ai.generate_core_analysis()
    )

    st.warning(
        ai.generate_regime_report()
    )

    st.error(
        ai.generate_risk_report()
    )

    st.info(
        ai.generate_reversal_report()
    )
elif page=="Biểu đồ":

    st.header("📈 Phân tích vĩ mô")

    st.subheader("Investment Pressure Index")
    st.plotly_chart(
        charts.plot_ipi_trend(),
        width="stretch"
    )
    st.info(
        charts.get_chart_insight()
    )

    st.divider()

    st.subheader("Macro Core Comparison")

    selected=st.multiselect(
        "Chọn Core",
        [
            "Liquidity_FX_Core",
            "Inflation_Yield_Core",
            "Valuation_Core"
        ],
        default=[
            "Liquidity_FX_Core",
            "Inflation_Yield_Core",
            "Valuation_Core"
        ]
    )

    st.plotly_chart(
        charts.plot_core_comparison(selected),
        width="stretch"
    )

    st.info(
        charts.get_core_insight()
    )

    st.divider()

    st.subheader("Macro Factors")

    st.plotly_chart(
        charts.plot_macro_factors_comparison(),
        width="stretch"
    )

    st.caption("So sánh các biến vĩ mô sau khi chuẩn hóa.")

    st.divider()

    st.subheader("Macro Distribution")

    st.plotly_chart(
        charts.plot_macro_distribution(),
        width="stretch"
    )

    st.caption("Phân bố dữ liệu của từng biến kinh tế.")

    st.divider()

    st.subheader("Factor vs Investment Pressure Index")

    factor=st.selectbox(
        "Chọn biến",
        [
            "DXY",
            "USD-VND",
            "CPI",
            "US10Y - VN10Y",
            "Lãi suất liên ngân hàng",
            "P/E"
        ]
    )

    st.plotly_chart(
        charts.plot_factor_vs_ipi(factor),
        width="stretch"
    )

    st.caption("Quan hệ giữa từng biến và Investment Pressure Index.")

    st.divider()

    st.subheader("Heatmap Correlation")

    year=st.selectbox(
        "Chọn năm",
        years
    )

    st.plotly_chart(
        charts.plot_heatmap(
            model.get_data_by_year(year)
        ),
        width="stretch"
    )

    st.info(
        charts.get_heatmap_insight()
    )

    st.divider()

    st.subheader("Market Regime Timeline")

    st.plotly_chart(
        charts.plot_regime_timeline(),
        width="stretch"
    )

    st.info(
        ai.generate_regime_report()
    )

    st.divider()

    st.subheader("Signal Timeline")

    st.plotly_chart(
        charts.plot_signal_timeline(),
        width="stretch"
    )

    st.caption("Lịch sử BUY - HOLD - SELL của mô hình.")

    st.divider()

    st.subheader("Dominant Core Timeline")

    st.plotly_chart(
        charts.plot_dominant_core_timeline(),
        width="stretch"
    )

    st.info(
        ai.generate_core_analysis()
    )

    st.divider()

    st.subheader("Confidence Timeline")

    st.plotly_chart(
        charts.plot_confidence_timeline(),
        width="stretch"
    )

    st.caption("Sự thay đổi mức độ tin cậy của mô hình theo thời gian.")

    st.divider()

    st.subheader("Macro Cycle Timeline")

    st.plotly_chart(
        charts.plot_macro_cycle_timeline(),
        width="stretch"
    )

    st.caption("Diễn biến chu kỳ kinh tế theo từng giai đoạn.")

    st.divider()

    st.subheader("Early Warning Timeline")

    st.plotly_chart(
        charts.plot_warning_timeline(),
        width="stretch"
    )

    st.caption("Theo dõi chỉ số cảnh báo sớm qua thời gian.")

    st.divider()

    st.subheader("Rolling Mean (MA30 & MA90)")

    st.plotly_chart(
        charts.plot_rolling_statistics(),
        width="stretch"
    )

    st.caption("Đường trung bình động 30 và 90 phiên của Investment Pressure Index.")

    st.divider()

    st.subheader("Rolling Volatility")

    st.plotly_chart(
        charts.plot_rolling_volatility(),
        width="stretch"
    )

    st.caption("Biến động của Investment Pressure Index theo thời gian.")

    st.divider()

    st.subheader("Reversal Score")

    st.plotly_chart(
        charts.plot_reversal_score(),
        width="stretch"
    )

    st.warning(
        charts.get_reversal_insight()
    )

    st.info(
        ai.generate_reversal_report()
    )

    st.divider()

    st.subheader("Feature Importance")

    importance=model.get_feature_importance()

    st.plotly_chart(
        charts.plot_feature_importance(
            importance
        ),
        width="stretch"
    )

    st.caption("Mức độ quan trọng của từng biến trong mô hình Random Forest.")

    st.divider()

    st.subheader("Correlation With Investment Pressure Index")

    st.plotly_chart(
        charts.plot_ipi_correlation_bar(),
        width="stretch"
    )

    st.caption("Mức tương quan giữa từng biến vĩ mô và Investment Pressure Index.")
elif page=="Dự báo":

    st.header("🔮 Dự báo vĩ mô")
    
    c1,c2=st.columns(2)

    with c1:

        col1,col2,col3=st.columns([4,1,1])
        with col1:
            dxy=st.number_input(
                "DXY",
                min_value=float(feature_ranges["DXY"]["min"]),
                max_value=float(feature_ranges["DXY"]["max"]),
                value=float(latest["DXY"]),
                key="dxy"
            )
        with col2:
            st.metric("Min",f"{feature_ranges['DXY']['min']:.2f}")
        with col3:
            st.metric("Max",f"{feature_ranges['DXY']['max']:.2f}")

        col1,col2,col3=st.columns([4,1,1])
        with col1:
            usdvnd=st.number_input(
                "USD/VND",
                min_value=float(feature_ranges["USD-VND"]["min"]),
                max_value=float(feature_ranges["USD-VND"]["max"]),
                value=float(latest["USD-VND"]),
                key="usdvnd"
            )
        with col2:
            st.metric("Min",f"{feature_ranges['USD-VND']['min']:.2f}")
        with col3:
            st.metric("Max",f"{feature_ranges['USD-VND']['max']:.2f}")

        col1,col2,col3=st.columns([4,1,1])
        with col1:
            interbank=st.number_input(
                "Lãi suất liên ngân hàng",
                min_value=float(feature_ranges["Lãi suất liên ngân hàng"]["min"]),
                max_value=float(feature_ranges["Lãi suất liên ngân hàng"]["max"]),
                value=float(latest["Lãi suất liên ngân hàng"]),
                key="interbank"
            )
        with col2:
            st.metric("Min",f"{feature_ranges['Lãi suất liên ngân hàng']['min']:.2f}")
        with col3:
            st.metric("Max",f"{feature_ranges['Lãi suất liên ngân hàng']['max']:.2f}")

    with c2:

        col1,col2,col3=st.columns([4,1,1])
        with col1:
            cpi=st.number_input(
                "CPI",
                min_value=float(feature_ranges["CPI"]["min"]),
                max_value=float(feature_ranges["CPI"]["max"]),
                value=float(latest["CPI"]),
                key="cpi"
            )
        with col2:
            st.metric("Min",f"{feature_ranges['CPI']['min']:.2f}")
        with col3:
            st.metric("Max",f"{feature_ranges['CPI']['max']:.2f}")

        col1,col2,col3=st.columns([4,1,1])
        with col1:
            spread=st.number_input(
                "US10Y - VN10Y",
                min_value=float(feature_ranges["US10Y - VN10Y"]["min"]),
                max_value=float(feature_ranges["US10Y - VN10Y"]["max"]),
                value=float(latest["US10Y - VN10Y"]),
                key="spread"
            )
        with col2:
            st.metric("Min",f"{feature_ranges['US10Y - VN10Y']['min']:.2f}")
        with col3:
            st.metric("Max",f"{feature_ranges['US10Y - VN10Y']['max']:.2f}")

        col1,col2,col3=st.columns([4,1,1])
        with col1:
            pe=st.number_input(
                "P/E",
                min_value=float(feature_ranges["P/E"]["min"]),
                max_value=float(feature_ranges["P/E"]["max"]),
                value=float(latest["P/E"]),
                key="pe"
            )
        with col2:
            st.metric("Min",f"{feature_ranges['P/E']['min']:.2f}")
        with col3:
            st.metric("Max",f"{feature_ranges['P/E']['max']:.2f}")

    st.divider()

    if st.button("🚀 Dự báo",width="stretch"):

        result=model.predict_macro_state(
            dxy,
            usdvnd,
            interbank,
            cpi,
            spread,
            pe
        )

        c1,c2,c3,c4=st.columns(4)

        with c1:
            st.metric("IPI",f"{result['IPI']:.2f}")

        with c2:
            st.metric("Signal",result["Signal"])

        with c3:
            st.metric("Regime",result["Regime"])

        with c4:
            st.metric("Risk Score",f"{result['Risk_Score']:.2f}")

        st.divider()

        st.plotly_chart(
            charts.plot_ipi_gauge(result["IPI"]),
            width="stretch"
        )

        st.divider()

        c1,c2,c3=st.columns(3)

        with c1:
            st.plotly_chart(
                charts.plot_core_gauge(
                    result["Liquidity_FX_Core"],
                    "Liquidity"
                ),
                width="stretch"
            )

        with c2:
            st.plotly_chart(
                charts.plot_core_gauge(
                    result["Inflation_Yield_Core"],
                    "Inflation"
                ),
                width="stretch"
            )

        with c3:
            st.plotly_chart(
                charts.plot_core_gauge(
                    result["Valuation_Core"],
                    "Valuation"
                ),
                width="stretch"
            )

        st.info(result["Insight"])
elif page=="AI Assistant":
    st.header("🤖 Macro AI Assistant")
    st.caption("Đặt câu hỏi về Investment Pressure Index, Risk Score, Macro Cycle, DXY, CPI, P/E và các chỉ số kinh tế vĩ mô.")
    if "messages" not in st.session_state:
        st.session_state.messages=[
            {
                "role":"assistant",
                "content":ai.get_initial_message()
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt=st.chat_input("Nhập câu hỏi...")

    if prompt:
        st.session_state.messages.append(
            {
                "role":"user",
                "content":prompt
            }   
        )

        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("MacroBot đang phân tích..."):
            answer=ai.answer_question(prompt)
        st.session_state.messages.append(
            {
                "role":"assistant",
                "content":answer
            }
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

    st.divider()

    c1,c2=st.columns(2)

    with c1:

        if st.button("🗑️ Xóa hội thoại",use_container_width=True):
            ai.clear_history()
            st.session_state.messages=[
                {
                    "role":"assistant",
                    "content":ai.get_initial_message()
                }
            ]
            st.rerun()

    with c2:

        st.download_button(
            label="📄 Tải lịch sử chat",
            data=ai.export_chat(),
            file_name="macrobot_chat.txt",
            mime="text/plain",
            use_container_width=True
        )

elif page=="Dataset":

    st.header("📄 Processed Dataset")

    st.dataframe(
        df,
        width="stretch"
    )

    st.divider()

    st.subheader("Dataset Summary")

    st.json(
        model.dataset_summary()
    )