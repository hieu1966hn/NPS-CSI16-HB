# charts_module.py
from translation import *
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MacroCharts:
    def __init__(self, model):
        self.model = model
        self.df = model.df

    COLORS = {
        "IPI": "#2563EB",
        "Liquidity": "#EF4444",
        "Inflation": "#F59E0B",
        "Valuation": "#10B981",
        "Risk": "#8B5CF6"
    }

    def _make_gauge(
        self,
        value,
        title,
        color="#2563EB"
    ):
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(value),
                title={"text": title},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 30], "color": "#DCFCE7"},
                        {"range": [30, 70], "color": "#FEF9C3"},
                        {"range": [70, 100], "color": "#FECACA"}
                    ]
                }
            )
        )
        fig.update_layout(
            height=320,
            margin=dict(
                l=20,
                r=20,
                t=40,
                b=20
            )   
        )
        return fig
    
    def plot_ipi_trend(self):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.df["Date"],
                y=self.df["IPI"],
                mode="lines",
                name="IPI",
                line=dict(width=3, color=self.COLORS["IPI"])
            )
        )
        if "IPI_MA30" in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df["Date"],
                    y=self.df["IPI_MA30"],
                    mode="lines",
                    name="Đường MA30",
                    line=dict(dash="dot")
                )
            )
        if "IPI_MA90" in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df["Date"],
                    y=self.df["IPI_MA90"],
                    mode="lines",
                    name="Đường MA90",
                    line=dict(dash="dash")
                )
            )
        fig.add_hline(y=30, line_dash="dash", annotation_text="Vùng Mua (An toàn)")
        fig.add_hline(y=70, line_dash="dash", annotation_text="Vùng Bán (Rủi ro)")
        fig.update_layout(title="Xu Hướng IPI", height=600, hovermode="x unified")
        return fig

    def plot_ipi_gauge(self, value=None):
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=value,
                title={"text": "Chỉ số IPI"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 30], "color": "lightgreen"},
                        {"range": [30, 70], "color": "gold"},
                        {"range": [70, 100], "color": "salmon"}
                    ]
                }
            )
        )
        fig.update_layout(height=350)
        return fig

    def plot_core_gauge(self, value, title):
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(value),
                number={"valueformat": ".2f"},
                title={"text": title},
                gauge={"axis": {"range": [0, 1]}}
            )
        )
        fig.update_layout(height=300)
        return fig

    def plot_core_comparison(self, selected_cores=None):
        if selected_cores is None:
            selected_cores = ["Liquidity_FX_Core", "Inflation_Yield_Core", "Valuation_Core"]
        fig = go.Figure()
        color_map = {
            "Liquidity_FX_Core": "#EF4444",
            "Inflation_Yield_Core": "#F59E0B",
            "Valuation_Core": "#10B981"
        }
        name_map = {
            "Liquidity_FX_Core": "Thanh Khoản & Tỷ Giá",
            "Inflation_Yield_Core": "Lạm Phát & Lợi Suất",
            "Valuation_Core": "Định Giá"
        }
        for core in selected_cores:
            if core not in self.df.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=self.df["Date"],
                    y=self.df[core],
                    mode="lines",
                    name=name_map.get(core, core),
                    line=dict(width=3, color=color_map.get(core, None))
                )
            )
        fig.update_layout(
            title="So Sánh Các Yếu Tố Cốt Lõi",
            xaxis_title="Ngày",
            yaxis_title="Điểm số",
            hovermode="x unified",
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    def plot_macro_factors_comparison(self):
        fig = go.Figure()
        factors = [
            "DXY_Norm", "USD-VND_Norm", "CPI_Norm", 
            "US10Y - VN10Y_Norm", "Lãi suất liên ngân hàng_Norm", "P/E_Norm"
        ]
        name_map = {
            "DXY_Norm": "Chỉ số DXY",
            "USD-VND_Norm": "Tỷ giá USD/VND",
            "CPI_Norm": "Chỉ số CPI",
            "US10Y - VN10Y_Norm": "Chênh lệch US10Y-VN10Y",
            "Lãi suất liên ngân hàng_Norm": "Lãi suất liên ngân hàng",
            "P/E_Norm": "Định giá P/E"
        }
        for factor in factors:
            if factor not in self.df.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=self.df["Date"],
                    y=self.df[factor],
                    mode="lines",
                    name=name_map[factor],
                    line=dict(width=2)
                )
            )
        fig.update_layout(
            title="So Sánh Các Biến Vĩ Mô (Đã chuẩn hóa)",
            xaxis_title="Ngày",
            yaxis_title="Giá trị chuẩn hóa",
            hovermode="x unified",
            height=650,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    def plot_macro_distribution(self):
        factors = ["DXY", "USD-VND", "CPI", "US10Y - VN10Y", "Lãi suất liên ngân hàng", "P/E"]
        available_factors = [f for f in factors if f in self.df.columns]

        cols = 3
        rows = (len(available_factors) + cols - 1) // cols

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=available_factors,
            vertical_spacing=0.22,
            horizontal_spacing=0.06
        )

        for i, factor in enumerate(available_factors):
            fig.add_trace(
                go.Box(
                    y=self.df[factor],
                    name=factor,
                    marker_color=self.COLORS["IPI"],
                    showlegend=False
                ),
                row=(i // cols) + 1,
                col=(i % cols) + 1
            )

        fig.update_xaxes(showticklabels=False)
        fig.update_layout(title="Phân Bố Dữ Liệu Của Các Biến Vĩ Mô", height=600)
        return fig

    def plot_factor_vs_ipi(self, factor):
        fig = px.scatter(
            self.df, x=factor, y="IPI", trendline="ols", title=f"Mối Quan Hệ Giữa {factor} và Chỉ Số IPI"
        )
        fig.update_layout(height=600)
        return fig

    def plot_heatmap(self, df_year=None):
        if df_year is None:
            df_year = self.df
        cols = [
            "DXY", "USD-VND", "CPI", "US10Y - VN10Y", "Lãi suất liên ngân hàng", "P/E",
            "Liquidity_FX_Core", "Inflation_Yield_Core", "Valuation_Core", "IPI"
        ]
        available_cols = [c for c in cols if c in df_year.columns]
        corr = df_year[available_cols].corr(numeric_only=True).round(2)
        
        fig = px.imshow(
            corr, text_auto=True, aspect="auto", 
            color_continuous_scale="RdBu_r", title="Bản Đồ Nhiệt Tương Quan Các Chỉ Số"
        )
        fig.update_layout(height=750)
        return fig

    def plot_regime_timeline(self):
        df = self.df.copy()
        color_map = {"Risk On": "#10B981", "Neutral": "#F59E0B", "Risk Off": "#EF4444"}
        name_map = {"Risk On": "Thích Rủi Ro", "Neutral": "Trung Lập", "Risk Off": "Né Tránh Rủi Ro"}
        fig = go.Figure()
        for regime in ["Risk On", "Neutral", "Risk Off"]:
            temp = df[df["Regime"] == regime]
            if len(temp) == 0:
                continue
            fig.add_trace(
                go.Scatter(
                    x=temp["Date"],
                    y=temp["IPI"],
                    mode="markers",
                    marker=dict(size=8, color=color_map[regime]),
                    name=name_map[regime]
                )
            )
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["IPI"],
                mode="lines",
                line=dict(width=1, color="gray"),
                showlegend=False
            )
        )
        fig.update_layout(
            title="Dòng Thời Gian Trạng Thái Thị Trường",
            xaxis_title="Ngày",
            yaxis_title="Chỉ số IPI",
            height=650,
            hovermode="x unified"
        )
        return fig

    def plot_rolling_statistics(self):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=self.df["Date"], y=self.df["IPI"], mode="lines", name="IPI", line=dict(width=2))
        )
        if "IPI_MA30" in self.df.columns:
            fig.add_trace(
                go.Scatter(x=self.df["Date"], y=self.df["IPI_MA30"], mode="lines", name="MA30", line=dict(width=3))
            )
        if "IPI_MA90" in self.df.columns:
            fig.add_trace(
                go.Scatter(x=self.df["Date"], y=self.df["IPI_MA90"], mode="lines", name="MA90", line=dict(width=3, dash="dash"))
            )
        fig.update_layout(title="Thống Kê Trượt Trung Bình", xaxis_title="Ngày", yaxis_title="Chỉ số IPI", hovermode="x unified", height=650)
        return fig

    def plot_rolling_volatility(self):
        if "IPI_STD30" not in self.df.columns:
            raise ValueError("Thiếu cột IPI_STD30")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=self.df["Date"], y=self.df["IPI_STD30"], mode="lines", name="Biến động 30 Ngày")
        )
        if "IPI_STD90" in self.df.columns:
            fig.add_trace(
                go.Scatter(x=self.df["Date"], y=self.df["IPI_STD90"], mode="lines", name="Biến động 90 Ngày")
            )
        fig.update_layout(title="Mức Độ Biến Động Của IPI", xaxis_title="Ngày", yaxis_title="Độ Lệch Chuẩn", hovermode="x unified", height=600)
        return fig

    def plot_reversal_score(self):
        if "Reversal_Score" not in self.df.columns:
            raise ValueError("Thiếu cột Reversal_Score")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.df["Date"],
                y=self.df["Reversal_Score"],
                mode="lines",
                name="Điểm Đảo Chiều",
                line=dict(width=3, color="#8B5CF6")
            )
        )
        fig.add_hline(y=10, line_dash="dash", annotation_text="Vùng Theo Dõi")
        fig.add_hline(y=20, line_dash="dash", annotation_text="Rủi Ro Đảo Chiều Cao")
        fig.update_layout(title="Điểm Cảnh Báo Đảo Chiều Động Lực", xaxis_title="Ngày", yaxis_title="Điểm số", hovermode="x unified", height=650)
        return fig

    def plot_feature_importance(self, importance_df):
        fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h", title="Mức Độ Ảnh Hưởng Của Dữ Liệu Tới IPI")
        fig.update_layout(height=600)
        return fig

    def plot_signal_timeline(self):
        if "Signal" not in self.df.columns:
            raise ValueError("Thiếu cột Signal")
        df = self.df.copy()
        signal_map = {"BUY": 1, "HOLD": 0, "SELL": -1}
        df["Signal_Value"] = df["Signal"].map(signal_map)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df["Date"], y=df["Signal_Value"], mode="lines+markers", name="Tín Hiệu")
        )
        fig.update_layout(title="Lịch Sử Tín Hiệu Giao Dịch", xaxis_title="Ngày", yaxis_title="Loại Tín Hiệu", height=600)
        fig.update_yaxes(tickvals=[-1, 0, 1], ticktext=["BÁN", "GIỮ", "MUA"])
        return fig

    def plot_risk_score(self):
        if "Risk_Score" not in self.df.columns:
            raise ValueError("Thiếu cột Risk_Score")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.df["Date"],
                y=self.df["Risk_Score"],
                mode="lines",
                name="Điểm Rủi Ro",
                line=dict(width=3, color="#DC2626")
            )
        )
        fig.update_layout(title="Xu Hướng Của Điểm Rủi Ro", xaxis_title="Ngày", yaxis_title="Điểm Rủi Ro", hovermode="x unified", height=650)
        return fig

    def plot_dominant_core_timeline(self):
        if "Dominant_Core" not in self.df.columns:
            raise ValueError("Thiếu cột Dominant_Core")
        df = self.df.copy()
        core_map = {"Liquidity": 1, "Inflation": 2, "Valuation": 3}
        df["Core_ID"] = df["Dominant_Core"].map(core_map)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df["Date"], y=df["Core_ID"], mode="markers", marker=dict(size=8), name="Yếu Tố Chi Phối")
        )
        fig.update_layout(title="Sự Thay Đổi Các Yếu Tố Cốt Lõi Qua Thời Gian", xaxis_title="Ngày", yaxis_title="Yếu Tố Chi Phối", height=600)
        fig.update_yaxes(tickvals=[1, 2, 3], ticktext=["Thanh Khoản", "Lạm Phát", "Định Giá"])
        return fig

    def plot_ipi_correlation_bar(self):
        factors = ["DXY", "USD-VND", "CPI", "US10Y - VN10Y", "Lãi suất liên ngân hàng", "P/E"]
        correlations = []
        for col in factors:
            if col not in self.df.columns:
                continue
            corr = self.df[col].corr(self.df["IPI"])
            correlations.append({"Factor": col, "Correlation": corr})
        corr_df = pd.DataFrame(correlations)
        fig = px.bar(corr_df, x="Factor", y="Correlation", title="Mức Độ Tương Quan Của Các Biến Với IPI")
        fig.update_layout(height=600)
        return fig

    def get_chart_insight(self):
        latest = self.df.iloc[-1]
        ipi = latest["IPI"]
        if ipi < 30:
            return """
IPI đang nằm trong vùng thấp.
Tình trạng: Rủi ro vĩ mô thấp, thanh khoản thuận lợi, định giá hấp dẫn.
Lịch sử cho thấy đây thường là giai đoạn tích lũy lý tưởng dài hạn.
"""
        elif ipi > 70:
            return """
IPI đang nằm trong vùng cao.
Tình trạng: Rủi ro vĩ mô gia tăng mạnh, thanh khoản bị thắt chặt, định giá không còn rẻ.
Cần ưu tiên hạ tỷ trọng và chú ý khả năng điều chỉnh của thị trường.
"""
        else:
            return """
IPI đang ở vùng trung tính.
Tình trạng: Chưa xuất hiện lợi thế rõ ràng cho cả phe bán lẫn phe mua.
Nên quan sát các chỉ số cốt lõi dẫn dắt để đưa ra quyết định xu hướng tiếp theo.
"""

    def get_core_insight(self):
        latest = self.df.iloc[-1]
        liquidity = latest["Liquidity_FX_Core"]
        inflation = latest["Inflation_Yield_Core"]
        valuation = latest["Valuation_Core"]
        dominant = max({"Liquidity": liquidity, "Inflation": inflation, "Valuation": valuation}, key=lambda x: {"Liquidity": liquidity, "Inflation": inflation, "Valuation": valuation}[x])
        if dominant == "Liquidity":
            return """
Điểm Thanh Khoản đang chiếm ưu thế chi phối.
Yếu tố dẫn dắt chính bao gồm DXY, Tỷ giá USD/VND và Lãi suất liên ngân hàng.
Nền kinh tế lúc này rất nhạy cảm với các đợt rút hoặc bơm ròng dòng tiền.
"""
        elif dominant == "Inflation":
            return """
Điểm Lạm Phát đang chiếm ưu thế chi phối.
Yếu tố dẫn dắt chính bao gồm Chỉ số giá tiêu dùng CPI và chênh lệch lợi suất.
Thị trường đang định giá cao rủi ro tăng lãi suất để đối phó lạm phát.
"""
        else:
            return """
Điểm Định Giá đang chiếm ưu thế chi phối.
Yếu tố dẫn dắt chính là chỉ số P/E.
Tâm điểm của thị trường hiện tại nằm ở yếu tố đắt/rẻ của các loại tài sản.
"""

    def get_heatmap_insight(self):
        cols = ["DXY", "USD-VND", "CPI", "US10Y - VN10Y", "Lãi suất liên ngân hàng", "P/E", "IPI"]
        available_cols = [c for c in cols if c in self.df.columns]
        corr = self.df[available_cols].corr(numeric_only=True)["IPI"].drop("IPI").sort_values(ascending=False)
        strongest = corr.index[0]
        value = corr.iloc[0]
        return f"""
Phân tích tương quan:
Biến có hệ số tương quan mạnh nhất với IPI là {strongest} ({value:.2f}).
Điều này chứng tỏ {strongest} đang đóng vai trò tạo xu hướng lớn nhất đối với bức tranh tổng thể.
"""

    def get_reversal_insight(self):
        latest = self.df.iloc[-1]
        score = latest["Reversal_Score"]
        if score > 25:
            return """
Mức cảnh báo đảo chiều ở mức Rất Cao.
Dấu hiệu cho thấy nhiều biến vĩ mô nền tảng đang xoay chiều cùng lúc. Cần khẩn trương rà soát lại rủi ro trong danh mục.
"""
        elif score > 10:
            return """
Có tín hiệu thay đổi động lực nhưng chưa xác nhận hoàn toàn xu hướng đảo chiều mới. Tiếp tục bám sát các chỉ số lạm phát và thanh khoản.
"""
        return """
Xu hướng vĩ mô hiện tại vẫn duy trì ổn định, chưa có thay đổi đột ngột nào xuất hiện trong các cấu phần.
"""

    def get_risk_insight(self):
        latest = self.df.iloc[-1]
        risk = latest["Risk_Score"]
        if risk >= 70:
            return """
Cảnh báo: Điểm rủi ro đang ở mức nguy hiểm.
Hành động ưu tiên: Bảo toàn vốn, hạn chế margin tối đa và kích hoạt các lệnh chặn lỗ.
"""
        elif risk >= 40:
            return """
Cảnh báo: Điểm rủi ro ở mức trung bình cao.
Hành động: Quản lý chặt chẽ tỷ trọng danh mục, không nên sử dụng đòn bẩy quá lớn vào thời điểm này.
"""
        return """
An toàn: Điểm rủi ro thấp.
Khẩu vị rủi ro tổng thể có thể mở rộng, ủng hộ cho việc nắm giữ tài sản tăng trưởng.
"""

    def get_dashboard_summary(self):
        latest=self.model.get_latest_dashboard()

        signal=translate_signal(latest["Signal"])
        regime=translate_regime(latest["Regime"])
        core=translate_core(latest["Dominant_Core"])
        cycle=translate_cycle(latest["Macro_Cycle"])

        text=f"""
    📊 ĐÁNH GIÁ TỔNG QUAN THỊ TRƯỜNG

    • Investment Pressure Index (IPI): {latest["IPI"]:.2f}

    • Tín hiệu: {signal}

    • Trạng thái: {regime}

    • Core chi phối: {core}

    • Điểm rủi ro: {latest["Risk_Score"]:.2f}

    • Điểm đảo chiều: {latest["Reversal_Score"]:.2f}

    • Điểm tin cậy: {latest["Confidence_Score"]:.2f}

    • Chu kỳ kinh tế: {cycle}

    • Chỉ số cảnh báo sớm: {latest["Early_Warning"]:.2f}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    📌 Nhận định

    {latest["Insight"]}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    💡 Khuyến nghị

    """

        if signal=="MUA":
            text+="Điều kiện vĩ mô thuận lợi. Có thể cân nhắc giải ngân từng phần."
        elif signal=="GIỮ":
            text+="Thị trường đang cân bằng. Nên quan sát thêm các chỉ báo vĩ mô."
        else:
            text+="Rủi ro đang tăng lên. Ưu tiên quản trị danh mục và hạn chế giải ngân."

        return text
    
    def plot_confidence_score(self):
        latest = self.df.iloc[-1]
        return self._make_gauge(
            latest["Confidence_Score"],
            "Độ tin cậy dự báo",
            "#2563EB"
        )

    def plot_early_warning(self):
        latest = self.df.iloc[-1]
        return self._make_gauge(
            latest["Early_Warning"],
            "Chỉ số cảnh báo sớm",
            "#DC2626"
        )

    def plot_macro_cycle(self):
        latest = self.df.iloc[-1]
        cycle = latest["Macro_Cycle"]
        mapping = {
            "Recovery":20,
            "Expansion":40,
            "Peak":60,
            "Slowdown":80,
            "Recession":100
        }
        return self._make_gauge(
            mapping.get(
                cycle,
                50
            ),
            f"Chu kỳ kinh tế\n{cycle}",
            "#0F766E"
        )

    def plot_macro_dashboard(self):
        latest = self.df.iloc[-1]
        fig = px.bar(
            x=[
                "IPI",
                "Risk",
                "Confidence",
                "Warning"
            ],
            y=[
                latest["IPI"],
                latest["Risk_Score"],
                latest["Confidence_Score"],
                latest["Early_Warning"]
            ],
            labels={
                "x":"",
                "y":"Điểm"
            }
        )
        fig.update_layout(
            height=380,
            title="Tổng quan các chỉ số"
        )
        return fig

    def plot_confidence_timeline(self):
        fig = px.line(
            self.df,
            x="Date",
            y="Confidence_Score",
            title="Độ tin cậy dự báo theo thời gian"
        )
        return fig

    def plot_warning_timeline(self):
        fig = px.line(
            self.df,
            x="Date",
            y="Early_Warning",
            title="Chỉ số cảnh báo sớm"
        )
        return fig

    def plot_macro_cycle_timeline(self):
        mapping = {
            "Recovery":1,
            "Expansion":2,
            "Peak":3,
            "Slowdown":4,
            "Recession":5
        }
        df = self.df.copy()
        df["Cycle_Value"] = df["Macro_Cycle"].map(mapping)
        fig = px.line(
            df,
            x="Date",
            y="Cycle_Value",
            title="Chu kỳ kinh tế"
        )
        fig.update_yaxes(
            tickvals=[1,2,3,4,5],
            ticktext=[
                "Recovery",
                "Expansion",
                "Peak",
                "Slowdown",
                "Recession"
            ]
        )
        return fig