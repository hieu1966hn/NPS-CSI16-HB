# ==========================================================
# ai_module.py
# ==========================================================
from translation import *
import os
import json
import streamlit as st

from dotenv import load_dotenv

from google import genai

load_dotenv()

class MacroAI:

    def __init__(
        self,
        model,
        knowledge_path="knowledge.json"
    ):
        self.model = model
        self.df = model.df
        if "chat_history" not in st.session_state:
            st.session_state.chat_history=[]
        self.history=st.session_state.chat_history
        self.knowledge = self.load_knowledge(
            knowledge_path
        )
        self.client = self.initialize_gemini()

    def initialize_gemini(self):
        api_key = os.getenv(
            "GEMINI_API_KEY"
        )
        if not api_key:
            raise ValueError(
                "Không tìm thấy GEMINI_API_KEY trong file .env"
            )
        return genai.Client(
            api_key=api_key
        )

    def load_knowledge(
        self,
        path
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Không tìm thấy {path}"
            )
        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)

    def get_latest(self):
        return self.df.iloc[-1]

    def build_dashboard_context(self):
        dashboard=self.model.get_latest_dashboard()

        dashboard["Signal"]=translate_signal(dashboard["Signal"])
        dashboard["Regime"]=translate_regime(dashboard["Regime"])
        dashboard["Dominant_Core"]=translate_core(dashboard["Dominant_Core"])
        dashboard["Macro_Cycle"]=translate_cycle(dashboard["Macro_Cycle"])
        context = f"""
Thông tin thị trường hiện tại
Investment Pressure Index (IPI):
{dashboard["IPI"]}
Signal:
{dashboard["Signal"]}
Regime:
{dashboard["Regime"]}
Dominant Core:
{dashboard["Dominant_Core"]}
Risk Score:
{dashboard["Risk_Score"]}
Reversal Score:
{dashboard["Reversal_Score"]}
Confidence Score:
{dashboard["Confidence_Score"]}
Macro Cycle:
{dashboard["Macro_Cycle"]}
Early Warning:
{dashboard["Early_Warning"]}
Insight:
{dashboard["Insight"]}
"""
        return context.strip()

    def add_history(
        self,
        role,
        content
    ):
        self.history.append(
            {
                "role": role,
                "content": content
            }
        )
        self.history = self.history[-20:]

    def build_history(self):
        if not self.history:
            return ""
        text = ""
        for item in self.history:
            if item["role"] == "user":
                text += f"Người dùng: {item['content']}\n"
            else:
                text += f"MacroBot: {item['content']}\n"
        return text
    
    def build_system_prompt(self):
        rules="\n".join(f"- {rule}" for rule in self.knowledge["rules"])
        topics=", ".join(self.knowledge["allowed_topics"])
        prompt=f"""
    Bạn là {self.knowledge["bot_name"]}.

    Vai trò:
    {self.knowledge["role"]}

    Mục tiêu:
    {self.knowledge["goal"]}

    Bạn chỉ được trả lời các chủ đề sau:
    {topics}

    Quy tắc:
    {rules}

    Nếu câu hỏi ngoài phạm vi hãy trả lời đúng câu sau:
    {self.knowledge["out_of_scope_message"]}

    Không được bịa số liệu.
    Không tự tạo dữ liệu.
    Luôn sử dụng Dashboard hiện tại.
    Không được trả lời ngoài phạm vi.
    """
        return prompt.strip()

    def build_prompt(
        self,
        question
    ):
        system_prompt = self.build_system_prompt()
        dashboard = self.build_dashboard_context()
        history = self.build_history()
        prompt = f"""
    {system_prompt}
    ==========================
    Dashboard hiện tại
    {dashboard}
    ==========================
    Lịch sử hội thoại
    {history}
    ==========================
    Câu hỏi của người dùng
    {question}
    Hãy trả lời bằng tiếng Việt.
    """
        return prompt.strip()
        
    def generate_market_summary(self):
        latest=self.get_latest()
        signal=translate_signal(latest["Signal"])
        advice={
            "BUY":"Điều kiện vĩ mô đang khá thuận lợi. Có thể cân nhắc giải ngân từng phần và tiếp tục quản trị tỷ trọng hợp lý.",
            "HOLD":"Thị trường đang ở trạng thái cân bằng. Nên quan sát thêm các chỉ báo vĩ mô trước khi đưa ra quyết định đầu tư.",
            "SELL":"Áp lực vĩ mô đang gia tăng. Nên ưu tiên bảo toàn vốn và hạn chế sử dụng đòn bẩy tài chính."
        }
        return f"""
    📊 ĐÁNH GIÁ TỔNG QUAN THỊ TRƯỜNG
    • Investment Pressure Index (IPI): {latest["IPI"]:.2f}
    • Signal: {latest["Signal"]}
    • Regime: {translate_regime(latest["Regime"])}
    • Core chi phối: {translate_core(latest["Dominant_Core"])}
    • Risk Score: {latest["Risk_Score"]:.2f}
    • Reversal Score: {latest["Reversal_Score"]:.2f}
    • Confidence Score: {latest["Confidence_Score"]:.2f}
    • Macro Cycle: {translate_cycle(latest["Macro_Cycle"])}
    • Early Warning Index: {latest["Early_Warning"]:.2f}
    ━━━━━━━━━━━━━━━━━━━━━━
    📌 Nhận định
    {latest["Insight"]}
    ━━━━━━━━━━━━━━━━━━━━━━
    💡 Khuyến nghị
    {advice.get(signal,"Tiếp tục theo dõi diễn biến kinh tế vĩ mô.")}
    """.strip()

    def generate_risk_report(self):
        latest=self.get_latest()
        risk=latest["Risk_Score"]
        if risk>=80:
            level="Rất cao"
            advice="Ưu tiên bảo toàn vốn, hạn chế giải ngân và tránh sử dụng margin."
        elif risk>=60:
            level="Cao"
            advice="Giảm tỷ trọng cổ phiếu, quan sát diễn biến của DXY và CPI."
        elif risk>=40:
            level="Trung bình"
            advice="Có thể nắm giữ danh mục hiện tại nhưng cần quản trị rủi ro."
        else:
            level="Thấp"
            advice="Môi trường vĩ mô tương đối ổn định, có thể mở rộng khẩu vị đầu tư."
        return f"""
    
    ⚠️ BÁO CÁO RỦI RO
    Risk Score: {risk:.2f}
    Mức đánh giá: {level}
    {advice}
    """.strip()

    def generate_core_analysis(self):
        latest=self.get_latest()
        dominant=translate_core(latest["Dominant_Core"])
        explanation={
            "Liquidity":"Thanh khoản và tỷ giá hiện là yếu tố ảnh hưởng lớn nhất tới thị trường.",
            "Inflation":"Lạm phát đang là động lực chính quyết định xu hướng đầu tư.",
            "Valuation":"Định giá doanh nghiệp hiện đóng vai trò chi phối tâm lý thị trường."
        }
        return f"""
    📈 PHÂN TÍCH CÁC CORE
    Liquidity Core: {latest["Liquidity_FX_Core"]:.2f}
    Inflation Core: {latest["Inflation_Yield_Core"]:.2f}
    Valuation Core: {latest["Valuation_Core"]:.2f}
    Core chi phối:
    {dominant}
    {explanation.get(dominant,"")}
    """.strip()

    def generate_regime_report(self):
        latest=self.get_latest()
        regime=translate_regime(latest["Regime"])
        descriptions={
            "Risk On":"Dòng tiền có xu hướng chấp nhận rủi ro. Điều kiện vĩ mô đang tích cực.",
            "Neutral":"Các yếu tố vĩ mô đang cân bằng. Chưa xuất hiện xu hướng rõ ràng.",
            "Risk Off":"Thị trường ưu tiên phòng thủ do môi trường vĩ mô kém thuận lợi."
        }
        return f"""
    🌍 TRẠNG THÁI THỊ TRƯỜNG
    Regime: {regime}
    {descriptions.get(regime,"")}
    """.strip()

    def generate_reversal_report(self):
        latest=self.get_latest()
        score=latest["Reversal_Score"]
        if score>=25:
            comment="Khả năng đảo chiều xu hướng khá cao."
        elif score>=10:
            comment="Đã xuất hiện dấu hiệu thay đổi xu hướng nhưng chưa xác nhận."
        else:
            comment="Xu hướng hiện tại vẫn khá ổn định."
        return f"""
    🔄 ĐIỂM ĐẢO CHIỀU
    Reversal Score: {score:.2f}
    {comment}
    """.strip()

    def explain_factor(self,factor):
        explanations={
            "dxy":"DXY phản ánh sức mạnh của đồng USD. DXY tăng thường tạo áp lực lên tỷ giá và dòng vốn toàn cầu.",
            "usd":"USD/VND phản ánh áp lực tỷ giá của Việt Nam.",
            "usd/vnd":"USD/VND phản ánh áp lực tỷ giá của Việt Nam.",
            "cpi":"CPI đại diện cho mức lạm phát của nền kinh tế.",
            "p/e":"P/E là chỉ số định giá thị trường dựa trên lợi nhuận doanh nghiệp.",
            "pe":"P/E là chỉ số định giá thị trường dựa trên lợi nhuận doanh nghiệp.",
            "us10y":"US10Y-VN10Y Spread phản ánh chênh lệch lợi suất giữa Mỹ và Việt Nam.",
            "vn10y":"US10Y-VN10Y Spread phản ánh chênh lệch lợi suất giữa Mỹ và Việt Nam.",
            "lãi suất":"Lãi suất liên ngân hàng phản ánh thanh khoản của hệ thống ngân hàng."
        }
        factor=factor.lower()
        for key,value in explanations.items():
            if key in factor:
                return value
        return "Hiện tôi chưa có thông tin chi tiết về chỉ báo này."
    
    def ask_gemini(self,question):
        prompt=self.build_prompt(question)+f"""

    ==========================
    YÊU CẦU QUAN TRỌNG

    Nếu câu hỏi KHÔNG liên quan đến:

    - Investment Pressure Index (IPI)
    - Risk Score
    - Reversal Score
    - Confidence Score
    - Macro Cycle
    - Early Warning
    - Liquidity
    - Inflation
    - Valuation
    - DXY
    - USD/VND
    - CPI
    - P/E
    - US10Y
    - VN10Y
    - Spread
    - Lãi suất
    - Dashboard hiện tại
    - Dự báo từ Dashboard
    - Tín hiệu BUY/HOLD/SELL
    - Regime
    - Dominant Core

    thì KHÔNG được trả lời.

    Chỉ trả về đúng câu sau:

    {self.knowledge["out_of_scope_message"]}

    Nếu câu hỏi nằm trong phạm vi trên:

    - Trả lời như một chuyên gia tư vấn đầu tư.
    - Giải thích dễ hiểu.
    - Ưu tiên sử dụng dữ liệu Dashboard hiện tại.
    - Không bịa số liệu.
    - Nếu không có dữ liệu thì nói rõ.
    - Có thể đưa ra nhận định nhưng phải dựa trên Dashboard.

    ==========================

    Câu hỏi:

    {question}
    """
        try:
            response=self.client.models.generate_content(
                model=os.getenv("MODEL_NAME","gemini-2.5-flash"),
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            return f"Không thể kết nối Gemini.\n\n{e}"

    def answer_question(self,question):
        self.add_history(
            "user",
            question
        )
        answer=self.ask_gemini(question)
        self.add_history(
            "assistant",
            answer
        )
        st.session_state.chat_history=self.history
        return answer
    
    def clear_history(self):
        self.history=[]

    def get_chat_history(self):
        history=[]
        for item in self.history:
            history.append(
                {
                    "role":item["role"],
                    "content":item["content"]
                }
            )
        return history
    
    def export_chat(self):
        text=""
        for item in self.history:
            if item["role"]=="user":
                text+=f"Bạn: {item['content']}\n\n"
            else:
                text+=f"MacroBot: {item['content']}\n\n"
        return text
    
    def get_initial_message(self):
        return self.knowledge["initial_bot_message"]
    
    def get_bot_name(self):
        return self.knowledge["bot_name"]