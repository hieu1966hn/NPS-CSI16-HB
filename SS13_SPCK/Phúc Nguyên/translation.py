CORE_TRANSLATION = {
    "Liquidity": "Thanh khoản",
    "Inflation": "Lạm phát",
    "Valuation": "Định giá"
}

REGIME_TRANSLATION = {
    "Risk On": "Chấp nhận rủi ro",
    "Neutral": "Trung lập",
    "Risk Off": "Phòng thủ"
}

SIGNAL_TRANSLATION = {
    "BUY": "MUA",
    "SELL": "BÁN",
    "HOLD": "GIỮ"
}

CYCLE_TRANSLATION = {
    "Recovery": "Phục hồi",
    "Expansion": "Mở rộng",
    "Peak": "Đỉnh",
    "Slowdown": "Suy giảm",
    "Recession": "Suy thoái"
}

COLUMN_TRANSLATION = {
    "IPI": "Investment Pressure Index",
    "Risk_Score": "Điểm rủi ro",
    "Reversal_Score": "Điểm đảo chiều",
    "Confidence_Score": "Điểm tin cậy",
    "Early_Warning": "Cảnh báo sớm",
    "Macro_Cycle": "Chu kỳ kinh tế",
    "Liquidity_FX_Core": "Điểm thanh khoản",
    "Inflation_Yield_Core": "Điểm lạm phát",
    "Valuation_Core": "Điểm định giá",
    "Dominant_Core": "Yếu tố chi phối",
    "Signal": "Tín hiệu",
    "Regime": "Trạng thái"
}

def translate_core(x):
    return CORE_TRANSLATION.get(x, x)

def translate_regime(x):
    return REGIME_TRANSLATION.get(x, x)

def translate_signal(x):
    return SIGNAL_TRANSLATION.get(x, x)

def translate_cycle(x):
    return CYCLE_TRANSLATION.get(x, x)

def translate_column(x):
    return COLUMN_TRANSLATION.get(x, x)