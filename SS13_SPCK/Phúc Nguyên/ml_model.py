# ==========================================================
# ml_model.py
# ==========================================================

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "DATA"
MODEL_DIR = BASE_DIR / "Models"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RAW_FILE = DATA_DIR / "Macro_Data_2017_2026_Final.csv"
PROCESSED_FILE = DATA_DIR / "macro_data_processed.csv"
MODEL_FILE = MODEL_DIR / "ipi_model.pkl"
SCALER_FILE = MODEL_DIR / "scalers.pkl"

class MacroIPIModel:
    def __init__(self):
        self.df = None
        self.scalers = {}
        self.model = None
        self.feature_importance = None

    def load_data(self):
        if not RAW_FILE.exists():
            raise FileNotFoundError(f"Không tìm thấy file:\n{RAW_FILE}")
        df = pd.read_csv(RAW_FILE)
        df.columns = df.columns.str.strip()

        required_cols = [
            "Date", "US10Y - VN10Y", "CPI", "DXY", 
            "Lãi suất liên ngân hàng", "P/E", "USD-VND"
        ]
        missing = [col for col in required_cols if col not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Thiếu cột dữ liệu: {missing}")

        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        for col in required_cols[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df = df.dropna()

        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Quarter"] = df["Date"].dt.quarter

        self.df = df
        print(f"Loaded {len(df)} rows")
        return df

    def normalize_column(self, column):
        scaler = MinMaxScaler()
        self.df[f"{column}_Norm"] = scaler.fit_transform(self.df[[column]])
        self.scalers[column] = scaler

    def normalize_all(self):
        factors = ["US10Y - VN10Y", "CPI", "DXY", "Lãi suất liên ngân hàng", "P/E", "USD-VND"]
        for factor in factors:
            self.normalize_column(factor)
        print("Normalization completed")

    def calculate_liquidity_core(self):
        self.df["Liquidity_FX_Core"] = self.df[["DXY_Norm", "USD-VND_Norm", "Lãi suất liên ngân hàng_Norm"]].mean(axis=1)

    def calculate_inflation_core(self):
        self.df["Inflation_Yield_Core"] = ((self.df["CPI_Norm"] + (1 - self.df["US10Y - VN10Y_Norm"])) / 2)

    def calculate_valuation_core(self):
        self.df["Valuation_Core"] = self.df["P/E_Norm"]

    def calculate_ipi(self):
        self.df["IPI"] = ((self.df["Liquidity_FX_Core"] * 0.4 + self.df["Inflation_Yield_Core"] * 0.4 + (1 - self.df["Valuation_Core"]) * 0.2) * 100)

    def calculate_risk_score(self):
        self.df["Risk_Score"] = ((self.df["Liquidity_FX_Core"] * 0.45 + self.df["Inflation_Yield_Core"] * 0.35 + self.df["Valuation_Core"] * 0.20) * 100)

    def calculate_regime(self):
        regimes = []
        for _, row in self.df.iterrows():
            ipi = row["IPI"]
            risk = row["Risk_Score"]
            if ipi <= 30 and risk <= 35:
                regimes.append("Risk On")
            elif ipi >= 70 or risk >= 70:
                regimes.append("Risk Off")
            else:
                regimes.append("Neutral")
        self.df["Regime"] = regimes

    def calculate_dominant_core(self):
        dominant = []
        for _, row in self.df.iterrows():
            scores = {
                "Liquidity": row["Liquidity_FX_Core"],
                "Inflation": row["Inflation_Yield_Core"],
                "Valuation": row["Valuation_Core"]
            }
            dominant.append(max(scores, key=scores.get))
        self.df["Dominant_Core"] = dominant

    def build_macro_features(self):
        self.calculate_liquidity_core()
        self.calculate_inflation_core()
        self.calculate_valuation_core()
        self.calculate_ipi()
        self.calculate_risk_score()
        self.calculate_regime()
        self.calculate_dominant_core()
        print("Macro features completed.")

    def calculate_rolling_metrics(self):
        self.df["IPI_MA30"] = self.df["IPI"].rolling(window=30, min_periods=1).mean()
        self.df["IPI_MA90"] = self.df["IPI"].rolling(window=90, min_periods=1).mean()
        self.df["IPI_STD30"] = self.df["IPI"].rolling(window=30, min_periods=2).std().fillna(0)
        self.df["IPI_STD90"] = self.df["IPI"].rolling(window=90, min_periods=2).std().fillna(0)

    def calculate_reversal_score(self):
        change_liquidity = self.df["Liquidity_FX_Core"].diff().abs()
        change_inflation = self.df["Inflation_Yield_Core"].diff().abs()
        change_valuation = self.df["Valuation_Core"].diff().abs()
        change_ipi = self.df["IPI"].diff().abs() / 100
        reversal = (change_liquidity * 0.30 + change_inflation * 0.30 + change_valuation * 0.20 + change_ipi * 0.20)
        self.df["Reversal_Score"] = (reversal * 100).fillna(0)

    def calculate_confidence_score(self):
        """
        Đánh giá độ tin cậy của mô hình.
        """
        volatility = self.df["Rolling_STD30"].fillna(0)
        confidence = (
            100
            - 0.45 * self.df["Risk_Score"]
            - 0.35 * self.df["Reversal_Score"]
            - 50 * volatility
        )
        self.df["Confidence_Score"] = confidence.clip(
            lower=0,
            upper=100
        )

    def calculate_macro_cycle(self):
        """
        Xác định chu kỳ kinh tế.
        """
        cycle = []
        for _, row in self.df.iterrows():
            liq = row["Liquidity_FX_Core"]
            inf = row["Inflation_Yield_Core"]
            val = row["Valuation_Core"]
            if liq < 0.35 and inf < 0.35:
                cycle.append("Recovery")
            elif liq < 0.50 and inf < 0.50:
                cycle.append("Expansion")
            elif val > 0.70:
                cycle.append("Peak")
            elif inf > 0.60:
                cycle.append("Slowdown")
            else:
                cycle.append("Recession")
        self.df["Macro_Cycle"] = cycle

    def calculate_early_warning(self):
        """
        Chỉ số cảnh báo sớm.
        """
        warning = (
            0.50 * self.df["Risk_Score"]
            + 0.35 * self.df["Reversal_Score"]
            + 15 * self.df["Rolling_STD30"].fillna(0)
        )   
        self.df["Early_Warning"] = warning.clip(
            lower=0,
            upper=100
        )

    def generate_signals(self):
        signals = []
        ipi = self.df["IPI"].values
        ma30 = self.df["IPI_MA30"].values
        for i in range(len(self.df)):
            if i == 0:
                signals.append("HOLD")
                continue
            if ipi[i] < 30 and ipi[i] > ma30[i]:
                signals.append("BUY")
            elif ipi[i] > 70 and ipi[i] < ma30[i]:
                signals.append("SELL")
            else:
                signals.append("HOLD")
        self.df["Signal"] = signals

    def generate_insight(self, row):
        messages = []
        if row["Liquidity_FX_Core"] >= 0.75:
            messages.append("Thanh khoản và áp lực tỷ giá đang ở mức cao.")
        elif row["Liquidity_FX_Core"] <= 0.30:
            messages.append("Thanh khoản hệ thống khá thuận lợi.")
        
        if row["Inflation_Yield_Core"] >= 0.70:
            messages.append("Áp lực lạm phát có xu hướng gia tăng.")
        elif row["Inflation_Yield_Core"] <= 0.30:
            messages.append("Lạm phát vẫn đang được kiểm soát.")
            
        if row["Valuation_Core"] >= 0.75:
            messages.append("Định giá thị trường tương đối cao.")
        elif row["Valuation_Core"] <= 0.30:
            messages.append("Định giá thị trường khá hấp dẫn.")
            
        if row["Regime"] == "Risk Off":
            messages.append("Ưu tiên quản trị rủi ro.")
        elif row["Regime"] == "Risk On":
            messages.append("Điều kiện vĩ mô khá thuận lợi.")
            
        if row["Reversal_Score"] >= 20:
            messages.append("Xuất hiện khả năng đảo chiều xu hướng.")
            
        if len(messages) == 0:
            return "Thị trường đang ở trạng thái cân bằng."
        return " ".join(messages)

    def build_insights(self):
        self.df["Insight"] = self.df.apply(self.generate_insight, axis=1)
        print("Insight generation completed.")

    def train_model(self):
        features = ["DXY", "USD-VND", "Lãi suất liên ngân hàng", "CPI", "US10Y - VN10Y", "P/E"]
        X = self.df[features]
        y = self.df["IPI"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

        model = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        mse = mean_squared_error(y_test, prediction)
        print(f"Random Forest MSE = {mse:.4f}")
        
        self.model = model
        self._build_feature_importance()
        return mse

    def _build_feature_importance(self):
        features = self.get_factor_columns()
        importance = pd.DataFrame({"Feature": features, "Importance": self.model.feature_importances_})
        importance = importance.sort_values("Importance", ascending=False).reset_index(drop=True)
        self.feature_importance = importance

    def get_feature_importance(self):
        if self.feature_importance is None:
            raise ValueError("Model chưa được train.")
        return self.feature_importance.copy()

    def save_model(self):
        if self.model is None:
            raise ValueError("Model chưa được train.")
        joblib.dump(self.model, MODEL_FILE)
        joblib.dump(self.scalers, SCALER_FILE)
        print(f"Model saved -> {MODEL_FILE}")
        print(f"Scaler saved -> {SCALER_FILE}")

    def save_processed_data(self):
        self.df.to_csv(PROCESSED_FILE, index=False, encoding="utf-8-sig")
        print(f"Processed data saved -> {PROCESSED_FILE}")

    def load_model(self):
        if MODEL_FILE.exists():
            self.model = joblib.load(MODEL_FILE)
            self._build_feature_importance()
        if SCALER_FILE.exists():
            self.scalers = joblib.load(SCALER_FILE)

    def get_feature_ranges(self):
        features = ["DXY", "USD-VND", "Lãi suất liên ngân hàng", "CPI", "US10Y - VN10Y", "P/E"]
        ranges = {}
        for feature in features:
            ranges[feature] = {
                "min": round(float(self.df[feature].min()), 2),
                "max": round(float(self.df[feature].max()), 2),
                "mean": round(float(self.df[feature].mean()), 2)
            }
        return ranges

    def normalize_single(self, column, value):
        if column not in self.scalers:
            raise ValueError(f"Scaler của {column} chưa tồn tại.")
        scaler = self.scalers[column]
        value = np.array([[value]])
        return float(scaler.transform(value)[0][0])

    def calculate_cores_from_input(self, dxy, usdvnd, interbank_rate, cpi, spread, pe):
        dxy_norm = self.normalize_single("DXY", dxy)
        usdvnd_norm = self.normalize_single("USD-VND", usdvnd)
        rate_norm = self.normalize_single("Lãi suất liên ngân hàng", interbank_rate)
        cpi_norm = self.normalize_single("CPI", cpi)
        spread_norm = self.normalize_single("US10Y - VN10Y", spread)
        pe_norm = self.normalize_single("P/E", pe)

        liquidity_core = (dxy_norm + usdvnd_norm + rate_norm) / 3
        inflation_core = (cpi_norm + (1 - spread_norm)) / 2
        valuation_core = pe_norm

        return {
            "Liquidity_FX_Core": liquidity_core,
            "Inflation_Yield_Core": inflation_core,
            "Valuation_Core": valuation_core
        }

    def calculate_ipi_from_core(self, liquidity_core, inflation_core, valuation_core):
        return ((liquidity_core * 0.4 + inflation_core * 0.4 + (1 - valuation_core) * 0.2) * 100)

    def calculate_risk_from_core(self, liquidity_core, inflation_core, valuation_core):
        return ((liquidity_core * 0.45 + inflation_core * 0.35 + valuation_core * 0.20) * 100)

    def classify_regime(self, ipi, risk):
        if ipi <= 30 and risk <= 35:
            return "Risk On"
        elif ipi >= 70 or risk >= 70:
            return "Risk Off"
        else:
            return "Neutral"

    def dominant_core(self, liquidity, inflation, valuation):
        scores = {"Liquidity": liquidity, "Inflation": inflation, "Valuation": valuation}
        return max(scores, key=scores.get)

    def classify_ipi(self, ipi):
        if ipi < 30:
            return "BUY"
        elif ipi > 70:
            return "SELL"
        else:
            return "HOLD"

    def predict_macro_state(self, dxy, usdvnd, interbank_rate, cpi, spread, pe):
        cores = self.calculate_cores_from_input(dxy, usdvnd, interbank_rate, cpi, spread, pe)
        ipi = self.calculate_ipi_from_core(cores["Liquidity_FX_Core"], cores["Inflation_Yield_Core"], cores["Valuation_Core"])
        risk = self.calculate_risk_from_core(cores["Liquidity_FX_Core"], cores["Inflation_Yield_Core"], cores["Valuation_Core"])
        regime = self.classify_regime(ipi, risk)
        signal = self.classify_ipi(ipi)
        dominant = self.dominant_core(cores["Liquidity_FX_Core"], cores["Inflation_Yield_Core"], cores["Valuation_Core"])
        insight = self.generate_insight({
            "Liquidity_FX_Core": cores["Liquidity_FX_Core"],
            "Inflation_Yield_Core": cores["Inflation_Yield_Core"],
            "Valuation_Core": cores["Valuation_Core"],
            "IPI": ipi,
            "Risk_Score": risk,
            "Regime": regime,
            "Reversal_Score": 0
        })
        confidence = max(
            0,
            100 - 0.45 * risk - 5 * abs(ipi - 50) / 10
        )

        warning = min(
            100,
            0.5 * risk + 0.5 * abs(ipi - 50)
        )

        if ipi < 30:
            cycle = "Recovery"
        elif ipi < 50:
            cycle = "Expansion"
        elif ipi < 70:
            cycle = "Peak"
        elif ipi < 85:
            cycle = "Slowdown"
        else:
            cycle = "Recession"

        return {
            "Liquidity_FX_Core": round(cores["Liquidity_FX_Core"], 4),
            "Inflation_Yield_Core": round(cores["Inflation_Yield_Core"], 4),
            "Valuation_Core": round(cores["Valuation_Core"], 4),
            "IPI": round(ipi, 2),
            "Risk_Score": round(risk, 2),
            "Confidence_Score": round(confidence, 2),
            "Macro_Cycle": cycle,
            "Early_Warning": round(warning, 2),
            "Signal": signal,
            "Regime": regime,
            "Dominant_Core": dominant,
            "Insight": insight
        }

    def run_pipeline(self):
        self.load_data()
        self.normalize_all()
        self.build_macro_features()
        self.calculate_rolling_metrics()
        self.calculate_reversal_score()
        self.generate_signals()
        self.calculate_confidence_score()
        self.calculate_macro_cycle()
        self.calculate_early_warning()
        self.build_insights()
        self.train_model()
        self.save_model()
        self.save_processed_data()

    def initialize(self):
        required_columns = [
            "Date", "US10Y - VN10Y", "CPI", "DXY", "Lãi suất liên ngân hàng", "P/E", "USD-VND",
            "US10Y - VN10Y_Norm", "CPI_Norm", "DXY_Norm", "Lãi suất liên ngân hàng_Norm", "P/E_Norm", "USD-VND_Norm",
            "Liquidity_FX_Core", "Inflation_Yield_Core", "Valuation_Core",
            "IPI", "Risk_Score", "Regime", "Dominant_Core", "Signal", "Insight", "Reversal_Score",
            "IPI_MA30", "IPI_MA90", "IPI_STD30", "IPI_STD90"
        ]
        rebuild = False

        if not RAW_FILE.exists():
            raise FileNotFoundError(f"Không tìm thấy:\n{RAW_FILE}")
        if not PROCESSED_FILE.exists():
            print("Processed Data chưa tồn tại.")
            rebuild = True

        if not rebuild:
            raw_time = RAW_FILE.stat().st_mtime
            processed_time = PROCESSED_FILE.stat().st_mtime
            if raw_time > processed_time:
                print("Raw CSV đã được cập nhật.")
                rebuild = True

        if not rebuild:
            self.df = pd.read_csv(PROCESSED_FILE)
            self.df["Date"] = pd.to_datetime(self.df["Date"])

        if not rebuild:
            missing = [c for c in required_columns if c not in self.df.columns]
            if len(missing) > 0:
                print("Thiếu cột:", missing)
                rebuild = True

        if not rebuild:
            raw_rows = len(pd.read_csv(RAW_FILE))
            processed_rows = len(self.df)
            if raw_rows != processed_rows:
                print("Số dòng dữ liệu thay đổi.")
                rebuild = True

        if not rebuild and not MODEL_FILE.exists():
            print("Model chưa tồn tại.")
            rebuild = True

        if not rebuild and not SCALER_FILE.exists():
            print("Scaler chưa tồn tại.")
            rebuild = True

        if not rebuild and MODEL_FILE.exists():
            if RAW_FILE.stat().st_mtime > MODEL_FILE.stat().st_mtime:
                print("Model đã cũ.")
                rebuild = True

        if not rebuild and SCALER_FILE.exists():
            if RAW_FILE.stat().st_mtime > SCALER_FILE.stat().st_mtime:
                print("Scaler đã cũ.")
                rebuild = True

        if rebuild:
            print("\n============================================================")
            print("AUTO REBUILD PIPELINE")
            print("============================================================\n")
            self.run_pipeline()
        else:
            print("Processed Data hợp lệ.")
            self.load_model()
            print("Model loaded.")
            print("Scaler loaded.")

        print("\n============================================================")
        print("SYSTEM READY")
        print("============================================================\n")

    def validate_dataset(self):
        if self.df is None:
            raise ValueError("Dataset chưa được load.")
        if self.df.empty:
            raise ValueError("Dataset rỗng.")
            
        required_columns = [
            "Date", "US10Y - VN10Y", "CPI", "DXY", "Lãi suất liên ngân hàng", "P/E", "USD-VND",
            "Liquidity_FX_Core", "Inflation_Yield_Core", "Valuation_Core",
            "IPI", "Signal", "Risk_Score", "Dominant_Core", "Regime", "Reversal_Score", "Insight"
        ]
        
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if len(missing_columns) > 0:
            raise ValueError(f"Thiếu các cột: {missing_columns}")
            
        check_df = self.df[required_columns]
        total_nan = check_df.isnull().sum().sum()
        if total_nan > 0:
            print("\n===== NaN CHECK =====")
            print(check_df.isnull().sum())
            raise ValueError(f"Các cột bắt buộc còn {total_nan} giá trị NaN.")
            
        if len(self.df) < 30:
            raise ValueError("Dataset quá ít dữ liệu để phân tích.")
        return True

    def dataset_summary(self):
        self.validate_dataset()
        return {
            "Rows": len(self.df),
            "Columns": len(self.df.columns),
            "Start": self.df["Date"].min(),
            "End": self.df["Date"].max(),
            "Years": sorted(self.df["Year"].unique().tolist())
        }

    def get_dataframe(self):
        self.validate_dataset()
        return self.df.copy()

    def get_latest_row(self):
        self.validate_dataset()
        return self.df.iloc[-1]

    def get_core_columns(self):
        return ["Liquidity_FX_Core", "Inflation_Yield_Core", "Valuation_Core"]

    def get_factor_columns(self):
        return ["DXY", "USD-VND", "Lãi suất liên ngân hàng", "CPI", "US10Y - VN10Y", "P/E"]

    def validate_input(self, feature, value):
        if feature not in self.df.columns:
            raise ValueError(f"{feature} không tồn tại.")
        minimum = float(self.df[feature].min())
        maximum = float(self.df[feature].max())
        return (minimum <= value <= maximum)

    def validate_prediction_inputs(self, dxy, usdvnd, interbank_rate, cpi, spread, pe):
        inputs = {
            "DXY": dxy, "USD-VND": usdvnd, "Lãi suất liên ngân hàng": interbank_rate,
            "CPI": cpi, "US10Y - VN10Y": spread, "P/E": pe
        }
        invalid = []
        for feature, value in inputs.items():
            if not self.validate_input(feature, value):
                invalid.append(feature)
        return invalid

    def get_available_years(self):
        self.validate_dataset()
        return sorted(self.df["Year"].unique().tolist())

    def get_data_by_year(self, year):
        self.validate_dataset()
        return self.df[self.df["Year"] == year].copy()

    def get_latest_dashboard(self):
        latest = self.get_latest_row()
        dashboard = {
            "Date": latest["Date"],
            "IPI": round(float(latest["IPI"]), 2),
            "Risk_Score": round(float(latest["Risk_Score"]), 2),
            "Signal": latest["Signal"],
            "Regime": latest["Regime"],
            "Dominant_Core": latest["Dominant_Core"],
            "Reversal_Score": round(float(latest["Reversal_Score"]), 2),
            "Confidence_Score": round(float(latest["Confidence_Score"]), 2),
            "Macro_Cycle": latest["Macro_Cycle"],
            "Early_Warning": round(float(latest["Early_Warning"]), 2),
            "Liquidity_FX_Core": round(float(latest["Liquidity_FX_Core"]), 3),
            "Inflation_Yield_Core": round(float(latest["Inflation_Yield_Core"]), 3),
            "Valuation_Core": round(float(latest["Valuation_Core"]), 3),
            "Insight": latest["Insight"]
        }
        return dashboard

    def get_prediction_template(self):
        latest = self.get_latest_row()
        template = {}
        for feature in self.get_factor_columns():
            template[feature] = {
                "min": round(float(self.df[feature].min()), 2),
                "max": round(float(self.df[feature].max()), 2),
                "mean": round(float(self.df[feature].mean()), 2),
                "latest": round(float(latest[feature]), 2)
            }
        return template