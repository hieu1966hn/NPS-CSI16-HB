import pandas as pd

file_path = "macro_data_processed.csv"

df = pd.read_csv(file_path)

df["Confidence_Score"] = (
    100
    - 0.45 * df["Risk_Score"]
    - 5 * abs(df["IPI"] - 50) / 10
).clip(0, 100)

df["Early_Warning"] = (
    0.5 * df["Risk_Score"]
    + 0.5 * abs(df["IPI"] - 50)
).clip(0, 100)

def get_macro_cycle(ipi):
    if ipi < 30:
        return "Recovery"
    elif ipi < 50:
        return "Expansion"
    elif ipi < 70:
        return "Peak"
    elif ipi < 85:
        return "Slowdown"
    else:
        return "Recession"

df["Macro_Cycle"] = df["IPI"].apply(get_macro_cycle)

df.to_csv(
    file_path,
    index=False,
    encoding="utf-8-sig"
)

print("Đã thêm thành công 3 cột:")
print("- Confidence_Score")
print("- Early_Warning")
print("- Macro_Cycle")
print(f"Tổng số cột hiện tại: {len(df.columns)}")