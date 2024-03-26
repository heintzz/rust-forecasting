import data
import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

overall_stats = data.overall
rust_stats = data.rust

start_time = overall_stats["start"]
end_time = start_time + len(overall_stats["values"]) * overall_stats["step"]

df = pd.DataFrame(
    {
        "datetime": pd.date_range(start=pd.to_datetime(start_time, unit='s'), periods=len(overall_stats["values"]), freq='h'),
        "overall": overall_stats["values"],
        "rust": rust_stats["values"],
    }
)

model = ARIMA(df["rust"], exog=df["overall"], order=(1, 0, 0))

model_fit = model.fit()

next_datetime = df["datetime"].iloc[-1] + pd.DateOffset(hours=1)
forecast = model_fit.forecast(steps=1, exog=df["overall"].iloc[-1])

plt.title("Rust Players Forecast")
plt.xlabel("Date")
plt.ylabel("Players")

plt.plot(df["datetime"], df["rust"])
plt.plot(next_datetime, forecast.values[0], "ro")
plt.text(next_datetime,
         forecast.values[0], f"{round(forecast.values[0])}", ha='center', va='bottom')

plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.25, left=0.175)

if not os.path.exists("output"):
    os.makedirs("output")

plt.savefig("output/rust_causality.png")

print(f"Forecast for {next_datetime}: {forecast.values[0]}")
