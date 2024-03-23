import data
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

rust_stats = data.rust

start_time = rust_stats["start"]
end_time = start_time + len(rust_stats["values"]) * rust_stats["step"]

df = pd.DataFrame(
    {
        "datetime": pd.date_range(start=pd.to_datetime(start_time, unit='s'), periods=len(rust_stats["values"]), freq='h'),
        "rust": rust_stats["values"],
    }
)

model = ARIMA(df["rust"], order=(1, 0, 0))

model_fit = model.fit()

next_datetime = df["datetime"].iloc[-1] + pd.DateOffset(hours=1)
forecast = model_fit.forecast()

plt.title("Rust Players Forecast")
plt.xlabel("Date")
plt.ylabel("Players")

plt.plot(df["datetime"], df["rust"], label="Actual")
plt.plot(next_datetime, forecast.values[0], "ro", label="Forecast")

plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.25, left=0.175)

plt.savefig("rust_timeseries.png")

print(f"Forecast for {next_datetime}: {forecast.values[0]}")
