import os
import data
import pandas as pd
import numpy as np
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

forecast_count = 5

df_rust = df["rust"][:-forecast_count]

model = ARIMA(df_rust, order=(3, 3, 3))
model_fit = model.fit()

labels = np.array(df["rust"][-5:])
predictions = np.array(model_fit.forecast(steps=forecast_count))

plt.title("Rust Players Forecast")
plt.xlabel("Date")
plt.ylabel("Players")

plt.plot(df["datetime"][:-forecast_count], df_rust)
for i, prediction in enumerate(predictions):
    plt.plot(df["datetime"].iloc[-forecast_count] +
             pd.DateOffset(hours=i+1), prediction, "bo")

plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.25, left=0.175)

errors = []
for i, label in enumerate(labels):
    abs_percentage_error = (
        np.abs(label - predictions[i]) / np.abs(label)) * 100
    errors.append(abs_percentage_error)

mape = np.mean(errors)
print(mape)

if not os.path.exists("output"):
    os.makedirs("output")

plt.savefig("output/rust_timeseries.png")
