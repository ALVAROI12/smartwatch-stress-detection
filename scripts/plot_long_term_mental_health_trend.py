import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from pathlib import Path

DATA_URL = "https://ourworldindata.org/grapher/share-with-mental-and-substance-disorders.csv"

OUTPUT_PATH = Path("results/advanced_figures/mental_health_trend_1990_2025.png")

print("Downloading data from Our World in Data…")
headers = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/octet-stream",
    "Referer": "https://ourworldindata.org/mental-health",
}
response = requests.get(
    DATA_URL,
    headers=headers,
    params={"download-format": "tab"},
    timeout=30,
)
response.raise_for_status()
df = pd.read_csv(io.StringIO(response.text), sep="\t")

if "Entity" not in df.columns:
    raise ValueError("Unexpected dataset format—'Entity' column not found.")

indicator_col = [
    col for col in df.columns
    if col.startswith("Prevalence - Mental disorders")
]
if not indicator_col:
    raise ValueError("Mental health prevalence column not found in dataset.")
indicator_col = indicator_col[0]

country = "United States"
country_df = df[(df["Entity"] == country) & (df["Year"] >= 1990)][["Year", indicator_col]].copy()
country_df.rename(columns={indicator_col: "prevalence_percent"}, inplace=True)
country_df.dropna(inplace=True)

if country_df.empty:
    raise ValueError(f"No data available for {country} in the selected period.")

last_year = int(country_df["Year"].max())
if last_year < 2019:
    print(
        "Warning: Dataset ends before 2019; projections will start from last available year."
    )

x = country_df["Year"].values
y = country_df["prevalence_percent"].values
coef = np.polyfit(x, y, deg=1)
pred_years = np.arange(last_year + 1, 2026)
pred_values = np.polyval(coef, pred_years)
future_df = pd.DataFrame({
    "Year": pred_years,
    "prevalence_percent": pred_values,
    "type": "Projected"
})

country_df["type"] = "Observed"
plot_df = pd.concat([country_df, future_df], ignore_index=True)

plt.figure(figsize=(10, 6))
observed = plot_df[plot_df["type"] == "Observed"]
projected = plot_df[plot_df["type"] == "Projected"]

plt.plot(
    observed["Year"],
    observed["prevalence_percent"],
    marker="o",
    color="#1f77b4",
    label="Observed prevalence"
)
plt.plot(
    projected["Year"],
    projected["prevalence_percent"],
    marker="o",
    linestyle="--",
    color="#ff7f0e",
    label="Linear projection"
)

plt.title(
    "Share of U.S. Population Living with Mental or Substance Use Disorders\n"
    "Historical Data (1990-2019) and Linear Projection to 2025"
)
plt.xlabel("Year")
plt.ylabel("Population share (%)")
plt.ylim(0, max(plot_df["prevalence_percent"]) * 1.1)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.annotate(
    "Source: IHME Global Burden of Disease (via Our World in Data)\n"
    "2020-2025 values are a linear projection for illustration",
    xy=(0, -0.15),
    xycoords="axes fraction",
    ha="left",
    fontsize=9
)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300)
print(f"Mental health trend figure saved to {OUTPUT_PATH}")
