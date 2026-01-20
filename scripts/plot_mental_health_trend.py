import matplotlib.pyplot as plt
import pandas as pd

# Data: KFF analysis of U.S. Census Household Pulse Survey (anxiety or depressive symptoms)
data = {
    "Year": [2019, 2020, 2021, 2022, 2023],
    "Adults reporting symptoms (%)": [11, 31, 36, 32, 32]
}

df = pd.DataFrame(data)

plt.figure(figsize=(8, 5))
plt.plot(df["Year"], df["Adults reporting symptoms (%)"], marker="o", color="#4c72b0")
plt.title("Rise in Anxiety or Depressive Symptoms Among U.S. Adults")
plt.xlabel("Year")
plt.ylabel("Share of adults (%)")
plt.ylim(0, 40)
plt.grid(alpha=0.3)
plt.annotate("COVID-19 onset", xy=(2020, 31), xytext=(2020.4, 34),
             arrowprops=dict(arrowstyle="->", color="#8c1515"), color="#8c1515")
plt.tight_layout()
plt.savefig("results/advanced_figures/mental_health_trend.png", dpi=300)
print("Mental health trend figure saved to results/advanced_figures/mental_health_trend.png")
