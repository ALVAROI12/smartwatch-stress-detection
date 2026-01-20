import json
from pathlib import Path
import pandas as pd

results_path = Path('results/wesad_ml_results.json')
if not results_path.exists():
    raise FileNotFoundError('results/wesad_ml_results.json not found. Run the evaluation pipeline first.')

with results_path.open('r') as f:
    data = json.load(f)

comparison_rows = data['comparison']
df = pd.DataFrame(comparison_rows)

output_columns = ['Metric', 'Random Forest', 'XGBoost', 'Difference', 'Better_Model']
df = df[output_columns].copy()
df['Random Forest'] = df['Random Forest'].map(lambda x: f"{x:.3f}")
df['XGBoost'] = df['XGBoost'].map(lambda x: f"{x:.3f}")
df['Difference'] = df['Difference'].map(lambda x: f"{x:+.3f}")

table_path = Path('results/advanced_figures/rf_vs_xgb_comparison_table.md')
table_path.parent.mkdir(parents=True, exist_ok=True)
headers = ' | '.join(output_columns)
separator = ' | '.join(['---'] * len(output_columns))
rows = [' | '.join(df.iloc[i].astype(str).tolist()) for i in range(len(df))]
markdown = '\n'.join([headers, separator] + rows)
table_path.write_text(markdown, encoding='utf-8')
print(f"Random Forest vs XGBoost comparison table saved to {table_path}")
