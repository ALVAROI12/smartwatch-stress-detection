import json
import numpy as np

with open('results/wesad_ml_results.json', 'r') as f:
    results = json.load(f)

rows = []
for model_key, label in [('random_forest', 'Random Forest'), ('xgboost', 'XGBoost')]:
    if model_key not in results:
        continue
    cv = results[model_key]['cv_scores']
    acc = np.mean(cv.get('test_accuracy', cv.get('accuracy', [])))
    prec = np.mean(cv.get('test_precision', cv.get('precision', [])))
    rec = np.mean(cv.get('test_recall', cv.get('recall', [])))
    f1 = np.mean(cv.get('test_f1', cv.get('f1', [])))
    rows.append([label, acc, prec, rec, f1])

with open('results/advanced_figures/classification_accuracy_comparison.md', 'w') as f:
    f.write('| Model         | Accuracy | Precision | Recall | F1-score |\n')
    f.write('|---------------|----------|-----------|--------|----------|\n')
    for row in rows:
        f.write(f'| {row[0]:<13} | {row[1]:.3f}   | {row[2]:.3f}    | {row[3]:.3f} | {row[4]:.3f}   |\n')
print('Classification accuracy comparison table saved to results/advanced_figures/classification_accuracy_comparison.md')
