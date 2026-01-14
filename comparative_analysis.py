import pandas as pd
import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from src.data_loader import clean_missing_values
from src.preprocessing import encode_labels, split_data, scale_features
from src.genetic_programming import setup_gp, run_active_learning_ga, run_ga, GPClassifierWrapper
from src.ensemble_learning import GPEnsembleClassifier

def save_experiment_results(results, plots_dir='experiment/plots'):
    """
    Saves metrics and plots to the experiment directory.
    """
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save metrics to metrics.json
    with open('experiment/metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    # Generate report.md
    with open('experiment/report.md', 'w') as f:
        f.write("# Comparative Analysis Report\n\n")
        f.write("## Performance Metrics\n\n")
        f.write("| Method | Accuracy | Precision | Recall | F1-Score | Time (s) |\n")
        f.write("|--------|----------|-----------|--------|----------|----------|\n")
        for method, metrics in results.items():
            f.write(f"| {method} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['time']:.2f} |\n")
            
    print(f"Results saved to experiment/ directory.")

def plot_comparisons(results, plots_dir='experiment/plots'):
    """
    Generates comparison plots.
    """
    methods = list(results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1']
    
    # Bar plot for metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    width = 0.2
    
    for i, metric in enumerate(metrics_names):
        values = [results[m][metric] for m in methods]
        ax.bar(x + i*width, values, width, label=metric.capitalize())
        
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison by Method')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(methods)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/metrics_comparison.png")
    plt.close()

    # Time comparison
    plt.figure(figsize=(8, 5))
    times = [results[m]['time'] for m in methods]
    plt.bar(methods, times, color='skyblue')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time Comparison')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/time_comparison.png")
    plt.close()

def evaluate_model(y_true, y_pred, elapsed_time):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'time': elapsed_time
    }

def main():
    raw_data_path = 'data/atlas-higgs-challenge-2014-v2.csv'
    print(f"Loading data subset from {raw_data_path}...")
    
    try:
        df = pd.read_csv(raw_data_path, nrows=5000)
    except FileNotFoundError:
        print(f"Error: {raw_data_path} not found.")
        return

    # Preprocessing
    df = clean_missing_values(df)
    df, le = encode_labels(df)
    X_train, X_test, y_train, y_test, _, _ = split_data(df, test_size=0.3)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
    
    y_test_vals = y_test.values
    
    # Common GP Params
    n_features = X_train_scaled.shape[1]
    pset, toolbox = setup_gp(n_features)
    n_gen = 10
    pop_size = 50
    
    results = {}

    # 1. GA (Baseline)
    print("\n>>> Running GA (Baseline)...")
    start_time = time.time()
    pop_ga, _, hof_ga = run_ga(X_train_scaled, y_train.values, pset, toolbox, n_gen=n_gen, pop_size=pop_size)
    elapsed_ga = time.time() - start_time
    
    best_ind_ga = hof_ga[0]
    func_ga = toolbox.compile(expr=best_ind_ga)
    preds_ga = [1 if func_ga(*p) > 0 else 0 for p in X_test_scaled]
    results['GA'] = evaluate_model(y_test_vals, preds_ga, elapsed_ga)

    # 2. GA + AL
    print("\n>>> Running GA + Active Learning...")
    # Setup for AL
    n_initial = 200
    X_train_initial = X_train_scaled[:n_initial]
    y_train_initial = y_train.values[:n_initial]
    X_pool = X_train_scaled[n_initial:]
    y_pool = y_train.values[n_initial:]
    
    start_time = time.time()
    pop_al, _, hof_al = run_active_learning_ga(
        X_train_initial, y_train_initial, 
        X_pool, y_pool, 
        pset, toolbox, 
        n_gen=n_gen, pop_size=pop_size, 
        k=3, n_instances=20
    )
    elapsed_al = time.time() - start_time
    
    best_ind_al = hof_al[0]
    func_al = toolbox.compile(expr=best_ind_al)
    preds_al = [1 if func_al(*p) > 0 else 0 for p in X_test_scaled]
    results['GA+AL'] = evaluate_model(y_test_vals, preds_al, elapsed_al)

    # 3. GA + AL + EL
    print("\n>>> Evaluating GA + AL + Ensemble (Soft Voting)...")
    # We use the population from GA+AL run to create ensemble
    start_time = time.time()
    # Sorted by fitness
    sorted_pop = sorted(pop_al, key=lambda ind: ind.fitness.values[0], reverse=True)
    ensemble = GPEnsembleClassifier(sorted_pop[:10], toolbox)
    preds_el = ensemble.predict(X_test_scaled, voting='soft')
    elapsed_el = elapsed_al + (time.time() - (start_time + elapsed_al)) # Time to train + time to ensemble
    # Actually training time for GA+AL+EL is AL time + negligible ensemble creation time
    elapsed_total_el = elapsed_al + (time.time() - start_time) 
    
    results['GA+AL+EL'] = evaluate_model(y_test_vals, preds_el, elapsed_total_el)

    # Save and Plot
    save_experiment_results(results)
    plot_comparisons(results)
    
    # Specific plots for GA+AL+EL (as requested)
    plots_dir = 'experiment/plots'
    # Confusion Matrix for GA+AL+EL
    cm = confusion_matrix(y_test_vals, preds_el)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix: GA+AL+EL')
    plt.savefig(f"{plots_dir}/confusion_matrix.png")
    plt.close()

    # ROC Curve for GA+AL+EL
    probs_el = ensemble.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_vals, probs_el)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: GA+AL+EL')
    plt.legend(loc="lower right")
    plt.savefig(f"{plots_dir}/roc_curve.png")
    plt.close()

    print("\nComparison Summary:")
    print(pd.DataFrame(results).T)

if __name__ == "__main__":
    main()
