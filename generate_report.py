"""
Comprehensive Cross-Validation Report Generator
Creates a detailed image report with improvement recommendations
"""
import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

def create_comprehensive_report(cv_results_path, output_path):
    """Create a comprehensive CV report with all visualizations."""
    
    # Load results
    with open(cv_results_path, 'r') as f:
        results = json.load(f)
    
    fold_results = results['fold_results']
    mean_metrics = results['mean_metrics']
    std_metrics = results['std_metrics']
    overall_metrics = results['overall_test_metrics']
    config = results['config']
    
    n_folds = results['n_folds']
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(5, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Heatwave Prediction Model - Cross-Validation Report\n', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Summary Metrics (top row)
    ax1 = fig.add_subplot(gs[0, :2])
    metrics_names = ['F1 Score', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC']
    mean_vals = [mean_metrics.get('f1', 0), mean_metrics.get('precision', 0), 
                 mean_metrics.get('recall', 0), mean_metrics.get('roc_auc', 0), 
                 mean_metrics.get('pr_auc', 0)]
    std_vals = [std_metrics.get('f1', 0), std_metrics.get('precision', 0), 
                std_metrics.get('recall', 0), std_metrics.get('roc_auc', 0), 
                std_metrics.get('pr_auc', 0)]
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    bars = ax1.barh(metrics_names, mean_vals, xerr=std_vals, color=colors, alpha=0.8, capsize=5)
    ax1.set_xlim(0, 1.1)
    ax1.set_xlabel('Score')
    ax1.set_title('Mean Test Metrics Across Folds', fontweight='bold')
    
    # Add value labels
    for bar, mean, std in zip(bars, mean_vals, std_vals):
        ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{mean:.4f} +/- {std:.4f}', va='center', fontsize=10)
    
    # 2. Fold-by-Fold Performance
    ax2 = fig.add_subplot(gs[0, 2:])
    folds = [f['fold'] for f in fold_results]
    f1_scores = [f['test_metrics']['f1'] for f in fold_results]
    precisions = [f['test_metrics']['precision'] for f in fold_results]
    recalls = [f['test_metrics']['recall'] for f in fold_results]
    
    x = np.arange(n_folds)
    width = 0.25
    
    ax2.bar(x - width, f1_scores, width, label='F1', color='#2ecc71', alpha=0.8)
    ax2.bar(x, precisions, width, label='Precision', color='#3498db', alpha=0.8)
    ax2.bar(x + width, recalls, width, label='Recall', color='#e74c3c', alpha=0.8)
    
    ax2.axhline(y=0.7, color='red', linestyle='--', label='Target (0.70)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Fold {i}' for i in folds])
    ax2.set_ylabel('Score')
    ax2.set_title('Performance by Fold', fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_ylim(0.8, 1.0)
    
    # 3. Confusion Matrix
    ax3 = fig.add_subplot(gs[1, :2])
    tp = int(overall_metrics.get('tp', 0))
    fp = int(overall_metrics.get('fp', 0))
    fn = int(overall_metrics.get('fn', 0))
    tn = int(overall_metrics.get('tn', 0))
    
    cm = np.array([[tn, fp], [fn, tp]])
    im = ax3.imshow(cm, interpolation='nearest', cmap='Blues')
    ax3.figure.colorbar(im, ax=ax3)
    
    classes = ['Non-Heatwave', 'Heatwave']
    ax3.set(xticks=np.arange(2), yticks=np.arange(2),
           xticklabels=classes, yticklabels=classes,
           ylabel='Actual', xlabel='Predicted',
           title='Aggregated Confusion Matrix')
    
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=16, fontweight='bold')
    
    # 4. Class Distribution by Fold
    ax4 = fig.add_subplot(gs[1, 2:])
    train_rates = [f['train_positive_rate'] for f in fold_results]
    test_rates = [f['test_positive_rate'] for f in fold_results]
    
    x = np.arange(n_folds)
    ax4.bar(x - 0.15, train_rates, 0.3, label='Train', color='#3498db', alpha=0.8)
    ax4.bar(x + 0.15, test_rates, 0.3, label='Test', color='#e74c3c', alpha=0.8)
    ax4.axhline(y=np.mean(train_rates), color='#3498db', linestyle='--', alpha=0.5)
    ax4.axhline(y=np.mean(test_rates), color='#e74c3c', linestyle='--', alpha=0.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)])
    ax4.set_ylabel('Positive Rate')
    ax4.set_title('Heatwave Event Distribution', fontweight='bold')
    ax4.legend()
    
    # 5. Training Time Analysis
    ax5 = fig.add_subplot(gs[2, :2])
    times = [f['training_time_seconds'] for f in fold_results]
    ax5.bar(range(1, n_folds + 1), times, color='#9b59b6', alpha=0.8)
    ax5.axhline(y=np.mean(times), color='red', linestyle='--', label=f'Mean: {np.mean(times):.1f}s')
    ax5.set_xlabel('Fold')
    ax5.set_ylabel('Time (seconds)')
    ax5.set_title('Training Time per Fold', fontweight='bold')
    ax5.legend()
    
    # 6. Threshold Analysis
    ax6 = fig.add_subplot(gs[2, 2:])
    thresholds = [f['optimal_threshold'] for f in fold_results]
    ax6.bar(range(1, n_folds + 1), thresholds, color='#f39c12', alpha=0.8)
    ax6.axhline(y=results.get('optimal_overall_threshold', 0.36), color='red', linestyle='--', 
               label=f'Overall: {results.get("optimal_overall_threshold", 0.36):.2f}')
    ax6.set_xlabel('Fold')
    ax6.set_ylabel('Threshold')
    ax6.set_title('Optimal Probability Thresholds', fontweight='bold')
    ax6.legend()
    ax6.set_ylim(0, 0.6)
    
    # 7. Improvement Recommendations
    ax7 = fig.add_subplot(gs[3, :])
    ax7.axis('off')
    
    # Calculate gaps
    target_f1 = 0.70
    target_precision = 0.70
    target_recall = 0.70
    
    f1_gap = target_f1 - mean_metrics.get('f1', 0)
    precision_gap = target_precision - mean_metrics.get('precision', 0)
    recall_gap = target_recall - mean_metrics.get('recall', 0)
    
    # Identify issues
    issues = []
    if recall_gap < 0:
        issues.append(f"Recall exceeds target by {abs(recall_gap):.4f} (good!)")
    else:
        issues.append(f"Recall needs +{recall_gap:.4f} to reach target")
    
    if precision_gap < 0:
        issues.append(f"Precision exceeds target by {abs(precision_gap):.4f} (good!)")
    else:
        issues.append(f"Precision needs +{precision_gap:.4f} to reach target")
    
    if std_metrics.get('f1', 0) > 0.05:
        issues.append(f"High variance in F1 (std: {std_metrics.get('f1', 0):.4f})")
    
    if mean_metrics.get('specificity', 0) < 0.6:
        issues.append(f"Low specificity ({mean_metrics.get('specificity', 0):.4f}) - many false alarms")
    
    # Create recommendations text
    rec_text = "CURRENT PERFORMANCE ANALYSIS:\n"
    rec_text += "=" * 50 + "\n\n"
    
    rec_text += f"F1 Score:  {mean_metrics.get('f1', 0):.4f} +/- {std_metrics.get('f1', 0):.4f}\n"
    rec_text += f"Precision: {mean_metrics.get('precision', 0):.4f} +/- {std_metrics.get('precision', 0):.4f}\n"
    rec_text += f"Recall:    {mean_metrics.get('recall', 0):.4f} +/- {std_metrics.get('recall', 0):.4f}\n"
    rec_text += f"ROC-AUC:   {mean_metrics.get('roc_auc', 0):.4f} +/- {std_metrics.get('roc_auc', 0):.4f}\n"
    rec_text += f"Specificity: {mean_metrics.get('specificity', 0):.4f}\n\n"
    
    rec_text += "ISSUES IDENTIFIED:\n"
    rec_text += "-" * 50 + "\n"
    for issue in issues:
        rec_text += f"  - {issue}\n"
    
    rec_text += "\n" + "RECOMMENDATIONS FOR IMPROVEMENT:\n"
    rec_text += "-" * 50 + "\n"
    rec_text += "  1. Reduce False Alarms: Increase threshold to improve precision\n"
    rec_text += "  2. Feature Engineering: Add seasonal/monthly indicators\n"
    rec_text += "  3. Model Ensemble: Combine RF with Gradient Boosting (XGBoost/LightGBM)\n"
    rec_text += "  4. Hyperparameter Tuning: Grid search for n_estimators, max_depth\n"
    rec_text += "  5. Class Weighting: Adjust class_weight parameter for balance\n"
    rec_text += "  6. Cross-Features: Add interaction terms between temperature and humidity\n"
    
    ax7.text(0.02, 0.95, rec_text, transform=ax7.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 8. Action Plan
    ax8 = fig.add_subplot(gs[4, :])
    ax8.axis('off')
    
    action_items = [
        ("HIGH PRIORITY", [
            "Tune probability threshold (current: {:.2f})".format(results.get('optimal_overall_threshold', 0.36)),
            "Add temporal features (month, season, lag features)",
            "Implement model ensemble (RF + XGBoost)",
        ]),
        ("MEDIUM PRIORITY", [
            "Increase n_estimators to 500 for more stable predictions",
            "Add feature interactions (T x Humidity, T x Pressure)",
            "Implement stratified sampling for class balance",
        ]),
        ("LOW PRIORITY", [
            "Try different model architectures (ConvLSTM, Transformer)",
            "Add more ERA5 variables (wind, cloud cover)",
            "Implement data augmentation techniques",
        ]),
    ]
    
    y_pos = 0.9
    for priority, items in action_items:
        color = '#e74c3c' if priority == 'HIGH PRIORITY' else '#3498db' if priority == 'MEDIUM PRIORITY' else '#95a5a6'
        ax8.text(0.02, y_pos, priority, fontsize=12, fontweight='bold', color=color)
        y_pos -= 0.08
        for item in items:
            ax8.text(0.05, y_pos, f"  - {item}", fontsize=10)
            y_pos -= 0.07
        y_pos -= 0.02
    
    # Model configuration summary
    config_text = f"""
Model Configuration:
  - Sequence Length: {config.get('seq_len', 7)}
  - Future Sequence: {config.get('future_seq', 2)}
  - n_estimators: {config.get('rf_n_estimators', 300)}
  - max_depth: {config.get('rf_max_depth', 20)}
  - Positive Rate: ~{(sum(train_rates)/len(train_rates)):.1%}
    """
    ax8.text(0.6, 0.7, config_text, fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Report saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    results_path = "D:\\Heat-wave-backend\\output\\cv_results_1775669029.json"
    output_path = "D:\\Heat-wave-backend\\output\\cv_detailed_report.png"
    
    create_comprehensive_report(results_path, output_path)