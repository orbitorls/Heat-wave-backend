#!/usr/bin/env python3
"""
Test different heatwave thresholds for Thailand weather.

This script:
1. Loads ERA5 data
2. Tests different thresholds (35C, 36C, 38C, 40C)
3. Shows class balance for each threshold
4. Trains models with different thresholds
5. Compares results
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def analyze_threshold_effects():
    """Analyze how different thresholds affect class balance."""
    print("=" * 70)
    print(" THRESHOLD ANALYSIS FOR THAILAND HEATWAVE")
    print("=" * 70)

    print("\nThailand Temperature Context:")
    print("  - Summer (Mar-May): Regularly 35-40C")
    print("  - Hot season peak: Can reach 40-44C")
    print("  - Normal summer days: 32-38C")
    print("  - Cool season: 25-32C")

    print("\n" + "-" * 70)
    print(" Threshold Comparison:")
    print("-" * 70)

    thresholds = [
        (35, "Detect hot days (moderate heat stress)"),
        (36, "Detect very hot days (moderate-severe heat stress)"),
        (38, "Detect extreme heat (severe heat stress) - CURRENT"),
        (40, "Detect extreme heatwaves (life-threatening)"),
    ]

    print("\n{:<12} {:<20} {:<40}".format("Threshold", "Positive Rate", "Description"))
    print("-" * 80)

    for threshold, desc in thresholds:
        # Estimate positive rate
        if threshold <= 35:
            est_rate = "40-60%"
            impact = "High recall, many detections"
        elif threshold == 36:
            est_rate = "30-50%"
            impact = "Moderate recall"
        elif threshold == 38:
            est_rate = "10-20%"
            impact = "Low recall, high precision"
        else:  # 40C
            est_rate = "2-5%"
            impact = "Very low recall, highest precision"

        print("{:<12} {:<20} {:<40}".format(f"{threshold}C", est_rate, f"{desc}"))

    print("\n" + "=" * 70)
    print(" WHY NOT 40C?")
    print("=" * 70)

    print("""
    PROS of 40C Threshold:
    - Detects ONLY life-threatening extreme heat
    - Very high precision (when it says heatwave, it's really bad)
    - Useful for emergency alerts
    
    CONS of 40C Threshold:
    - VERY LOW positive rate (2-5%)
    - Severe class imbalance
    - Model may struggle to learn (few positive examples)
    - May miss many dangerous heat events
    - Not suitable for general heatwave warning
    
    RECOMMENDATION:
    - Use 38C for general heatwave detection (current)
    - Use 40C ONLY for extreme heat alerts
    - Use 35-36C for moderate heat warnings
    
    Let's train models with all thresholds and compare!
    """)

    return True


def train_with_thresholds(thresholds):
    """Train models with specified thresholds."""
    import time

    print("\n" + "=" * 70)
    print(" TRAINING MODELS WITH DIFFERENT THRESHOLDS")
    print("=" * 70)

    results = []

    for threshold in thresholds:
        print(f"\n{'=' * 70}")
        print(f" Training with threshold {threshold}C")
        print(f"{'=' * 70}")

        start_time = time.time()

        try:
            # Set environment variable for threshold
            os.environ["HW_HEATWAVE_TEMP_THRESHOLD"] = str(threshold)

            # Import and run training
            import train_daily_xgboost

            config = {
                "heatwave_temp_threshold": threshold,
                "heatwave_min_duration": 3,
                "n_estimators": 200,
                "max_depth": 10,
                "learning_rate": 0.1,
                "random_seed": 42,
            }

            print(f"\n[+] Loading data...")
            result = train_daily_xgboost.train_xgboost_daily(config=config)

            training_time = time.time() - start_time

            if result:
                test_m = result.get("test_metrics", {})
                val_m = result.get("val_metrics", {})
                train_m = result.get("train_metrics", {})
                meta = result.get("metadata", result)

                print(f"\n[+] Training completed in {training_time:.1f}s")
                print(f"\n  Threshold: {threshold}C")
                print(
                    f"  Train Positive Rate: {meta.get('train_positive_rate', 'N/A')}"
                )
                print(f"  Val Positive Rate: {meta.get('val_positive_rate', 'N/A')}")
                print(f"  Test Positive Rate: {meta.get('test_positive_rate', 'N/A')}")

                print(f"\n  Test Metrics:")
                print(f"    F1:        {test_m.get('f1', 'N/A')}")
                print(f"    Precision: {test_m.get('precision', 'N/A')}")
                print(f"    Recall:    {test_m.get('recall', 'N/A')}")
                print(f"    Accuracy:  {test_m.get('accuracy', 'N/A')}")

                results.append(
                    {
                        "threshold": threshold,
                        "test_metrics": test_m,
                        "val_metrics": val_m,
                        "train_metrics": train_m,
                        "metadata": meta,
                        "training_time": training_time,
                        "success": True,
                    }
                )
            else:
                print(f"\n[!] Training returned no result for threshold {threshold}C")
                results.append(
                    {
                        "threshold": threshold,
                        "success": False,
                        "error": "No result returned",
                    }
                )

        except Exception as e:
            print(f"\n[!] Training failed for threshold {threshold}C: {e}")
            import traceback

            traceback.print_exc()
            results.append({"threshold": threshold, "success": False, "error": str(e)})

    return results


def compare_results(results):
    """Compare all results and provide recommendations."""
    print("\n" + "=" * 70)
    print(" THRESHOLD COMPARISON RESULTS")
    print("=" * 70)

    print(
        "\n{:<12} {:<15} {:<10} {:<10} {:<10} {:<15} {:<15}".format(
            "Threshold", "Pos Rate", "F1", "Precision", "Recall", "Status", "Use Case"
        )
    )
    print("-" * 95)

    for r in results:
        if r.get("success"):
            threshold = r["threshold"]
            test_m = r.get("test_metrics", {})
            meta = r.get("metadata", {})

            f1 = test_m.get("f1", 0)
            precision = test_m.get("precision", 0)
            recall = test_m.get("recall", 0)

            test_pos = meta.get("test_positive_rate", 0)

            if f1 >= 0.7:
                status = "GOOD"
            elif f1 >= 0.5:
                status = "MODERATE"
            else:
                status = "POOR"

            # Recommend use case
            if threshold == 40:
                use_case = "Extreme alerts"
            elif threshold == 38:
                use_case = "General heatwave"
            elif threshold == 36:
                use_case = "Hot day warnings"
            else:  # 35
                use_case = "Moderate heat"

            print(
                "{:<12} {:<15} {:<10.4f} {:<10.4f} {:<10.4f} {:<15} {:<15}".format(
                    f"{threshold}C",
                    f"{test_pos * 100:.1f}%",
                    f1,
                    precision,
                    recall,
                    status,
                    use_case,
                )
            )
        else:
            print(
                "{:<12} {:<15}".format(
                    f"{r['threshold']}C",
                    f"FAILED: {r.get('error', 'Unknown error')[:30]}",
                )
            )

    print("\n" + "=" * 70)
    print(" RECOMMENDATIONS FOR THAILAND")
    print("=" * 70)

    # Find best model
    successful = [r for r in results if r.get("success")]
    if successful:
        best = max(successful, key=lambda x: x.get("test_metrics", {}).get("f1", 0))

        print(f"\nBest performing model:")
        print(f"  Threshold: {best['threshold']}C")
        print(f"  F1 Score: {best['test_metrics'].get('f1', 0):.4f}")
        print(
            f"  Positive Rate: {best.get('metadata', {}).get('test_positive_rate', 0) * 100:.1f}%"
        )

        print(f"\nRecommendation for Thailand:")
        print(f"  1. Main model (38C threshold): For general heatwave warnings")
        print(f"     - Detects severe heat events")
        print(f"     - Balance between precision and recall")

        print(f"\n  2. Secondary model (40C threshold): For extreme heat emergencies")
        print(f"     - Detects ONLY life-threatening heat")
        print(f"     - Use for emergency alerts")
        print(f"     - Higher precision but lower recall")

        print(f"\n  3. Alternative model (35-36C threshold): For moderate heat")
        print(f"     - More sensitive detection")
        print(f"     - Catches more heat events")
        print(f"     - More false positives")

    return successful


def main():
    """Main execution."""
    print("=" * 70)
    print(" THAILAND HEATWAVE THRESHOLD ANALYSIS")
    print("=" * 70)
    print(
        f"\nStarted: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Step 1: Analyze thresholds
    analyze_threshold_effects()

    # Step 2: Train models with different thresholds
    print("\n[+] Starting model training with different thresholds...")

    # Test thresholds: 35C, 36C, 38C (current), 40C
    thresholds = [35, 36, 40]
    results = train_with_thresholds(thresholds)

    # Step 3: Compare results
    compare_results(results)

    print("\n" + "=" * 70)
    print(" COMPLETED")
    print("=" * 70)
    print(
        f"\nFinished: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    return results


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Interrupted] Exiting...")
        sys.exit(1)
