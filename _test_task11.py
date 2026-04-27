"""Task 11: End-to-End API Verification.

Starts the Flask server, tests all 4 prediction endpoints,
verifies no dummy values remain, saves response evidence.
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:5000"
EVIDENCE_DIR = os.path.join(".sisyphus", "evidence")
STARTUP_TIMEOUT = 90  # seconds
POLL_INTERVAL = 3  # seconds

results = {}
failures = []


def fetch_json(path: str, timeout: int = 30) -> dict:
    """GET a JSON endpoint and return parsed dict."""
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def save_evidence(name: str, data):
    """Save evidence to .sisyphus/evidence/task-11-{name}.json."""
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    path = os.path.join(EVIDENCE_DIR, f"task-11-{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [SAVED] {path}")


def assert_check(condition: bool, description: str):
    """Record an assertion result."""
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {description}")
    if not condition:
        failures.append(description)


def wait_for_server(proc: subprocess.Popen) -> bool:
    """Poll /api/health until model_loaded=True or timeout."""
    start = time.time()
    last_status = None
    while time.time() - start < STARTUP_TIMEOUT:
        if proc.poll() is not None:
            print(f"[ERROR] Server process exited with code {proc.returncode}")
            return False
        try:
            data = fetch_json("/api/health", timeout=5)
            model_loaded = data.get("model_loaded", False)
            status = data.get("status", "unknown")
            if last_status != (status, model_loaded):
                print(f"  Health: status={status}, model_loaded={model_loaded} ({time.time()-start:.0f}s)")
                last_status = (status, model_loaded)
            if model_loaded:
                return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            elapsed = time.time() - start
            if int(elapsed) % 10 == 0:
                print(f"  Waiting for server... ({elapsed:.0f}s)")
        time.sleep(POLL_INTERVAL)
    print(f"[ERROR] Server did not become ready within {STARTUP_TIMEOUT}s")
    return False


def test_health():
    """Test /api/health endpoint."""
    print("\n=== Testing /api/health ===")
    data = fetch_json("/api/health")
    results["health"] = data
    save_evidence("health", data)

    assert_check(data.get("status") == "ok", "status == 'ok'")
    assert_check(data.get("model_loaded") is True, "model_loaded == True")


def test_predict():
    """Test /api/predict endpoint."""
    print("\n=== Testing /api/predict ===")
    data = fetch_json("/api/predict")
    results["predict"] = data
    save_evidence("predict", data)

    # Date must not be dummy "2000-XX-XX"
    date_str = data.get("date", "")
    assert_check("2000" not in date_str, f"date does not contain '2000' (got: {date_str})")
    assert_check(date_str.startswith("202"), f"date starts with '202' (got: {date_str})")

    # Temperature in 20-60°C range
    weather = data.get("weather", {})
    t2m_max = weather.get("T2M_MAX")
    assert_check(
        t2m_max is not None and 20 <= t2m_max <= 60,
        f"T2M_MAX in 20-60°C range (got: {t2m_max})"
    )

    # Probability not in static set
    prob = data.get("probability")
    static_probs = {0.1, 0.5, 0.8, 0.95}
    assert_check(
        prob is not None and prob not in static_probs,
        f"probability not in static set {{0.1, 0.5, 0.8, 0.95}} (got: {prob})"
    )

    # RH2M and WS10M should be None
    assert_check(weather.get("RH2M") is None, f"RH2M is None (got: {weather.get('RH2M')})")
    assert_check(weather.get("WS10M") is None, f"WS10M is None (got: {weather.get('WS10M')})")


def test_forecast():
    """Test /api/forecast endpoint."""
    print("\n=== Testing /api/forecast ===")
    data = fetch_json("/api/forecast")
    results["forecast"] = data
    save_evidence("forecast", data)

    forecasts = data.get("forecasts", [])
    assert_check(len(forecasts) == 7, f"7 forecasts returned (got: {len(forecasts)})")

    if forecasts:
        day0 = forecasts[0]
        weather0 = day0.get("weather", {})
        assert_check(
            weather0.get("RH2M") is None,
            f"forecast day 0 RH2M is None (got: {weather0.get('RH2M')})"
        )
        assert_check(
            weather0.get("WS10M") is None,
            f"forecast day 0 WS10M is None (got: {weather0.get('WS10M')})"
        )

        # Check all days have reasonable temps
        for i, day in enumerate(forecasts):
            w = day.get("weather", {})
            t_max = w.get("T2M_MAX")
            if t_max is not None:
                assert_check(
                    20 <= t_max <= 60,
                    f"forecast day {i} T2M_MAX in 20-60°C (got: {t_max:.2f})"
                )


def test_map():
    """Test /api/map endpoint."""
    print("\n=== Testing /api/map ===")
    data = fetch_json("/api/map")
    results["map"] = data
    save_evidence("map", data)

    assert_check(
        data.get("type") == "FeatureCollection",
        f"GeoJSON type == FeatureCollection (got: {data.get('type')})"
    )

    features = data.get("features", [])
    assert_check(len(features) > 0, f"features count > 0 (got: {len(features)})")

    if features:
        temps = []
        for f in features:
            props = f.get("properties", {})
            t = props.get("temperature") or props.get("temp") or props.get("T2M_MAX")
            if t is not None:
                temps.append(float(t))

        if temps:
            min_t = min(temps)
            max_t = max(temps)
            assert_check(
                min_t >= -10 and max_t <= 70,
                f"temp range reasonable: [{min_t:.1f}, {max_t:.1f}]°C"
            )
            assert_check(
                max_t - min_t > 0.1,
                f"temp range has variance (spread: {max_t - min_t:.2f}°C)"
            )
        else:
            print("  [WARN] No temperature values found in features")


def build_comparison():
    """Compare with baseline from task-1 if available."""
    print("\n=== Building Comparison ===")
    baseline_path = os.path.join(EVIDENCE_DIR, "task-1-api-dummy-values.txt")
    comparison_lines = []
    comparison_lines.append("=" * 70)
    comparison_lines.append("Task 11: End-to-End API Verification — Comparison with Task 1 Baseline")
    comparison_lines.append("=" * 70)
    comparison_lines.append("")

    # Load baseline
    if os.path.exists(baseline_path):
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_text = f.read()
        comparison_lines.append("--- Task 1 Baseline (dummy values found) ---")
        comparison_lines.append(baseline_text.strip())
        comparison_lines.append("")
    else:
        comparison_lines.append("[No task-1 baseline file found]")
        comparison_lines.append("")

    # Current results
    comparison_lines.append("--- Task 11 Current Results ---")

    predict = results.get("predict", {})
    comparison_lines.append(f"  date:        {predict.get('date')}  (was: '2000-XX-XX')")
    weather = predict.get("weather", {})
    comparison_lines.append(f"  T2M_MAX:     {weather.get('T2M_MAX')}  (was: fake inference)")
    comparison_lines.append(f"  RH2M:        {weather.get('RH2M')}  (was: 60)")
    comparison_lines.append(f"  WS10M:       {weather.get('WS10M')}  (was: 2.5)")
    comparison_lines.append(f"  probability: {predict.get('probability')}  (was: static set)")
    comparison_lines.append(f"  model_type:  {predict.get('model_type')}")

    forecast = results.get("forecast", {})
    forecasts = forecast.get("forecasts", [])
    if forecasts:
        day0 = forecasts[0]
        w0 = day0.get("weather", {})
        comparison_lines.append(f"\n  Forecast day 0:")
        comparison_lines.append(f"    RH2M:       {w0.get('RH2M')}  (was: 60)")
        comparison_lines.append(f"    WS10M:      {w0.get('WS10M')}  (was: 2.5)")
        comparison_lines.append(f"    NDVI:       {w0.get('NDVI')}  (was: 0.5)")

    map_data = results.get("map", {})
    comparison_lines.append(f"\n  Map features: {len(map_data.get('features', []))}")

    comparison_lines.append("")
    comparison_lines.append("--- Verdict ---")
    if failures:
        comparison_lines.append(f"FAIL: {len(failures)} assertion(s) failed:")
        for f in failures:
            comparison_lines.append(f"  - {f}")
    else:
        comparison_lines.append("ALL PASS: No dummy values remain. All endpoints return real data.")

    comparison_text = "\n".join(comparison_lines)
    comp_path = os.path.join(EVIDENCE_DIR, "task-11-comparison.txt")
    with open(comp_path, "w", encoding="utf-8") as f:
        f.write(comparison_text)
    print(f"  [SAVED] {comp_path}")
    print()
    print(comparison_text)


def main():
    print("=" * 60)
    print("Task 11: End-to-End API Verification")
    print("=" * 60)

    # Start server as subprocess
    print("\n[1] Starting Flask server...")
    proc = subprocess.Popen(
        [sys.executable, "api_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    print(f"  Server PID: {proc.pid}")

    try:
        # Wait for server to load model
        print("\n[2] Waiting for server startup...")
        if not wait_for_server(proc):
            # Try to capture server output for debugging
            proc.terminate()
            try:
                stdout, _ = proc.communicate(timeout=5)
                if stdout:
                    print("\n--- Server Output ---")
                    print(stdout.decode("utf-8", errors="replace")[-2000:])
            except Exception:
                pass
            print("\n[FATAL] Server failed to start. Exiting.")
            sys.exit(1)

        # Run tests
        print("\n[3] Running endpoint tests...")
        test_health()
        test_predict()
        test_forecast()
        test_map()

        # Build comparison
        print("\n[4] Building comparison...")
        build_comparison()

        # Summary
        print("\n" + "=" * 60)
        if failures:
            print(f"RESULT: FAIL — {len(failures)} assertion(s) failed")
            for f in failures:
                print(f"  ✗ {f}")
            sys.exit(1)
        else:
            print("RESULT: ALL PASS — All endpoints verified, no dummy values")
            sys.exit(0)

    finally:
        # Kill server
        print(f"\n[5] Terminating server (PID {proc.pid})...")
        try:
            proc.terminate()
            proc.wait(timeout=10)
            print("  Server terminated.")
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=5)
                print("  Server killed.")
            except Exception as e:
                print(f"  Warning: Could not kill server: {e}")


if __name__ == "__main__":
    main()
