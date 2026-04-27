import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request


RESET = "\033[0m"
COLORS = {
    "green": "\033[92m",
    "cyan": "\033[96m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "bold": "\033[1m",
}

ACCENT = "cyan"
WARNING = "yellow"
SUCCESS = "green"
ERROR = "red"


def supports_color(no_color=False):
    if no_color or os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def colorize(text, color, no_color=False):
    if not supports_color(no_color):
        return text
    return f"{COLORS.get(color, '')}{text}{RESET}"


def clear_screen(no_color=False):
    if _rich_available() and not no_color:
        from rich.console import Console

        Console().clear()
        return
    os.system("cls" if os.name == "nt" else "clear")


def banner(no_color=False):
    if _rich_available() and not no_color:
        from rich.console import Console
        from rich.padding import Padding
        from rich.panel import Panel
        from rich.text import Text

        text = Text()
        text.append("Heatwave CLI", style="bold white")
        text.append("  ", style="white")
        text.append("Thailand Forecast Ops\n", style="bold cyan")
        text.append("train | api | diagnostics | checkpoints", style="white")
        Console().print(
            Padding(
                Panel(
                    text,
                    border_style="cyan",
                    expand=False,
                    title="[bold cyan]Control Surface[/bold cyan]",
                ),
                (0, 0, 1, 0),
            )
        )
        return
    title = colorize("Heatwave Backend CLI", "bold", no_color)
    line = colorize("=" * 56, "cyan", no_color)
    print(line)
    print(title)
    print(
        colorize(
            "Unified console: train, serve, forecast, data, diagnostics",
            "cyan",
            no_color,
        )
    )
    print(line)


def http_get_json(base_url, path, timeout):
    url = urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def print_json(data, no_color=False):
    print(colorize(json.dumps(data, indent=2, ensure_ascii=False), "green", no_color))


def print_panel(title, lines, no_color=False, width=72):
    if _rich_available() and not no_color:
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        table = Table(show_header=False, box=box.SIMPLE, expand=True, pad_edge=False)
        table.add_column("content", style="white")
        for line in lines:
            table.add_row(str(line))
        Console().print(
            Panel(
                table,
                title=f"[bold cyan]{title}[/bold cyan]",
                border_style="cyan",
                expand=False,
            )
        )
        return

    width = max(width, len(title) + 6)
    top = f"+{'-' * (width - 2)}+"
    print(colorize(top, "cyan", no_color))
    heading = f"| {title}"
    print(colorize(f"{heading:<{width - 1}}|", "cyan", no_color))
    print(colorize(top, "cyan", no_color))
    for line in lines:
        wrapped = textwrap.wrap(line, width=width - 4) or [""]
        for seg in wrapped:
            print(f"| {seg:<{width - 4}} |")
    print(colorize(top, "cyan", no_color))


def _rich_available():
    try:
        import rich  # noqa: F401

        return True
    except ImportError:
        return False


def detect_gpu_capability():
    torch_cuda = False
    nvidia_smi = False
    detail = []

    try:
        import torch

        torch_cuda = bool(torch.cuda.is_available())
        detail.append(f"torch.cuda={torch_cuda}")
    except Exception:
        detail.append("torch=unavailable")

    commands = []
    found = shutil.which("nvidia-smi")
    if found:
        commands.append(found)
    commands.append("nvidia-smi")
    commands.append(r"C:\Windows\System32\nvidia-smi.exe")
    commands.append(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")

    for cmd in commands:
        try:
            proc = subprocess.run(
                [cmd, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=4,
                check=False,
            )
            if proc.returncode == 0 and bool(proc.stdout.strip()):
                nvidia_smi = True
                break
        except Exception:
            continue
    detail.append(f"nvidia_smi={nvidia_smi}")

    gpu_detected = torch_cuda or nvidia_smi
    return {
        "gpu_detected": gpu_detected,
        "torch_cuda": torch_cuda,
        "nvidia_smi": nvidia_smi,
        "detail": ", ".join(detail),
    }


def _print_kv(title, rows, no_color=False):
    if not _rich_available():
        print_panel(title, [f"{k}: {v}" for k, v in rows], no_color)
        return
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    table = Table(show_header=False, box=None, expand=True)
    table.add_column("key", style="bold cyan", no_wrap=True)
    table.add_column("value", style="white")
    for key, value in rows:
        table.add_row(str(key), str(value))
    Console().print(Panel(table, title=title, border_style="cyan"))


def _print_json_pretty(data, no_color=False):
    if not _rich_available():
        print_json(data, no_color)
        return
    from rich.console import Console
    from rich.json import JSON

    Console().print(JSON.from_data(data))


def _print_header_card(no_color=False):
    gpu = detect_gpu_capability()
    gpu_status = (
        "GPU Ready"
        if gpu["gpu_detected"]
        else "GPU Not Detected"
    )

    if _rich_available() and not no_color:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        text = Text()
        text.append("Heatwave Control Center\n", style="bold cyan")
        text.append("Interactive AI operations console\n", style="white")
        text.append(gpu_status, style="green" if gpu["gpu_detected"] else "yellow")
        Console().print(Panel(text, border_style="cyan", expand=False))
        return

    print_panel(
        "Heatwave Control",
        [
            "Interactive AI operations console",
            "Train, serve, forecast, checkpoints, diagnostics",
            gpu_status,
        ],
        no_color,
    )


def cmd_health(args):
    data = http_get_json(args.base_url, "/api/health", args.timeout)
    _print_kv(
        "API Health",
        [
            ("status", data.get("status", "unknown")),
            ("model_loaded", data.get("model_loaded", False)),
            ("base_url", args.base_url),
        ],
        args.no_color,
    )
    if args.json:
        _print_json_pretty(data, args.no_color)


def cmd_predict(args):
    data = http_get_json(args.base_url, "/api/predict", args.timeout)
    if args.json:
        _print_json_pretty(data, args.no_color)
        return

    weather = data.get("weather", {})
    _print_kv(
        "Prediction Summary",
        [
            ("risk", data.get("risk_level", "N/A")),
            ("probability", data.get("probability", "N/A")),
            (
                "t2m_max_min",
                f"{weather.get('T2M_MAX', 'N/A')} / {weather.get('T2M_MIN', 'N/A')} C",
            ),
            ("advice", data.get("advice", "N/A")),
        ],
        args.no_color,
    )


def cmd_forecast(args):
    data = http_get_json(args.base_url, "/api/forecast", args.timeout)
    forecasts = data.get("forecasts", [])
    if args.json:
        _print_json_pretty(data, args.no_color)
        return

    if _rich_available():
        from rich.console import Console
        from rich.table import Table

        table = Table(title="7-Day Forecast", show_lines=False)
        table.add_column("Day", style="cyan", no_wrap=True)
        table.add_column("Date")
        table.add_column("Risk")
        table.add_column("Tmax")
        table.add_column("Tmin")
        for day in forecasts:
            weather = day.get("weather", {})
            table.add_row(
                str(day.get("day", "?")),
                str(day.get("date", "N/A")),
                str(day.get("risk_level", "N/A")),
                str(weather.get("T2M_MAX", "N/A")),
                str(weather.get("T2M_MIN", "N/A")),
            )
        Console().print(table)
    else:
        print(colorize("7-Day Forecast", "bold", args.no_color))
        for day in forecasts:
            weather = day.get("weather", {})
            print(
                f"Day {day.get('day', '?')}: {day.get('date', 'N/A')} "
                f"[{day.get('risk_level', 'N/A')}] "
                f"Tmax={weather.get('T2M_MAX', 'N/A')} Tmin={weather.get('T2M_MIN', 'N/A')}"
            )


def cmd_map(args):
    data = http_get_json(args.base_url, "/api/map", args.timeout)
    features = data.get("features", [])
    _print_kv("Map Summary", [("feature_count", len(features))], args.no_color)
    if args.sample > 0:
        sample = features[: args.sample]
        summary = [
            {
                "temperature": feat.get("properties", {}).get("temperature"),
                "risk_level": feat.get("properties", {}).get("risk_level"),
            }
            for feat in sample
        ]
        _print_json_pretty(summary, args.no_color)


def cmd_serve(args):
    from api_server import app, load_resources

    if not load_resources():
        print(
            colorize(
                "Model resources not loaded yet. Starting server anyway for web trainer.",
                "yellow",
                args.no_color,
            )
        )
    print(
        colorize(
            f"Serving API on http://{args.host}:{args.port}", "green", args.no_color
        )
    )
    app.run(host=args.host, port=args.port, debug=False)


def cmd_web(args):
    import threading
    import time as _time
    import webbrowser

    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 5000)
    web_url = f"http://{host}:{port}/trainer"

    def _open_browser():
        _time.sleep(1.2)
        try:
            webbrowser.open(web_url, new=1)
        except Exception:
            print(
                colorize(
                    f"Could not auto-open browser. Open manually: {web_url}",
                    "yellow",
                    args.no_color,
                )
            )

    threading.Thread(target=_open_browser, daemon=True).start()
    print(colorize(f"Launching web trainer at {web_url}", "cyan", args.no_color))
    cmd_serve(args)


def cmd_train(args):
    config = {
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "future_seq": args.future_seq,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "model_backend": "balanced_rf",
        "use_gpu": False,
        "force_gpu": False,
        "rf_n_estimators": args.rf_n_estimators,
        "rf_max_depth": args.rf_max_depth,
        "rf_min_samples_leaf": args.rf_min_samples_leaf,
        "rf_sampling_strategy": args.rf_sampling_strategy,
        "rf_replacement": args.rf_replacement,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "heatwave_percentile": args.heatwave_percentile,
    }
    print(colorize("Starting training backend=balanced_rf (CPU)...", "cyan", args.no_color))
    run_training_session(config, args.no_color)


def cmd_download(args):
    no_color = getattr(args, "no_color", False)
    print(colorize("Starting ERA5 download...", "cyan", no_color))
    print(colorize("Note: Requires CDS API credentials (~/.cdsapirc).", "yellow", no_color))
    try:
        from download_era5 import download_era5_data
        download_era5_data()
        print(colorize("ERA5 download complete.", "green", no_color))
    except ImportError:
        print(colorize("download_era5.py not found.", "red", no_color))
    except Exception as exc:
        msg = str(exc)
        if "cdsapirc" in msg.lower() or "credentials" in msg.lower() or "key" in msg.lower() or "token" in msg.lower() or "authorization" in msg.lower():
            print(colorize("CDS API credentials not configured.", "red", no_color))
            print(colorize("Create ~/.cdsapirc with your CDS API key:", "yellow", no_color))
            print(colorize("  url: https://cds.climate.copernicus.eu/api/v2", "cyan", no_color))
            print(colorize("  key: <YOUR-UID>:<YOUR-API-KEY>", "cyan", no_color))
            print(colorize("Get your key at: https://cds.climate.copernicus.eu/user/login", "cyan", no_color))
        else:
            print(colorize(f"Download failed: {exc}", "red", no_color))

def cmd_checkpoints(args):
    models_dir = Path(args.models_dir)
    files = sorted(
        list(models_dir.glob("heatwave_model_checkpoint_v*.pth"))
        + list(models_dir.glob("heatwave_convlstm_v*.pth")),
        key=lambda f: f.stat().st_mtime,
    )
    if not files:
        _print_kv(
            "Checkpoints",
            [("models_dir", str(models_dir)), ("status", "no checkpoints found")],
            args.no_color,
        )
        return

    if _rich_available():
        from rich.console import Console
        from rich.table import Table

        table = Table(title=f"Checkpoints ({models_dir})")
        table.add_column("Version", style="cyan")
        table.add_column("File")
        table.add_column("Size (MB)", justify="right")
        table.add_column("Modified")
        for file in files:
            try:
                ver = file.stem.split("_v")[-1]
            except Exception:
                ver = "?"
            stat = file.stat()
            size_mb = stat.st_size / 1024 / 1024
            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            table.add_row(ver, file.name, f"{size_mb:.2f}", modified)
        Console().print(table)
    else:
        lines = [f"{f.name}" for f in files]
        print_panel("Checkpoints", lines, args.no_color)


def cmd_system(args):
    gpu = detect_gpu_capability()
    device = "cuda" if gpu["gpu_detected"] else "cpu"
    try:
        import prompt_toolkit  # noqa: F401

        pt_status = "available"
    except ImportError:
        pt_status = "not installed"
    rows = [
        ("python", sys.version.split()[0]),
        ("platform", sys.platform),
        ("cwd", str(Path.cwd())),
        ("device", device),
        ("gpu_detected", gpu["gpu_detected"]),
        ("torch_cuda", gpu["torch_cuda"]),
        ("nvidia_smi", gpu["nvidia_smi"]),
        ("gpu_detail", gpu["detail"]),
        ("training_backend", "balanced_rf"),
        ("rich", "available" if _rich_available() else "not installed"),
        ("prompt_toolkit", pt_status),
    ]
    _print_kv("System", rows, args.no_color)


def run_training_session(config, no_color=False):
    from Train_Ai import train

    try:
        from rich.console import Console, Group
        from rich.live import Live
        from rich.panel import Panel
        from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
        from rich.table import Table

        console = Console()
        latest = {
            "epoch": 0,
            "total": int(config["epochs"]),
            "train": 0.0,
            "val": 0.0,
            "elapsed": 0.0,
        }
        progress = Progress(
            TextColumn("[bold cyan]Training"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )
        task_id = progress.add_task("epochs", total=int(config["epochs"]))

        def render():
            table = Table(title="Heatwave Trainer", expand=True)
            table.add_column("Epoch")
            table.add_column("Train Loss")
            table.add_column("Val Loss")
            table.add_column("Elapsed")
            table.add_row(
                f"{latest['epoch']}/{latest['total']}",
                f"{latest['train']:.6f}",
                f"{latest['val']:.6f}",
                f"{latest['elapsed']:.1f}s",
            )
            return Panel(
                Group(table, progress), title="Training Session", border_style="cyan"
            )

        def on_epoch_end(metrics):
            latest["epoch"] = metrics.epoch
            latest["total"] = metrics.total_epochs
            latest["train"] = metrics.train_loss
            latest["val"] = metrics.val_loss
            latest["elapsed"] = metrics.elapsed_seconds
            progress.update(task_id, completed=metrics.epoch)

        with progress:
            with Live(render(), console=console, refresh_per_second=8):
                result = train(config=config, on_epoch_end=on_epoch_end)
                console.print(render())
        if not result:
            console.print("[bold red]Training failed.[/bold red] See logs above.")
            return
        console.print(
            f"[bold green]Done:[/bold green] saved checkpoint to {result['save_path']}"
        )
    except ImportError:
        print(colorize("Running training (basic mode)...", "cyan", no_color))
        result = train(config=config)
        if not result:
            print(colorize("Training failed. See logs above.", "red", no_color))
            return
        print(colorize(f"Done: {result['save_path']}", "green", no_color))


def trainer_status(config, no_color=False):
    if _rich_available() and not no_color:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        table = Table(title="Trainer Console", expand=True)
        table.add_column("Parameter", style="bold cyan", no_wrap=True)
        table.add_column("Value", style="white")
        table.add_row("batch_size", str(config["batch_size"]))
        table.add_row("seq_len", str(config["seq_len"]))
        table.add_row("future_seq", str(config["future_seq"]))
        table.add_row("epochs", str(config["epochs"]))
        table.add_row("learning_rate", str(config["learning_rate"]))
        table.add_row("model_backend", str(config["model_backend"]))
        table.add_row("use_gpu", str(config["use_gpu"]))
        table.add_row("force_gpu", str(config["force_gpu"]))
        table.add_row("rf_n_estimators", str(config["rf_n_estimators"]))
        table.add_row("rf_max_depth", str(config["rf_max_depth"]))
        table.add_row("rf_min_samples_leaf", str(config["rf_min_samples_leaf"]))
        table.add_row("rf_sampling_strategy", str(config["rf_sampling_strategy"]))
        table.add_row("rf_replacement", str(config["rf_replacement"]))
        table.add_row("train_ratio", str(config["train_ratio"]))
        table.add_row("val_ratio", str(config["val_ratio"]))
        table.add_row("heatwave_percentile", str(config["heatwave_percentile"]))
        table.add_row(
            "Commands", "run | set <key> <value> | gpu | auto | reset | help | exit"
        )
        Console().print(Panel(table, border_style="cyan"))
        return

    print_panel(
        "Trainer Console",
        [
            f"batch_size   : {config['batch_size']}",
            f"seq_len      : {config['seq_len']}",
            f"future_seq   : {config['future_seq']}",
            f"epochs       : {config['epochs']}",
            f"learning_rate: {config['learning_rate']}",
            f"model_backend: {config['model_backend']}",
            f"use_gpu      : {config['use_gpu']}",
            f"force_gpu    : {config['force_gpu']}",
            f"rf_trees     : {config['rf_n_estimators']}",
            f"rf_depth     : {config['rf_max_depth']}",
            f"rf_leaf      : {config['rf_min_samples_leaf']}",
            f"rf_sampling  : {config['rf_sampling_strategy']}",
            f"rf_replace   : {config['rf_replacement']}",
            f"train_ratio  : {config['train_ratio']}",
            f"val_ratio    : {config['val_ratio']}",
            f"hw_percentile: {config['heatwave_percentile']}",
            "Commands: run | set <key> <value> | gpu | auto | reset | help | exit",
        ],
        no_color,
    )


def trainer_help(no_color=False):
    if _rich_available() and not no_color:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        table = Table(title="Trainer Commands", expand=True)
        table.add_column("Command", style="bold cyan", no_wrap=True)
        table.add_column("Description", style="white")
        table.add_row("run", "Start training now")
        table.add_row("set batch_size <int>", "Set batch size")
        table.add_row("set seq_len <int>", "Set input sequence length")
        table.add_row("set future_seq <int>", "Set forecast horizon for training")
        table.add_row("set epochs <int>", "Set training epochs")
        table.add_row("set learning_rate <float>", "Set optimizer learning rate")
        table.add_row("set model_backend <balanced_rf>", "Training backend (fixed)")
        table.add_row("set use_gpu <true|false>", "Toggle GPU path")
        table.add_row("set force_gpu <true|false>", "Bypass GPU detection checks")
        table.add_row("set rf_n_estimators <int>", "Set number of trees")
        table.add_row("set rf_max_depth <int>", "Set tree depth")
        table.add_row("set train_ratio <float>", "Set train split ratio")
        table.add_row("set val_ratio <float>", "Set validation split ratio")
        table.add_row("backend", "Show backend (fixed)")
        table.add_row("gpu", "Quick toggle gpu true/false")
        table.add_row("auto", "Apply recommended Balanced RF config")
        table.add_row("status", "Show active config")
        table.add_row("reset", "Reset to default config")
        table.add_row("exit", "Quit trainer console")
        Console().print(Panel(table, border_style="cyan"))
        return

    print_panel(
        "Trainer Commands",
        [
            "run                         Start training now",
            "set batch_size <int>        Set batch size",
            "set seq_len <int>           Set input sequence length",
            "set future_seq <int>        Set forecast horizon for training",
            "set epochs <int>            Set training epochs",
            "set learning_rate <float>   Set optimizer learning rate",
            "set model_backend <...>     balanced_rf only",
            "set use_gpu <true|false>    Toggle GPU mode",
            "set force_gpu <true|false>  Bypass GPU detection checks",
            "set rf_n_estimators <int>   Number of trees",
            "set rf_max_depth <int>      Tree depth",
            "set train_ratio <float>     Train split ratio",
            "set val_ratio <float>       Validation split ratio",
            "backend                     Show backend (fixed)",
            "gpu                         Quick toggle gpu mode",
            "auto                        Apply recommended Balanced RF config",
            "status                      Show active config",
            "reset                       Reset to default config",
            "exit                        Quit trainer console",
        ],
        no_color,
    )


def cmd_trainer(args):
    defaults = get_default_training_config()
    config = {
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "future_seq": args.future_seq,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "model_backend": "balanced_rf",
        "use_gpu": False,
        "force_gpu": False,
        "rf_n_estimators": args.rf_n_estimators,
        "rf_max_depth": args.rf_max_depth,
        "rf_min_samples_leaf": args.rf_min_samples_leaf,
        "rf_sampling_strategy": args.rf_sampling_strategy,
        "rf_replacement": args.rf_replacement,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "heatwave_percentile": args.heatwave_percentile,
    }

    import os

    try:
        from prompt_toolkit.shortcuts import (
            input_dialog,
            message_dialog,
            radiolist_dialog,
        )
        from prompt_toolkit.styles import Style

        dialog_style = Style.from_dict(
            {
                "dialog": "bg:#1e1e2e",
                "dialog frame.label": "bg:#00a3a3 #ffffff",
                "dialog.body": "bg:#1e1e2e #ffffff",
                "dialog shadow": "bg:#101018",
                "button": "bg:#2d3f5f #ffffff",
                "button.arrow": "bg:#2d3f5f #ffffff",
                "button.focused": "bg:#00a3a3 #000000 bold",
                "radio": "#9ad1ff",
                "radio-selected": "#00ffaa bold",
            }
        )

        def current_config_lines():
            return [
                f"batch_size: {config['batch_size']}",
                f"seq_len: {config['seq_len']}",
                f"future_seq: {config['future_seq']}",
                f"epochs: {config['epochs']}",
                f"learning_rate: {config['learning_rate']}",
                f"model_backend: {config['model_backend']}",
                f"use_gpu: {config['use_gpu']}",
                f"force_gpu: {config['force_gpu']}",
                f"rf_n_estimators: {config['rf_n_estimators']}",
                f"rf_max_depth: {config['rf_max_depth']}",
                f"rf_min_samples_leaf: {config['rf_min_samples_leaf']}",
                f"rf_sampling_strategy: {config['rf_sampling_strategy']}",
                f"rf_replacement: {config['rf_replacement']}",
                f"train_ratio: {config['train_ratio']}",
                f"val_ratio: {config['val_ratio']}",
                f"heatwave_percentile: {config['heatwave_percentile']}",
            ]

        def apply_auto():
            cpu_count = os.cpu_count() or 4
            config["batch_size"] = 8 if cpu_count >= 8 else 4
            config["epochs"] = max(int(config["epochs"]), 30)
            config["learning_rate"] = 0.001
            config["model_backend"] = "balanced_rf"
            config["use_gpu"] = False
            config["force_gpu"] = False
            config["rf_n_estimators"] = max(int(config["rf_n_estimators"]), 400)
            config["rf_max_depth"] = 10

        while True:
            choice = radiolist_dialog(
                title="Heatwave Trainer",
                text=(
                    "Use arrow keys to choose action and press Enter\n"
                    f"Current: backend={config['model_backend']} gpu={config['use_gpu']} "
                    f"batch={config['batch_size']} epochs={config['epochs']}"
                ),
                values=[
                    ("run", "Run training now"),
                    ("edit", "Edit config"),
                    ("gpu", "Toggle GPU true/false"),
                    ("auto", "Apply recommended config"),
                    ("show", "Show current config"),
                    ("reset", "Reset to defaults"),
                    ("exit", "Exit trainer"),
                ],
                style=dialog_style,
            ).run()

            if choice in (None, "exit"):
                break
            if choice == "run":
                run_training_session(config, args.no_color)
                continue
            if choice == "show":
                message_dialog(
                    title="Current Config",
                    text="\n".join(current_config_lines()),
                    style=dialog_style,
                ).run()
                continue
            if choice == "reset":
                config = defaults.copy()
                message_dialog(
                    title="Trainer",
                    text="Config reset to defaults.",
                    style=dialog_style,
                ).run()
                continue
            if choice == "auto":
                apply_auto()
                message_dialog(
                    title="Trainer",
                    text="Applied recommended config.",
                    style=dialog_style,
                ).run()
                continue
            if choice == "backend":
                message_dialog(
                    title="Trainer",
                    text="model_backend is fixed to balanced_rf",
                    style=dialog_style,
                ).run()
                continue
            if choice == "gpu":
                config["use_gpu"] = not bool(config["use_gpu"])
                message_dialog(
                    title="Trainer",
                    text=f"use_gpu -> {config['use_gpu']}",
                    style=dialog_style,
                ).run()
                continue
            if choice == "edit":
                key = radiolist_dialog(
                    title="Edit Config",
                    text="Select field to edit",
                    values=[
                        ("batch_size", f"batch_size ({config['batch_size']})"),
                        ("seq_len", f"seq_len ({config['seq_len']})"),
                        ("future_seq", f"future_seq ({config['future_seq']})"),
                        ("epochs", f"epochs ({config['epochs']})"),
                        ("learning_rate", f"learning_rate ({config['learning_rate']})"),
                        ("model_backend", f"model_backend ({config['model_backend']}, fixed)"),
                        ("use_gpu", f"use_gpu ({config['use_gpu']})"),
                        ("force_gpu", f"force_gpu ({config['force_gpu']})"),
                        ("rf_n_estimators", f"rf_n_estimators ({config['rf_n_estimators']})"),
                        ("rf_max_depth", f"rf_max_depth ({config['rf_max_depth']})"),
                        ("rf_min_samples_leaf", f"rf_min_samples_leaf ({config['rf_min_samples_leaf']})"),
                        ("rf_sampling_strategy", f"rf_sampling_strategy ({config['rf_sampling_strategy']})"),
                        ("rf_replacement", f"rf_replacement ({config['rf_replacement']})"),
                        ("train_ratio", f"train_ratio ({config['train_ratio']})"),
                        ("val_ratio", f"val_ratio ({config['val_ratio']})"),
                        ("heatwave_percentile", f"heatwave_percentile ({config['heatwave_percentile']})"),
                    ],
                    style=dialog_style,
                ).run()
                if key is None:
                    continue
                raw_value = input_dialog(
                    title="Edit Config",
                    text=f"Enter new value for {key}:",
                    default=str(config[key]),
                    style=dialog_style,
                ).run()
                if raw_value is None:
                    continue
                try:
                    if key == "model_backend":
                        config[key] = "balanced_rf"
                    elif key == "rf_sampling_strategy":
                        config[key] = str(raw_value).strip().lower()
                    elif key in {"use_gpu", "force_gpu", "rf_replacement"}:
                        config[key] = parse_bool(raw_value, bool(config[key]))
                    elif key in {
                        "learning_rate",
                        "train_ratio",
                        "val_ratio",
                        "heatwave_percentile",
                    }:
                        parsed = float(raw_value)
                        if parsed <= 0:
                            raise ValueError
                        config[key] = parsed
                    else:
                        parsed = int(raw_value)
                        if parsed <= 0:
                            raise ValueError
                        config[key] = parsed
                    message_dialog(
                        title="Edit Config",
                        text=f"Updated {key} -> {config[key]}",
                        style=dialog_style,
                    ).run()
                except ValueError:
                    message_dialog(
                        title="Invalid Value",
                        text="Please enter a positive number.",
                        style=dialog_style,
                    ).run()

        print(colorize("Exiting Trainer Console.", "cyan", args.no_color))
        return
    except ImportError:
        pass
    except Exception:
        print(
            colorize(
                "Interactive trainer is unavailable in this terminal, using basic mode.",
                "yellow",
                args.no_color,
            )
        )

    trainer_status(config, args.no_color)
    while True:
        try:
            user_input = input(colorize("trainer> ", "bold", args.no_color)).strip()
        except KeyboardInterrupt:
            print()
            continue
        except EOFError:
            print()
            break

        if not user_input:
            continue

        parts = shlex.split(user_input)
        cmd = parts[0].lower()

        if cmd in {"exit", "quit", "q"}:
            break
        if cmd in {"help", "?"}:
            trainer_help(args.no_color)
            continue
        if cmd in {"status", "show"}:
            trainer_status(config, args.no_color)
            continue
        if cmd == "reset":
            config = defaults.copy()
            print(colorize("Trainer config reset to defaults.", "green", args.no_color))
            continue
        if cmd == "run":
            run_training_session(config, args.no_color)
            continue
        if cmd == "auto":
            cpu_count = os.cpu_count() or 4
            recommended_batch = 8 if cpu_count >= 8 else 4
            config["batch_size"] = recommended_batch
            config["epochs"] = max(config["epochs"], 30)
            config["learning_rate"] = 0.001
            config["model_backend"] = "balanced_rf"
            config["use_gpu"] = False
            config["force_gpu"] = False
            config["rf_n_estimators"] = max(config["rf_n_estimators"], 400)
            config["rf_max_depth"] = 10
            print(
                colorize(
                    "Applied recommended config (Balanced RF CPU).",
                    "green",
                    args.no_color,
                )
            )
            continue
        if cmd == "backend":
            config["model_backend"] = "balanced_rf"
            print(colorize("model_backend is fixed to balanced_rf", "green", args.no_color))
            continue
        if cmd == "gpu":
            config["use_gpu"] = not bool(config["use_gpu"])
            print(
                colorize(
                    f"use_gpu -> {config['use_gpu']}",
                    "green",
                    args.no_color,
                )
            )
            continue
        if cmd == "set":
            if len(parts) < 3:
                print(colorize("Usage: set <key> <value>", "yellow", args.no_color))
                continue
            key = parts[1]
            value = parts[2]
            if key not in config:
                print(colorize(f"Unknown key: {key}", "yellow", args.no_color))
                continue
            try:
                if key == "model_backend":
                    config[key] = "balanced_rf"
                elif key == "rf_sampling_strategy":
                    config[key] = str(value).strip().lower()
                elif key in {"use_gpu", "force_gpu", "rf_replacement"}:
                    config[key] = parse_bool(value, bool(config[key]))
                elif key in {
                    "learning_rate",
                    "train_ratio",
                    "val_ratio",
                    "heatwave_percentile",
                }:
                    parsed = float(value)
                    if parsed <= 0:
                        raise ValueError
                    config[key] = parsed
                else:
                    parsed = int(value)
                    if parsed <= 0:
                        raise ValueError
                    config[key] = parsed
                print(
                    colorize(f"Updated {key} -> {config[key]}", "green", args.no_color)
                )
            except ValueError:
                print(
                    colorize(
                        "Invalid value. Must be positive number.",
                        "red",
                        args.no_color,
                    )
                )
            continue

        print(
            colorize(
                "Unknown command. Type 'help' for commands.", "yellow", args.no_color
            )
        )

    print(colorize("Exiting Trainer Console.", "cyan", args.no_color))


def studio_help(no_color=False):
    if _rich_available() and not no_color:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        table = Table(title="Studio Commands", expand=True)
        table.add_column("Key", style="bold cyan", no_wrap=True)
        table.add_column("Command", style="white")
        table.add_row("1 / health", "Check backend health quickly")
        table.add_row("2 / predict", "Show dashboard prediction summary")
        table.add_row("3 / forecast", "Show 7-day forecast")
        table.add_row("4 / map [N]", "Show map cells + N sample entries")
        table.add_row("5 / serve", "Start backend server")
        table.add_row("6 / train", "Start model training")
        table.add_row("7 / download", "Download ERA5 data")
        table.add_row("8 / trainer", "Open training-focused console")
        table.add_row("9 / checkpoints", "List saved model checkpoints")
        table.add_row("10 / system", "Show diagnostics")
        table.add_row("set url <URL>", "Change API base URL")
        table.add_row("set timeout <sec>", "Change request timeout")
        table.add_row("status", "Show current studio config")
        table.add_row("clear", "Clear terminal")
        table.add_row("exit", "Quit studio")
        Console().print(Panel(table, border_style="cyan"))
        return

    print_panel(
        "Studio Commands",
        [
            "1 / health          Check backend health quickly",
            "2 / predict         Show dashboard prediction summary",
            "3 / forecast        Show 7-day forecast",
            "4 / map [N]         Show map cells + N sample entries",
            "5 / serve           Start backend server",
            "6 / train           Start model training",
            "7 / download        Download ERA5 data",
            "8 / trainer         Open training-focused console",
            "9 / checkpoints     List saved model checkpoints",
            "10 / system         Show diagnostics",
            "set url <URL>       Change API base URL",
            "set timeout <sec>   Change request timeout",
            "status              Show current studio config",
            "clear               Clear terminal",
            "exit                Quit studio",
        ],
        no_color,
    )


def studio_status(base_url, timeout, no_color=False):
    if _rich_available() and not no_color:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        text = Text()
        text.append(f"API Base URL : {base_url}\n", style="white")
        text.append(f"Timeout      : {timeout:.1f} seconds\n", style="white")
        text.append("Tip          : Type 'help' to show command palette", style="cyan")
        Console().print(
            Panel(text, title="Studio Status", border_style="cyan", expand=False)
        )
        return

    print_panel(
        "Studio Status",
        [
            f"API Base URL : {base_url}",
            f"Timeout      : {timeout:.1f} seconds",
            "Tip          : Type 'help' to show command palette",
        ],
        no_color,
    )


def parse_positive_int(value, fallback):
    try:
        parsed = int(value)
        return parsed if parsed > 0 else fallback
    except ValueError:
        return fallback


def parse_bool(value, fallback=False):
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return fallback


def get_default_training_config():
    return {
        "batch_size": 4,
        "seq_len": 5,
        "future_seq": 2,
        "epochs": 20,
        "learning_rate": 1e-3,
        "model_backend": "balanced_rf",
        "use_gpu": False,
        "force_gpu": False,
        "rf_n_estimators": 300,
        "rf_max_depth": 10,
        "rf_min_samples_leaf": 2,
        "rf_sampling_strategy": "all",
        "rf_replacement": True,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "heatwave_percentile": 95.0,
    }


def get_latest_checkpoint_summary(models_dir="models"):
    p = Path(models_dir)
    files = list(p.glob("heatwave_model_checkpoint_v*.pth")) + list(p.glob("heatwave_convlstm_v*.pth"))
    if not files:
        return {
            "version": "none",
            "name": "no checkpoint",
            "modified": "-",
            "size_mb": "-",
        }

    latest = max(files, key=lambda file: file.stat().st_mtime)
    stat = latest.stat()
    return {
        "version": latest.stem.split("_v")[-1],
        "name": latest.name,
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        "size_mb": f"{stat.st_size / 1024 / 1024:.2f}",
    }


def get_runtime_snapshot(cfg, gpu):
    return [
        ("GPU", "ready" if gpu["gpu_detected"] else "not detected"),
        ("Backend", str(cfg["model_backend"])),
        ("GPU Mode", "on" if cfg["use_gpu"] else "off"),
        ("Force GPU", "on" if cfg["force_gpu"] else "off"),
        ("Trees", str(cfg["rf_n_estimators"])),
        ("Depth", str(cfg["rf_max_depth"])),
        ("Epochs", str(cfg["epochs"])),
        ("Detail", gpu["detail"]),
    ]


def get_action_rows():
    return [
        ("1", "Train now"),
        ("2", "Show backend"),
        ("3", "Toggle GPU"),
        ("4", "Balanced RF preset"),
        ("5", "Toggle force GPU"),
        ("6", "Advanced trainer"),
        ("7", "Launch web trainer"),
        ("s", "Start API server"),
        ("d", "Download ERA5 data"),
        ("8", "API health check"),
        ("9", "System diagnostics"),
        ("0", "Exit"),
    ]


def render_quick_dashboard(cfg, gpu, no_color=False):
    if not _rich_available() or no_color:
        gpu_line = "GPU: ready" if gpu["gpu_detected"] else "GPU: not detected"
        latest = get_latest_checkpoint_summary()
        print_panel(
            "Heatwave Dashboard",
            [
                f"{gpu_line} ({gpu['detail']})",
                (
                    f"Train backend={cfg['model_backend']} gpu={cfg['use_gpu']} "
                    f"force_gpu={cfg['force_gpu']} trees={cfg['rf_n_estimators']}"
                ),
                f"Latest checkpoint={latest['name']} modified={latest['modified']}",
                "Note: Training backend is fixed to balanced_rf (CPU).",
                "1) Train now",
                "2) Show backend",
                "3) Toggle GPU on/off",
                "4) Balanced RF preset",
                "5) Toggle Force GPU bypass",
                "6) Open advanced trainer",
                "7) Launch web trainer",
                "s) Start API server",
                "d) Download ERA5 data",
                "8) API health check",
                "9) System diagnostics",
                "0) Exit",
            ],
            no_color,
        )
        return

    from rich import box
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()
    status = Table(show_header=False, box=box.SIMPLE, expand=True)
    status.add_column("k", style="cyan", no_wrap=True)
    status.add_column("v", style="white")
    for key, value in get_runtime_snapshot(cfg, gpu):
        status.add_row(key, value)

    actions = Table(show_header=True, box=box.SIMPLE, expand=True)
    actions.add_column("Key", style="bold cyan", no_wrap=True)
    actions.add_column("Action", style="white")
    for key, label in get_action_rows():
        actions.add_row(key, label)

    latest = get_latest_checkpoint_summary()
    checkpoint = Table(show_header=False, box=box.SIMPLE, expand=True)
    checkpoint.add_column("k", style="cyan", no_wrap=True)
    checkpoint.add_column("v", style="white")
    checkpoint.add_row("Version", str(latest["version"]))
    checkpoint.add_row("File", latest["name"])
    checkpoint.add_row("Modified", latest["modified"])
    checkpoint.add_row("Size MB", latest["size_mb"])

    note_text = Text()
    note_text.append("Training backend is fixed to ", style="white")
    note_text.append("balanced_rf", style="bold yellow")
    note_text.append(" for consistency.", style="white")

    headline = Panel(
        Text.from_markup(
            "[bold white]Heatwave Operations Dashboard[/bold white]\n"
            "[cyan]Fast training control, GPU routing, API actions, and checkpoint visibility[/cyan]"
        ),
        border_style="cyan",
        expand=False,
    )
    note = Panel(
        note_text,
        border_style="yellow",
        title="[bold yellow]Training Note[/bold yellow]",
        expand=False,
    )
    console.print(headline)
    console.print(
        Columns(
            [
                Panel(
                    status,
                    title="[bold cyan]Runtime Status[/bold cyan]",
                    border_style="cyan",
                ),
                Panel(
                    actions,
                    title="[bold cyan]Quick Actions[/bold cyan]",
                    border_style="cyan",
                ),
                Panel(
                    checkpoint,
                    title="[bold cyan]Latest Checkpoint[/bold cyan]",
                    border_style="cyan",
                ),
            ],
            equal=True,
        )
    )
    console.print(note)


def run_studio_command(command, base_url, timeout, no_color=False):
    tokens = shlex.split(command)
    if not tokens:
        return base_url, timeout, False

    cmd = tokens[0].lower()
    if cmd in {"exit", "quit", "q"}:
        return base_url, timeout, True

    if cmd in {"help", "?"}:
        studio_help(no_color)
        return base_url, timeout, False

    if cmd in {"status", "st"}:
        studio_status(base_url, timeout, no_color)
        return base_url, timeout, False

    if cmd == "clear":
        os.system("cls" if os.name == "nt" else "clear")
        return base_url, timeout, False

    if cmd == "set":
        if len(tokens) < 3:
            print(
                colorize(
                    "Usage: set url <URL> | set timeout <seconds>", "yellow", no_color
                )
            )
            return base_url, timeout, False
        key = tokens[1].lower()
        value = " ".join(tokens[2:]).strip()
        if key == "url":
            base_url = value
            print(colorize(f"Updated base URL -> {base_url}", "green", no_color))
        elif key == "timeout":
            try:
                timeout = float(value)
                print(colorize(f"Updated timeout -> {timeout:.1f}s", "green", no_color))
            except ValueError:
                print(colorize("Timeout must be a number", "red", no_color))
        else:
            print(
                colorize("Unknown setting. Use 'url' or 'timeout'.", "yellow", no_color)
            )
        return base_url, timeout, False

    action_payload = {
        "base_url": base_url,
        "timeout": timeout,
        "json": False,
        "sample": 3,
        "host": "0.0.0.0",
        "port": 5000,
        "no_color": no_color,
    }
    action_payload.update(get_default_training_config())
    action_args = argparse.Namespace(**action_payload)

    try:
        if cmd in {"1", "health"}:
            cmd_health(action_args)
        elif cmd in {"2", "predict"}:
            cmd_predict(action_args)
        elif cmd in {"3", "forecast"}:
            cmd_forecast(action_args)
        elif cmd in {"4", "map"}:
            if len(tokens) > 1:
                action_args.sample = parse_positive_int(tokens[1], 3)
            cmd_map(action_args)
        elif cmd in {"5", "serve"}:
            cmd_serve(action_args)
        elif cmd in {"6", "train"}:
            cmd_train(action_args)
        elif cmd in {"7", "download"}:
            cmd_download(action_args)
        elif cmd in {"8", "trainer"}:
            trainer_args = argparse.Namespace(
                **get_default_training_config(),
                no_color=no_color,
            )
            cmd_trainer(trainer_args)
        elif cmd in {"9", "checkpoints"}:
            action_args.models_dir = "models"
            cmd_checkpoints(action_args)
        elif cmd in {"10", "system"}:
            cmd_system(action_args)
        else:
            print(colorize(f"Unknown command: {cmd}", "yellow", no_color))
            print(
                colorize("Type 'help' to see available commands.", "yellow", no_color)
            )
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        print(colorize(f"HTTP {exc.code}: {payload}", "red", no_color))
    except urllib.error.URLError as exc:
        print(colorize(f"Connection failed: {exc.reason}", "red", no_color))
    except KeyboardInterrupt:
        print(colorize("Interrupted.", "yellow", no_color))
    except Exception as exc:
        print(colorize(f"Error: {exc}", "red", no_color))

    return base_url, timeout, False


def cmd_studio(args):
    return cmd_quick(args)


def cmd_control(args):
    try:
        from prompt_toolkit.shortcuts import radiolist_dialog
        from prompt_toolkit.styles import Style
        import os

        # Check if we're in a compatible terminal
        if os.name == "nt":
            try:
                # Test if prompt_toolkit can work in this terminal
                from prompt_toolkit.shortcuts import message_dialog

                message_dialog(
                    title="Test", text="Testing terminal compatibility"
                ).run()
            except Exception:
                # If interactive dialogs fail, fall back to studio
                print(
                    colorize(
                        "Interactive control center is unavailable in this terminal, using quick mode.",
                        "yellow",
                        args.no_color,
                    )
                )
                return cmd_quick(args)

        dialog_style = Style.from_dict(
            {
                "dialog": "bg:#10121a",
                "dialog frame.label": "bg:#00a3a3 #ffffff",
                "dialog.body": "bg:#10121a #ffffff",
                "dialog shadow": "bg:#090b11",
                "button": "bg:#253147 #ffffff",
                "button.arrow": "bg:#253147 #ffffff",
                "button.focused": "bg:#00a3a3 #000000 bold",
                "radio": "#9ad1ff",
                "radio-selected": "#00ffaa bold",
            }
        )

        _print_header_card(args.no_color)

        while True:
            choice = radiolist_dialog(
                title="Heatwave Control Center",
                text=(
                    "Use arrow keys and Enter to manage every system\n"
                    f"Target API: {getattr(args, 'base_url', 'http://127.0.0.1:5000')}"
                ),
                values=[
                    ("health", "API: Health"),
                    ("predict", "API: Predict summary"),
                    ("forecast", "API: 7-day forecast"),
                    ("map", "API: Map summary"),
                    ("web", "Web: Launch trainer in browser"),
                    ("serve", "Server: Start API"),
                    ("trainer", "Training: Interactive trainer (BRF-only)"),
                    ("train", "Training: Run now with current defaults"),
                    ("download", "Data: Download ERA5"),
                    ("checkpoints", "Model: Checkpoints"),
                    ("system", "Diagnostics: System info"),
                    ("exit", "Exit"),
                ],
                style=dialog_style,
            ).run()

            if choice in (None, "exit"):
                break

            action_payload = {
                "base_url": getattr(args, "base_url", "http://127.0.0.1:5000"),
                "timeout": getattr(args, "timeout", 30.0),
                "json": False,
                "sample": 3,
                "host": "0.0.0.0",
                "port": 5000,
                "no_color": args.no_color,
                "models_dir": "models",
            }
            action_payload.update(get_default_training_config())
            action_args = argparse.Namespace(**action_payload)

            if choice == "health":
                cmd_health(action_args)
            elif choice == "predict":
                cmd_predict(action_args)
            elif choice == "forecast":
                cmd_forecast(action_args)
            elif choice == "map":
                cmd_map(action_args)
            elif choice == "serve":
                cmd_serve(action_args)
            elif choice == "web":
                web_args = argparse.Namespace(
                    host="127.0.0.1",
                    port=action_args.port,
                    no_color=args.no_color,
                )
                cmd_web(web_args)
            elif choice == "trainer":
                cmd_trainer(action_args)
            elif choice == "train":
                cmd_train(action_args)
            elif choice == "download":
                cmd_download(action_args)
            elif choice == "checkpoints":
                cmd_checkpoints(action_args)
            elif choice == "system":
                cmd_system(action_args)
        print(colorize("Goodbye from Heatwave Control Center.", "cyan", args.no_color))
        return
    except ImportError:
        pass
    except Exception as e:
        print(colorize(f"Error initializing control center: {e}", "red", args.no_color))
        print(colorize("Falling back to quick mode...", "yellow", args.no_color))
        return cmd_quick(args)

    # Fallback to quick mode if prompt_toolkit not available
    return cmd_quick(args)


def cmd_quick(args):
    cfg = get_default_training_config()
    base_url = getattr(args, "base_url", "http://127.0.0.1:5000")
    timeout = getattr(args, "timeout", 30.0)

    while True:
        clear_screen(args.no_color)
        gpu = detect_gpu_capability()
        render_quick_dashboard(cfg, gpu, args.no_color)
        if _rich_available() and not args.no_color:
            from rich.prompt import Prompt

            choice = Prompt.ask(
                "[bold cyan]Select action[/bold cyan]",
                choices=[row[0] for row in get_action_rows()],
                default="1",
            ).strip().lower()
        else:
            choice = input(
                colorize("Select action [0-9/s/d] > ", "bold", args.no_color)
            ).strip().lower()

        if choice in {"0", "exit", "quit", "q"}:
            break
        if choice == "1":
            run_training_session(cfg, args.no_color)
            continue
        if choice == "2":
            cfg["model_backend"] = "balanced_rf"
            print(colorize("backend is fixed -> balanced_rf", "green", args.no_color))
            continue
        if choice == "3":
            cfg["use_gpu"] = not bool(cfg["use_gpu"])
            print(colorize(f"use_gpu -> {cfg['use_gpu']}", "green", args.no_color))
            continue
        if choice == "4":
            cfg["model_backend"] = "balanced_rf"
            cfg["use_gpu"] = False
            cfg["force_gpu"] = False
            cfg["rf_n_estimators"] = 400
            cfg["rf_max_depth"] = 10
            cfg["epochs"] = max(int(cfg["epochs"]), 20)
            print(colorize("Applied Balanced RF preset.", "green", args.no_color))
            continue
        if choice == "5":
            cfg["force_gpu"] = not bool(cfg["force_gpu"])
            print(colorize(f"force_gpu -> {cfg['force_gpu']}", "green", args.no_color))
            continue
        if choice == "6":
            trainer_args = argparse.Namespace(**cfg, no_color=args.no_color)
            cmd_trainer(trainer_args)
            continue
        if choice == "7":
            cmd_web(
                argparse.Namespace(
                    host="127.0.0.1",
                    port=5000,
                    no_color=args.no_color,
                )
            )
            continue
        if choice in {"s", "serve"}:
            cmd_serve(
                argparse.Namespace(
                    host="0.0.0.0",
                    port=5000,
                    no_color=args.no_color,
                )
            )
            continue
        if choice in {"d", "download"}:
            cmd_download(argparse.Namespace(no_color=args.no_color))
            if _rich_available() and not args.no_color:
                from rich.prompt import Prompt
                Prompt.ask("[bold cyan]Press Enter to return[/bold cyan]", default="")
            else:
                input(colorize("Press Enter to return > ", "bold", args.no_color))
            continue
        if choice == "8":
            cmd_health(
                argparse.Namespace(
                    base_url=base_url,
                    timeout=timeout,
                    json=False,
                    no_color=args.no_color,
                )
            )
            continue
        if choice == "9":
            cmd_system(argparse.Namespace(no_color=args.no_color))
            if _rich_available() and not args.no_color:
                from rich.prompt import Prompt

                Prompt.ask("[bold cyan]Press Enter to return[/bold cyan]", default="")
            else:
                input(colorize("Press Enter to return > ", "bold", args.no_color))
            continue
        print(colorize("Invalid selection", "yellow", args.no_color))


def add_training_args(parser):
    defaults = get_default_training_config()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults["batch_size"],
        help="Training batch size",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=defaults["seq_len"],
        help="Input sequence length",
    )
    parser.add_argument(
        "--future-seq",
        type=int,
        default=defaults["future_seq"],
        help="Future sequence horizon",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=defaults["epochs"],
        help="Training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=defaults["learning_rate"],
        help="Training learning rate",
    )
    parser.add_argument(
        "--model-backend",
        default=defaults["model_backend"],
        choices=["balanced_rf"],
        help="Training backend",
    )
    parser.add_argument(
        "--use-gpu",
        dest="use_gpu",
        action="store_true",
        help="Enable GPU mode",
    )
    parser.add_argument(
        "--no-use-gpu",
        dest="use_gpu",
        action="store_false",
        help="Disable GPU mode",
    )
    parser.set_defaults(use_gpu=defaults["use_gpu"])
    parser.add_argument(
        "--force-gpu",
        dest="force_gpu",
        action="store_true",
        help="Bypass GPU detection checks",
    )
    parser.add_argument(
        "--no-force-gpu",
        dest="force_gpu",
        action="store_false",
        help="Disable GPU bypass",
    )
    parser.set_defaults(force_gpu=defaults["force_gpu"])
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=defaults["rf_n_estimators"],
        help="Number of trees",
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=defaults["rf_max_depth"],
        help="Maximum tree depth",
    )
    parser.add_argument(
        "--rf-min-samples-leaf",
        type=int,
        default=defaults["rf_min_samples_leaf"],
        help="Minimum samples per leaf",
    )
    parser.add_argument(
        "--rf-sampling-strategy",
        default=defaults["rf_sampling_strategy"],
        help="Sampling strategy for balanced RF",
    )
    parser.add_argument(
        "--rf-replacement",
        dest="rf_replacement",
        action="store_true",
        help="Use replacement in balanced RF",
    )
    parser.add_argument(
        "--no-rf-replacement",
        dest="rf_replacement",
        action="store_false",
        help="Disable replacement in balanced RF",
    )
    parser.set_defaults(rf_replacement=defaults["rf_replacement"])
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=defaults["train_ratio"],
        help="Train split ratio",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=defaults["val_ratio"],
        help="Validation split ratio",
    )
    parser.add_argument(
        "--heatwave-percentile",
        type=float,
        default=defaults["heatwave_percentile"],
        help="Heatwave percentile threshold",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="heatwave-cli",
        description="Command-line toolkit for Heatwave AI backend",
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    p_health = subparsers.add_parser("health", help="Check backend health")
    p_health.add_argument(
        "--base-url", default="http://127.0.0.1:5000", help="API base URL"
    )
    p_health.add_argument(
        "--timeout", type=float, default=20.0, help="HTTP timeout in seconds"
    )
    p_health.add_argument(
        "--json", action="store_true", help="Print full JSON response"
    )
    p_health.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )
    p_health.set_defaults(func=cmd_health)

    p_predict = subparsers.add_parser("predict", help="Get dashboard-style prediction")
    p_predict.add_argument(
        "--base-url", default="http://127.0.0.1:5000", help="API base URL"
    )
    p_predict.add_argument(
        "--timeout", type=float, default=30.0, help="HTTP timeout in seconds"
    )
    p_predict.add_argument(
        "--json", action="store_true", help="Print full JSON response"
    )
    p_predict.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )
    p_predict.set_defaults(func=cmd_predict)

    p_forecast = subparsers.add_parser("forecast", help="Get 7-day forecast")
    p_forecast.add_argument(
        "--base-url", default="http://127.0.0.1:5000", help="API base URL"
    )
    p_forecast.add_argument(
        "--timeout", type=float, default=30.0, help="HTTP timeout in seconds"
    )
    p_forecast.add_argument(
        "--json", action="store_true", help="Print full JSON response"
    )
    p_forecast.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )
    p_forecast.set_defaults(func=cmd_forecast)

    p_map = subparsers.add_parser("map", help="Get map feature summary")
    p_map.add_argument(
        "--base-url", default="http://127.0.0.1:5000", help="API base URL"
    )
    p_map.add_argument(
        "--timeout", type=float, default=60.0, help="HTTP timeout in seconds"
    )
    p_map.add_argument(
        "--sample", type=int, default=3, help="Number of sample cells to print"
    )
    p_map.add_argument("--no-color", action="store_true", help="Disable colored output")
    p_map.set_defaults(func=cmd_map)

    p_serve = subparsers.add_parser("serve", help="Start Flask API server")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind host")
    p_serve.add_argument("--port", type=int, default=5000, help="Bind port")
    p_serve.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )
    p_serve.set_defaults(func=cmd_serve)

    p_web = subparsers.add_parser("web", help="Launch web trainer in browser")
    p_web.add_argument("--host", default="127.0.0.1", help="Bind host")
    p_web.add_argument("--port", type=int, default=5000, help="Bind port")
    p_web.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )
    p_web.set_defaults(func=cmd_web)

    p_train = subparsers.add_parser("train", help="Run model training")
    add_training_args(p_train)
    p_train.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )
    p_train.set_defaults(func=cmd_train)

    p_download = subparsers.add_parser("download", help="Download ERA5 datasets")
    p_download.add_argument("--no-color", action="store_true", help="Disable colored output")
    p_download.set_defaults(func=cmd_download)

    p_checkpoints = subparsers.add_parser(
        "checkpoints", help="List available model checkpoints"
    )
    p_checkpoints.add_argument(
        "--models-dir", default="models", help="Checkpoint directory"
    )
    p_checkpoints.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )
    p_checkpoints.set_defaults(func=cmd_checkpoints)

    p_system = subparsers.add_parser("system", help="Show runtime/system diagnostics")
    p_system.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )
    p_system.set_defaults(func=cmd_system)

    p_trainer = subparsers.add_parser(
        "trainer", help="Interactive training-focused console"
    )
    add_training_args(p_trainer)
    p_trainer.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )
    p_trainer.set_defaults(func=cmd_trainer)

    p_quick = subparsers.add_parser("quick", help="Simple numbered quick menu")
    p_quick.add_argument(
        "--base-url", default="http://127.0.0.1:5000", help="API base URL"
    )
    p_quick.add_argument(
        "--timeout", type=float, default=30.0, help="HTTP timeout in seconds"
    )
    p_quick.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )
    p_quick.set_defaults(func=cmd_quick)

    p_control = subparsers.add_parser(
        "control", help="Arrow-key control center for all systems"
    )
    p_control.add_argument(
        "--base-url", default="http://127.0.0.1:5000", help="API base URL"
    )
    p_control.add_argument(
        "--timeout", type=float, default=30.0, help="HTTP timeout in seconds"
    )
    p_control.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )
    p_control.set_defaults(func=cmd_control)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    banner(args.no_color)
    if args.command is None:
        args.command = "quick"
        args.base_url = "http://127.0.0.1:5000"
        args.timeout = 30.0
        args.func = cmd_quick
    try:
        args.func(args)
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        print(colorize(f"HTTP {exc.code}: {payload}", "red", args.no_color))
        raise SystemExit(1)
    except urllib.error.URLError as exc:
        print(colorize(f"Connection failed: {exc.reason}", "red", args.no_color))
        raise SystemExit(1)
    except Exception as exc:
        print(colorize(f"Error: {exc}", "red", args.no_color))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
