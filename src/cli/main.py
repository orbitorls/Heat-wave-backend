import json
import os
import socket
import shutil
import subprocess
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional

import typer
import uvicorn

from src.core.config import settings
from src.core.logger import logger
from src.core.utils import detect_gpu_capability
from src.models.manager import model_manager

cli = typer.Typer(help="Thailand Heatwave Prediction System CLI")
SESSION_STATE: Dict[str, Any] = {
    "last_action": None,
    "last_error": None,
    "last_success_at": None,
}

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.table import Table

    RICH_AVAILABLE = True
    console = Console()
except Exception:
    RICH_AVAILABLE = False
    console = None


def _print(msg: str):
    if RICH_AVAILABLE:
        console.print(msg)
    else:
        print(msg)


def _mark_success(action: str):
    SESSION_STATE["last_action"] = action
    SESSION_STATE["last_error"] = None
    SESSION_STATE["last_success_at"] = time.strftime("%Y-%m-%d %H:%M:%S")


def _mark_error(action: str, err: str):
    SESSION_STATE["last_action"] = action
    SESSION_STATE["last_error"] = err


def _run_guarded(action: str, func, *args, **kwargs):
    if RICH_AVAILABLE:
        with console.status(f"[cyan]{action}...[/cyan]", spinner="dots"):
            result = func(*args, **kwargs)
    else:
        print(f":: {action} ...")
        result = func(*args, **kwargs)
    _mark_success(action)
    return result


def _header(title: str, subtitle: str = ""):
    if RICH_AVAILABLE:
        body = f"[bold cyan]{title}[/bold cyan]"
        if subtitle:
            body += f"\n{subtitle}"
        console.print(Panel(body, border_style="cyan", expand=False))
        return
    print("=" * 70)
    print(title)
    if subtitle:
        print(subtitle)
    print("=" * 70)


def _port_in_use(host: str, port: int) -> bool:
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((probe_host, int(port))) == 0


def _load_training_config(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _preflight_train(config_path: Optional[str] = None):
    data_stats = _scan_data_files(settings.DATA_DIR)
    if data_stats["total"] == 0:
        raise RuntimeError("No .nc files found in era5_data. Run download first.")
    if data_stats["groups"]["surface"] == 0 and data_stats["groups"]["upper"] == 0:
        raise RuntimeError("No standard ERA5 files found (surface/upper).")
    if config_path:
        _ = _load_training_config(config_path)


def _preflight_serve(host: str, port: int):
    if _port_in_use(host, port):
        raise RuntimeError(
            f"Port {port} is already in use on {host}. "
            "Use another port or stop the existing server."
        )


def _open_browser_after_delay(host: str, port: int):
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    target_url = f"http://{browser_host}:{port}/trainer"
    time.sleep(1.2)
    try:
        webbrowser.open(target_url, new=2)
    except Exception as exc:
        logger.warning(f"Unable to open browser automatically: {exc}")


def _scan_data_files(data_dir: Path) -> Dict[str, Any]:
    nc_files = sorted(data_dir.glob("*.nc"))
    groups = {"surface": 0, "upper": 0, "other": 0}
    examples = {"surface": [], "upper": [], "other": []}
    for path in nc_files:
        lower = path.name.lower()
        if "surface" in lower:
            key = "surface"
        elif "upper" in lower:
            key = "upper"
        else:
            key = "other"
        groups[key] += 1
        if len(examples[key]) < 5:
            examples[key].append(path.name)
    return {
        "total": len(nc_files),
        "groups": groups,
        "examples": examples,
    }


def _render_data_audit(data_dir: Path):
    stats = _scan_data_files(data_dir)
    if RICH_AVAILABLE:
        table = Table(title="ERA5 Data Audit", show_header=True, header_style="bold cyan")
        table.add_column("Group")
        table.add_column("Count", justify="right")
        table.add_column("Samples")
        for key in ("surface", "upper", "other"):
            samples = ", ".join(stats["examples"][key]) if stats["examples"][key] else "-"
            table.add_row(key, str(stats["groups"][key]), samples)
        console.print(table)
        console.print(f"Total .nc files: [bold]{stats['total']}[/bold] in [cyan]{data_dir}[/cyan]")
    else:
        print(f"ERA5 Data Audit ({data_dir})")
        for key in ("surface", "upper", "other"):
            print(f"- {key}: {stats['groups'][key]}")
        print(f"total: {stats['total']}")


def _organize_data_files(data_dir: Path, dry_run: bool = False) -> Dict[str, int]:
    archive_dir = data_dir / "_archive_misc"
    moved = 0
    kept = 0
    for path in sorted(data_dir.glob("*.nc")):
        lower = path.name.lower()
        if ("surface" in lower) or ("upper" in lower):
            kept += 1
            continue
        if dry_run:
            moved += 1
            continue
        archive_dir.mkdir(parents=True, exist_ok=True)
        target = archive_dir / path.name
        if target.exists():
            stem = path.stem
            suffix = path.suffix
            idx = 1
            while True:
                candidate = archive_dir / f"{stem}_{idx}{suffix}"
                if not candidate.exists():
                    target = candidate
                    break
                idx += 1
        shutil.move(str(path), str(target))
        moved += 1
    return {"moved": moved, "kept": kept}


def _run_training(config_path: Optional[str] = None):
    from Train_Ai import train as train_run

    config = _load_training_config(config_path)
    if config:
        _print(f"[cyan]Using training config:[/cyan] {config_path}")
    result = train_run(config=config or None)
    if not result:
        _print(
            "[yellow]Primary training attempt produced no result. "
            "Retrying with relaxed event settings...[/yellow]"
        )
        retry_cfg = dict(config or {})
        retry_cfg.setdefault("heatwave_percentile", 85.0)
        retry_cfg.setdefault("event_min_duration_days", 1)
        retry_cfg.setdefault("event_min_hot_fraction", 0.02)
        result = train_run(config=retry_cfg)
    if not result:
        raise RuntimeError(
            "Training finished but returned no results after fallback. "
            "Please verify ERA5 data quality and date coverage."
        )
    _print("[green]Training completed successfully.[/green]")
    if RICH_AVAILABLE:
        table = Table(title="Training Result", show_header=True, header_style="bold cyan")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("model_type", str(result.get("model_type")))
        table.add_row("save_path", str(result.get("save_path")))
        test_f1 = result.get("test_event_metrics", {}).get("f1")
        table.add_row("test_f1", f"{float(test_f1):.4f}" if test_f1 is not None else "n/a")
        console.print(table)
    else:
        print(result)


@cli.command()
def doctor():
    """Run environment and project diagnostics."""
    _header("Heatwave Doctor", "Runtime diagnostics")
    gpu = detect_gpu_capability()
    python_ok = bool(shutil.which("python") or shutil.which("py"))
    if RICH_AVAILABLE:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Check")
        table.add_column("Value")
        table.add_row("Project root", str(settings.PROJECT_ROOT))
        table.add_row("Data dir", str(settings.DATA_DIR))
        table.add_row("Models dir", str(settings.MODELS_DIR))
        table.add_row("Python in PATH", str(python_ok))
        table.add_row("GPU detected", str(gpu.get("gpu_detected")))
        table.add_row("torch CUDA", str(gpu.get("torch_cuda")))
        table.add_row("nvidia-smi", str(gpu.get("nvidia_smi")))
        console.print(table)
    else:
        print(f"Project root: {settings.PROJECT_ROOT}")
        print(f"Data dir: {settings.DATA_DIR}")
        print(f"Models dir: {settings.MODELS_DIR}")
        print(f"Python in PATH: {python_ok}")
        print(f"GPU detected: {gpu.get('gpu_detected')}")
    if SESSION_STATE.get("last_error"):
        _print(f"[yellow]Last error:[/yellow] {SESSION_STATE['last_error']}")


@cli.command("data-audit")
def data_audit(
    data_dir: Path = typer.Option(settings.DATA_DIR, help="ERA5 data directory")
):
    """Audit ERA5 .nc files and classification groups."""
    _header("Data Audit", str(data_dir))
    _render_data_audit(data_dir)


@cli.command("data-organize")
def data_organize(
    data_dir: Path = typer.Option(settings.DATA_DIR, help="ERA5 data directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without moving files"),
):
    """Move irregular .nc files to data_dir/_archive_misc to keep training input clean."""
    _header("Data Organize", str(data_dir))
    summary = _organize_data_files(data_dir, dry_run=dry_run)
    action = "Would move" if dry_run else "Moved"
    _print(f"[green]{action}[/green] {summary['moved']} irregular file(s). Kept {summary['kept']} standard file(s).")
    if not dry_run:
        _print(f"Archive folder: [cyan]{data_dir / '_archive_misc'}[/cyan]")


@cli.command()
def download():
    """Download ERA5 datasets."""
    _header("ERA5 Download", "Starting data acquisition")
    from download_era5 import download_era5_data

    download_era5_data()


@cli.command()
def train(
    config: Optional[str] = typer.Option(None, help="Path to training config JSON")
):
    """Run training from CLI."""
    _header("Training", "End-to-end model training")
    _run_guarded("Preflight checks", _preflight_train, config)
    _run_guarded("Training run", _run_training, config)


@cli.command()
def checkpoints(
    models_dir: Path = typer.Option(settings.MODELS_DIR, help="Checkpoint directory")
):
    """List available checkpoints."""
    files = sorted(models_dir.glob("heatwave_model_checkpoint_v*.pth"))
    if RICH_AVAILABLE:
        table = Table(title="Model Checkpoints", show_header=True, header_style="bold cyan")
        table.add_column("Name")
        table.add_column("Size (MB)", justify="right")
        table.add_column("Modified")
        for p in files:
            table.add_row(
                p.name,
                f"{p.stat().st_size / (1024 * 1024):.2f}",
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)),
            )
        if not files:
            table.add_row("-", "0", "No checkpoints found")
        console.print(table)
        return
    for p in files:
        print(p.name)
    if not files:
        print("No checkpoints found.")


@cli.command()
def serve(
    host: str = settings.API_HOST,
    port: int = settings.API_PORT,
    reload: bool = True,
    open_browser: bool = typer.Option(
        True,
        "--open-browser/--no-open-browser",
        help="Open trainer UI in the default browser after server starts.",
    ),
):
    """Start the FastAPI server."""
    logger.info(f"Starting API server on {host}:{port}")
    _run_guarded("Serve preflight", _preflight_serve, host, port)
    if open_browser:
        threading.Thread(
            target=_open_browser_after_delay, args=(host, port), daemon=True
        ).start()
    uvicorn.run("src.api.main:app", host=host, port=port, reload=reload)


@cli.command()
def predict(
    input_file: str = typer.Option(..., help="Path to input NetCDF file")
):
    """Run local prediction on a file using loaded model."""
    _header("Predict", input_file)
    model_ok = model_manager.load_model()
    if not model_ok:
        raise typer.BadParameter("No loadable checkpoint found in models directory.")
    _print("[green]Model loaded.[/green]")
    _print(
        "Prediction pipeline wiring for file-output formats is pending. "
        "Use /trainer web UI for current inference workflow."
    )


@cli.command()
def pipeline(
    download_data: bool = typer.Option(False, "--download-data/--skip-download-data"),
    organize_data: bool = typer.Option(True, "--organize-data/--skip-organize-data"),
    train_model: bool = typer.Option(True, "--train-model/--skip-train-model"),
    config: Optional[str] = typer.Option(None, help="Training config JSON"),
    serve_api: bool = typer.Option(True, "--serve-api/--skip-serve-api"),
    host: str = settings.API_HOST,
    port: int = settings.API_PORT,
):
    """Run full process: data -> train -> serve."""
    _header("Heatwave Pipeline", "Data to live web app")
    if download_data:
        _print("[cyan]Step 1:[/cyan] Download ERA5 data")
        from download_era5 import download_era5_data

        download_era5_data()
    if organize_data:
        _print("[cyan]Step 2:[/cyan] Organize ERA5 files")
        summary = _run_guarded(
            "Organizing ERA5 files",
            _organize_data_files,
            settings.DATA_DIR,
            False,
        )
        _print(f"Moved {summary['moved']} irregular file(s).")
    _print("[cyan]Step 3:[/cyan] Data audit")
    _run_guarded("Auditing dataset", _render_data_audit, settings.DATA_DIR)
    if train_model:
        _print("[cyan]Step 4:[/cyan] Training")
        try:
            _run_guarded("Preflight checks", _preflight_train, config)
            _run_guarded("Training run", _run_training, config)
        except Exception as exc:
            _print(f"[red]Pipeline stopped at training:[/red] {exc}")
            _mark_error("pipeline:training", str(exc))
            raise typer.Exit(code=1)
    _print("[cyan]Step 5:[/cyan] Checkpoints")
    checkpoints(models_dir=settings.MODELS_DIR)
    if serve_api:
        _print("[cyan]Step 6:[/cyan] Start API + Trainer UI")
        serve(host=host, port=port, reload=True, open_browser=True)


def _resolve_action_token(raw: str) -> str:
    token = (raw or "").strip().lower()
    aliases = {
        "1": "doctor",
        "2": "audit",
        "3": "organize",
        "4": "train",
        "5": "checkpoints",
        "6": "pipeline",
        "7": "serve",
        "0": "exit",
        "q": "exit",
        "quit": "exit",
        "exit": "exit",
        "h": "help",
        "?": "help",
        "help": "help",
        "doctor": "doctor",
        "audit": "audit",
        "data": "audit",
        "organize": "organize",
        "train": "train",
        "ckpt": "checkpoints",
        "checkpoints": "checkpoints",
        "pipeline": "pipeline",
        "serve": "serve",
    }
    return aliases.get(token, "unknown")


def _render_interactive_menu():
    if RICH_AVAILABLE:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Key", width=6)
        table.add_column("Action")
        table.add_row("1", "System doctor")
        table.add_row("2", "Data audit")
        table.add_row("3", "Organize data files")
        table.add_row("4", "Train model")
        table.add_row("5", "List checkpoints")
        table.add_row("6", "Run full pipeline")
        table.add_row("7", "Start API + open web")
        table.add_row("h", "Help / command palette")
        table.add_row("0", "Exit")
        console.print(table)
        if SESSION_STATE.get("last_error"):
            console.print(f"[bold yellow]Last error:[/bold yellow] {SESSION_STATE['last_error']}")
        if SESSION_STATE.get("last_success_at"):
            console.print(f"[green]Last success:[/green] {SESSION_STATE['last_success_at']}")
    else:
        print("1) Doctor 2) Data Audit 3) Organize 4) Train 5) Checkpoints 6) Pipeline 7) Serve h) Help 0) Exit")


def _show_help_panel():
    _header("Command Palette", "You can type number or command keyword")
    _print("Examples: `1`, `doctor`, `train`, `pipeline`, `serve`, `help`, `quit`")


@cli.command()
def interactive():
    """Interactive control center for full workflow."""
    _header("Heatwave Interactive Console", "End-to-end operations control")
    while True:
        _render_interactive_menu()
        if RICH_AVAILABLE:
            choice = Prompt.ask("Select action", default="1")
        else:
            choice = input("Select action: ").strip()
        action = _resolve_action_token(choice)

        try:
            if action == "doctor":
                _run_guarded("Doctor", doctor)
            elif action == "audit":
                _run_guarded("Data audit", data_audit, settings.DATA_DIR)
            elif action == "organize":
                do_move = True
                if RICH_AVAILABLE:
                    do_move = Confirm.ask("Move irregular .nc files to archive?", default=True)
                if do_move:
                    _run_guarded("Data organize", data_organize, settings.DATA_DIR, False)
                else:
                    _run_guarded("Data organize (dry-run)", data_organize, settings.DATA_DIR, True)
            elif action == "train":
                cfg = None
                if RICH_AVAILABLE:
                    cfg_in = Prompt.ask("Training config path (Enter to skip)", default="")
                    cfg = cfg_in.strip() or None
                _run_guarded("Train", train, cfg)
            elif action == "checkpoints":
                _run_guarded("List checkpoints", checkpoints, settings.MODELS_DIR)
            elif action == "pipeline":
                _run_guarded(
                    "Run full pipeline",
                    pipeline,
                    download_data=False,
                    organize_data=True,
                    train_model=True,
                    config=None,
                    serve_api=False,
                    host=settings.API_HOST,
                    port=settings.API_PORT,
                )
            elif action == "serve":
                _run_guarded(
                    "Start API server",
                    serve,
                    host=settings.API_HOST,
                    port=settings.API_PORT,
                    reload=True,
                    open_browser=True,
                )
            elif action == "help":
                _show_help_panel()
            elif action == "exit":
                _print("[yellow]Exiting interactive console.[/yellow]")
                break
            else:
                _print("[yellow]Unknown command. Type `h` for help.[/yellow]")
        except typer.Exit:
            _print("[yellow]Returned to interactive menu.[/yellow]")
        except Exception as exc:
            _mark_error(f"interactive:{action}", str(exc))
            _print(f"[red]Action failed:[/red] {exc}")


@cli.command()
def control():
    """Compatibility alias to interactive control center."""
    interactive()


@cli.command()
def trainer():
    """Compatibility alias to interactive control center."""
    interactive()


@cli.command()
def quick():
    """Compatibility alias to interactive control center."""
    interactive()


@cli.command()
def studio():
    """Compatibility alias to interactive control center."""
    interactive()


if __name__ == "__main__":
    cli()
