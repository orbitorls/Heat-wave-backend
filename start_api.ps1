$proc = Start-Process python -ArgumentList "api_server.py" -WorkingDirectory "D:\Heat-wave-backend" -PassThru -NoNewWindow
Start-Sleep 20
if (-not $proc.HasExited) {
    Write-Host "Server running on PID: $($proc.Id)"
} else {
    Write-Host "Server exited with code: $($proc.ExitCode)"
}