# PowerShell script to setup Auto Options Analyzer Scheduler
# Run this script as Administrator

Write-Host "Setting up Auto Options Analyzer Scheduler..." -ForegroundColor Green

# Set variables
$TaskName = "AIHedgeFundOptionsAnalyzer"
$PythonPath = "C:\Users\Admin\AppData\Local\pypoetry\Cache\virtualenvs\ai-hedge-fund-Dot02FSf-py3.13\Scripts\python.exe"
$ScriptPath = "$PSScriptRoot\junk\auto_market_hours_scheduler.py"
$WorkingDir = $PSScriptRoot

Write-Host "Task Name: $TaskName" -ForegroundColor Yellow
Write-Host "Python: $PythonPath" -ForegroundColor Yellow
Write-Host "Script: $ScriptPath" -ForegroundColor Yellow
Write-Host "Working Dir: $WorkingDir" -ForegroundColor Yellow

# Check if Python path exists
if (-not (Test-Path $PythonPath)) {
    Write-Host "❌ Python path not found: $PythonPath" -ForegroundColor Red
    exit 1
}

# Check if script path exists
if (-not (Test-Path $ScriptPath)) {
    Write-Host "❌ Script path not found: $ScriptPath" -ForegroundColor Red
    exit 1
}

# Create the scheduled task
$Action = New-ScheduledTaskAction -Execute $PythonPath -Argument "`"$ScriptPath`" --mode auto" -WorkingDirectory $WorkingDir
$Trigger = New-ScheduledTaskTrigger -AtStartup
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

try {
    # Register the scheduled task
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Force
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✅ Auto Options Analyzer Setup Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "The scheduler will:" -ForegroundColor White
    Write-Host "  ✅ Start automatically at system startup" -ForegroundColor Green
    Write-Host "  ✅ Run during market hours (9:30 AM - 3:30 PM IST)" -ForegroundColor Green
    Write-Host "  ✅ Skip weekends and trading holidays" -ForegroundColor Green
    Write-Host "  ✅ Run analysis every 15 minutes" -ForegroundColor Green
    Write-Host "  ✅ Save results to results/options_tracker/option_tracker.csv" -ForegroundColor Green
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor White
    Write-Host "  To run manually: Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Cyan
    Write-Host "  To stop: Stop-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Cyan
    Write-Host "  To delete: Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:$false" -ForegroundColor Cyan
    Write-Host "  To view: Get-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "The scheduler will start automatically tomorrow at 9:30 AM!" -ForegroundColor Green
    
} catch {
    Write-Host ""
    Write-Host "❌ Failed to create scheduled task" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please run this script as Administrator" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
