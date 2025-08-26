# PowerShell script to setup Maintenance Scheduler (6 AM Daily Updates)
# Run this script as Administrator

Write-Host "Setting up Maintenance Scheduler for 6 AM Daily Updates..." -ForegroundColor Green

# Set variables
$TaskName = "AIHedgeFundMaintenanceScheduler"
$PythonPath = "C:\Users\Admin\AppData\Local\pypoetry\Cache\virtualenvs\ai-hedge-fund-Dot02FSf-py3.13\Scripts\python.exe"
$ScriptPath = "$PSScriptRoot\src\data\update\maintenance_scheduler.py"
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
$Action = New-ScheduledTaskAction -Execute $PythonPath -Argument "`"$ScriptPath`"" -WorkingDirectory $WorkingDir
$Trigger = New-ScheduledTaskTrigger -AtStartup
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

try {
    # Register the scheduled task
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Force
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✅ Maintenance Scheduler Setup Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "The maintenance scheduler will:" -ForegroundColor White
    Write-Host "  ✅ Start automatically at system startup" -ForegroundColor Green
    Write-Host "  ✅ Run daily updates at 6 AM IST" -ForegroundColor Green
    Write-Host "  ✅ Update price data in comprehensive_equity.duckdb" -ForegroundColor Green
    Write-Host "  ✅ Download EOD extra data (FNO, equity, indices, FII/DII)" -ForegroundColor Green
    Write-Host "  ✅ Run weekly maintenance on Sundays" -ForegroundColor Green
    Write-Host "  ✅ Run monthly cleanup on first Sunday" -ForegroundColor Green
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor White
    Write-Host "  To run manually: Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Cyan
    Write-Host "  To stop: Stop-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Cyan
    Write-Host "  To delete: Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:$false" -ForegroundColor Cyan
    Write-Host "  To view: Get-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "The maintenance scheduler will start automatically at system startup!" -ForegroundColor Green
    Write-Host "Next daily update will run at 6 AM tomorrow." -ForegroundColor Green
    
} catch {
    Write-Host ""
    Write-Host "❌ Failed to create scheduled task" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please run this script as Administrator" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
