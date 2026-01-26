# Setup automatic backup task for ByAca Obsidian notes
# This script creates a Windows scheduled task

Write-Host "=== Setting up automatic backup task ===" -ForegroundColor Cyan

# Task configuration
$TaskName = "ByAca-AutoBackup"
$TaskDescription = "Automatically backup Obsidian notes from iCloud to GitHub"
$ScriptPath = "C:\Users\xinby\Documents\ByAca-backup\backup.ps1"
$Time = "20:00"  # 8:00 PM

# Check if task already exists
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($existingTask) {
    Write-Host "Task '$TaskName' already exists. Removing old task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Create action (run the backup script with -Push parameter)
$Action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`" -Push"

# Create trigger (daily at specified time)
$Trigger = New-ScheduledTaskTrigger -Daily -At $Time

# Create settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

# Create principal (run as current user)
$Principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive

# Register the task
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Description $TaskDescription `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Force | Out-Null
    
    Write-Host "`nTask created successfully!" -ForegroundColor Green
    Write-Host "`nTask details:" -ForegroundColor Cyan
    Write-Host "  Name: $TaskName"
    Write-Host "  Schedule: Daily at $Time"
    Write-Host "  Action: Backup and push to GitHub"
    Write-Host "  Script: $ScriptPath"
    
    Write-Host "`nYou can manage this task in:" -ForegroundColor Yellow
    Write-Host "  Task Scheduler -> Task Scheduler Library -> $TaskName"
    
    # Test if we can run the task now
    Write-Host "`nDo you want to test run the backup now? (Y/N): " -NoNewline -ForegroundColor Yellow
    $response = Read-Host
    
    if ($response -eq "Y" -or $response -eq "y") {
        Write-Host "`nRunning backup task..." -ForegroundColor Cyan
        Start-ScheduledTask -TaskName $TaskName
        Write-Host "Task started. Check the backup log for results." -ForegroundColor Green
    }
    
} catch {
    Write-Host "`nError creating task: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "You may need to run this script as Administrator." -ForegroundColor Yellow
    exit 1
}

Write-Host "`n=== Setup complete ===" -ForegroundColor Cyan
