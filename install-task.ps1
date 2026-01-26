# Install automatic backup task (non-interactive)
$TaskName = "ByAca-AutoBackup"
$TaskDescription = "Automatically backup Obsidian notes from iCloud to GitHub"
$ScriptPath = "C:\Users\xinby\Documents\ByAca-backup\backup.ps1"
$Time = "20:00"

# Remove existing task if present
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Create task
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`" -Push"
$Trigger = New-ScheduledTaskTrigger -Daily -At $Time
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 1)
$Principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive

Register-ScheduledTask -TaskName $TaskName -Description $TaskDescription -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Force | Out-Null

Write-Host "Task '$TaskName' created successfully!" -ForegroundColor Green
Write-Host "Schedule: Daily at $Time (8:00 PM)" -ForegroundColor Cyan
Write-Host "Action: Backup and push to GitHub" -ForegroundColor Cyan
