# ByAca Backup Task Management Tool
# 备份任务管理工具

param(
    [Parameter(Position=0)]
    [ValidateSet("status", "run", "disable", "enable", "delete", "help")]
    [string]$Action = "help"
)

$TaskName = "ByAca-AutoBackup"

function Show-Help {
    Write-Host "`n=== ByAca 备份任务管理工具 ===" -ForegroundColor Cyan
    Write-Host "`n用法: .\管理工具.ps1 <action>`n"
    Write-Host "可用操作:" -ForegroundColor Yellow
    Write-Host "  status  - 查看任务状态"
    Write-Host "  run     - 立即运行备份"
    Write-Host "  disable - 禁用自动备份"
    Write-Host "  enable  - 启用自动备份"
    Write-Host "  delete  - 删除备份任务"
    Write-Host "  help    - 显示此帮助"
    Write-Host ""
}

function Show-Status {
    Write-Host "`n=== 备份任务状态 ===" -ForegroundColor Cyan
    
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    
    if (-not $task) {
        Write-Host "任务不存在！请重新运行 install-task.ps1" -ForegroundColor Red
        return
    }
    
    $info = Get-ScheduledTaskInfo -TaskName $TaskName
    
    Write-Host "`n任务名称: " -NoNewline
    Write-Host $task.TaskName -ForegroundColor Green
    
    Write-Host "状态: " -NoNewline
    if ($task.State -eq "Ready") {
        Write-Host $task.State -ForegroundColor Green
    } elseif ($task.State -eq "Disabled") {
        Write-Host $task.State -ForegroundColor Yellow
    } else {
        Write-Host $task.State -ForegroundColor Gray
    }
    
    Write-Host "描述: " -NoNewline
    Write-Host $task.Description -ForegroundColor Gray
    
    Write-Host "`n上次运行: " -NoNewline
    if ($info.LastRunTime -and $info.LastRunTime.Year -gt 2000) {
        Write-Host $info.LastRunTime -ForegroundColor Cyan
    } else {
        Write-Host "从未运行" -ForegroundColor Yellow
    }
    
    Write-Host "下次运行: " -NoNewline
    Write-Host $info.NextRunTime -ForegroundColor Green
    
    Write-Host "上次结果: " -NoNewline
    if ($info.LastTaskResult -eq 0) {
        Write-Host "成功 (0)" -ForegroundColor Green
    } elseif ($info.LastTaskResult -eq 267011) {
        Write-Host "未运行 (267011)" -ForegroundColor Yellow
    } else {
        Write-Host "错误 ($($info.LastTaskResult))" -ForegroundColor Red
    }
    
    Write-Host ""
}

function Run-Task {
    Write-Host "`n正在运行备份任务..." -ForegroundColor Yellow
    Start-ScheduledTask -TaskName $TaskName
    Write-Host "备份任务已启动！" -ForegroundColor Green
    Write-Host "查看日志: Get-ScheduledTaskInfo -TaskName '$TaskName'" -ForegroundColor Gray
    Write-Host ""
}

function Disable-Task {
    Write-Host "`n正在禁用自动备份..." -ForegroundColor Yellow
    Disable-ScheduledTask -TaskName $TaskName | Out-Null
    Write-Host "自动备份已禁用" -ForegroundColor Yellow
    Write-Host "使用 '.\管理工具.ps1 enable' 重新启用" -ForegroundColor Gray
    Write-Host ""
}

function Enable-Task {
    Write-Host "`n正在启用自动备份..." -ForegroundColor Yellow
    Enable-ScheduledTask -TaskName $TaskName | Out-Null
    Write-Host "自动备份已启用" -ForegroundColor Green
    Write-Host ""
}

function Delete-Task {
    Write-Host "`n警告: 这将删除自动备份任务！" -ForegroundColor Red
    Write-Host "确认删除? (Y/N): " -NoNewline -ForegroundColor Yellow
    $response = Read-Host
    
    if ($response -eq "Y" -or $response -eq "y") {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "备份任务已删除" -ForegroundColor Yellow
        Write-Host "使用 install-task.ps1 重新创建" -ForegroundColor Gray
    } else {
        Write-Host "取消删除" -ForegroundColor Green
    }
    Write-Host ""
}

# Main logic
switch ($Action) {
    "status"  { Show-Status }
    "run"     { Run-Task }
    "disable" { Disable-Task }
    "enable"  { Enable-Task }
    "delete"  { Delete-Task }
    "help"    { Show-Help }
    default   { Show-Help }
}
