$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut("$env:USERPROFILE\Desktop\Market Bot.lnk")
$sc.TargetPath = "python"
$sc.Arguments = "run_service.py"
$sc.WorkingDirectory = "C:\Users\mattg\Downloads\Claude"
$sc.IconLocation = "C:\Windows\System32\cmd.exe,0"
$sc.Save()
Write-Host "Shortcut created on Desktop"
