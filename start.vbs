Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

projectDir = "C:\Users\Brady Huang\Mitsubishi-Electric-Drawing-Replacement-Project"
pythonExe = "C:\Users\Brady Huang\miniconda3\envs\mitsubishi\python.exe"
logsDir = projectDir & "\logs"

' Create logs directory
If Not fso.FolderExists(logsDir) Then
    fso.CreateFolder(logsDir)
End If

' Start Backend (hidden, logs to file)
backendCmd = "cmd /c cd /d """ & projectDir & """ && """ & pythonExe & """ -m uvicorn api.main:app --host 0.0.0.0 --port 8000 > """ & logsDir & "\backend.log"" 2>&1"
WshShell.Run backendCmd, 0, False

' Start Frontend (hidden, logs to file)
frontendCmd = "cmd /c cd /d """ & projectDir & "\dashboard"" && npm run dev > """ & logsDir & "\frontend.log"" 2>&1"
WshShell.Run frontendCmd, 0, False

' Wait for servers to start, then open browser
WScript.Sleep 3000
WshShell.Run "http://localhost:3000", 1, False
