$ErrorActionPreference = 'Stop'
$toolDirectory = [IO.Path]::GetDirectoryName($PSCommandPath)
$projectRoot = [IO.Path]::GetDirectoryName($toolDirectory)
$manager = Join-Path $toolDirectory 'Program\unified_curation_manager.py'
$candidates = New-Object System.Collections.Generic.List[string]
if ($env:AP_PHY_PYTHON) { $candidates.Add($env:AP_PHY_PYTHON) }
foreach ($name in @('pythonw.exe', 'python.exe', 'python')) {
    $command = Get-Command $name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command -and $command.Source) { $candidates.Add($command.Source) }
}
foreach ($root in @(
    (Join-Path $env:USERPROFILE '.conda\envs'),
    (Join-Path $env:USERPROFILE 'anaconda3\envs'),
    (Join-Path $env:USERPROFILE 'miniconda3\envs'),
    (Join-Path $env:ProgramData 'anaconda3\envs'),
    (Join-Path $env:ProgramData 'miniconda3\envs')
)) {
    $candidates.Add((Join-Path $root 'phy\pythonw.exe'))
    $candidates.Add((Join-Path $root 'phy\python.exe'))
}
$python = $null
foreach ($candidate in $candidates | Select-Object -Unique) {
    if ($candidate -and -not [IO.Path]::IsPathRooted($candidate)) { $candidate = Join-Path $projectRoot $candidate }
    if (-not $candidate -or -not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
    $testExecutable = if ($candidate.EndsWith('pythonw.exe')) {
        Join-Path ([IO.Path]::GetDirectoryName($candidate)) 'python.exe'
    } else { $candidate }
    if (-not (Test-Path -LiteralPath $testExecutable -PathType Leaf)) { continue }
    
    
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    & $testExecutable -B -c 'import numpy, phy, phylib, PyQt5' 2>$null
    $probeExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
    if ($probeExitCode -eq 0) { $python = $candidate; break }
}
if (-not $python) {
    Write-Host 'ERROR: A Python environment containing Phy, phylib, PyQt5, and NumPy was not found.' -ForegroundColor Red
    Write-Host 'Activate the Phy environment or set AP_PHY_PYTHON to its Python executable.'
    exit 2
}
$logDirectory = Join-Path $projectRoot '004_Result\Phy\Batch_Logs\Manager_Startup'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$stdoutLog = Join-Path $logDirectory ($stamp + '_stdout.log')
$stderrLog = Join-Path $logDirectory ($stamp + '_stderr.log')




$quotedManager = '"' + $manager + '"'
$process = Start-Process -FilePath $python -ArgumentList $quotedManager `
    -WorkingDirectory (Join-Path $projectRoot '004_Result\Phy') `
    -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru

Start-Sleep -Milliseconds 1200
if ($process.HasExited -and $process.ExitCode -ne 0) {
    Write-Host "ERROR: Curation Manager exited during startup (code $($process.ExitCode))." -ForegroundColor Red
    if (Test-Path -LiteralPath $stderrLog) {
        Get-Content -LiteralPath $stderrLog
    }
    Write-Host "Startup log: $stderrLog"
    exit $process.ExitCode
}
exit 0
