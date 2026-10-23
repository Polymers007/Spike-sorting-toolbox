param(
    [string]$Device,
    [int]$MaxSpikes = 200,
    [int]$MaxChannels = 16,
    [int]$MaxShift = 6,
    [double]$MinimumCorrelation = 0.30,
    [switch]$NoAlign,
    [switch]$DryRun,
    [switch]$Force,
    [switch]$Pilot
)

$ErrorActionPreference = 'Stop'
$programDirectory = [System.IO.Path]::GetDirectoryName($PSCommandPath)
$projectRoot = Split-Path (Split-Path $programDirectory -Parent) -Parent
$candidates = @()
if ($env:AP_PHY_PYTHON) {
    $candidate = $env:AP_PHY_PYTHON
    if (-not [IO.Path]::IsPathRooted($candidate)) { $candidate = Join-Path $projectRoot $candidate }
    $candidates += $candidate
}
foreach ($name in @('python.exe', 'python')) {
    $command = Get-Command $name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command) { $candidates += $command.Source }
}
foreach ($base in @('.conda\envs', 'anaconda3\envs', 'miniconda3\envs')) {
    $candidates += Join-Path (Join-Path $env:USERPROFILE $base) 'phy\python.exe'
}
$pythonExecutable = $null
foreach ($candidate in $candidates | Select-Object -Unique) {
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
    & $candidate -B -c 'import numpy, phy, phylib, PyQt5' 2>$null
    if ($LASTEXITCODE -eq 0) { $pythonExecutable = $candidate; break }
}
if (-not $pythonExecutable) {
    Write-Host 'ERROR: Set AP_PHY_PYTHON to the Phy environment Python executable.' -ForegroundColor Red
    exit 2
}

$arguments = @(
    (Join-Path $programDirectory 'batch_phy_prepare.py'),
    '--max-spikes', $MaxSpikes,
    '--max-channels', $MaxChannels,
    '--max-shift', $MaxShift,
    '--minimum-correlation', $MinimumCorrelation
)
if ($Device) { $arguments += @('--device', $Device) }
if ($NoAlign) { $arguments += '--no-align' }
if ($DryRun) { $arguments += '--dry-run' }
if ($Force) { $arguments += '--force' }
if ($Pilot) { $arguments += '--pilot' }
& $pythonExecutable @arguments
exit $LASTEXITCODE
