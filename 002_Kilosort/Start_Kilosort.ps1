param(
    [ValidateSet('Gui', 'Batch', 'Parameters', 'Diagnose')]
    [string]$Mode = 'Gui',

    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$programDirectory = [System.IO.Path]::GetDirectoryName($PSCommandPath)
$projectRoot = [System.IO.Path]::GetDirectoryName($programDirectory)

function Test-KilosortPython([string]$pythonExecutable) {
    if ($pythonExecutable -and -not [IO.Path]::IsPathRooted($pythonExecutable)) { $pythonExecutable = Join-Path $projectRoot $pythonExecutable }
    if (-not $pythonExecutable -or -not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
        return $false
    }
    Push-Location $programDirectory
    try {
        & $pythonExecutable -B -c 'import torch, qtpy, kilosort' 2>$null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
    finally {
        Pop-Location
    }
}

function Find-KilosortPython {
    $candidates = New-Object System.Collections.Generic.List[string]
    if ($env:AP_KILOSORT_PYTHON) {
        $candidates.Add($env:AP_KILOSORT_PYTHON)
    }

    foreach ($commandName in @('python.exe', 'python')) {
        $command = Get-Command $commandName -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($command -and $command.Source) {
            $candidates.Add($command.Source)
        }
    }

    $environmentRoots = @(
        (Join-Path $env:USERPROFILE '.conda\envs'),
        (Join-Path $env:USERPROFILE 'anaconda3\envs'),
        (Join-Path $env:USERPROFILE 'miniconda3\envs'),
        (Join-Path $env:ProgramData 'anaconda3\envs'),
        (Join-Path $env:ProgramData 'miniconda3\envs')
    )
    foreach ($root in $environmentRoots) {
        $candidates.Add((Join-Path $root 'kilosort\python.exe'))
        if (Test-Path -LiteralPath $root -PathType Container) {
            foreach ($environment in Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue) {
                $candidates.Add((Join-Path $environment.FullName 'python.exe'))
            }
        }
    }

    foreach ($candidate in $candidates) {
        if (-not [IO.Path]::IsPathRooted($candidate)) { $candidate = Join-Path $projectRoot $candidate }
        if (Test-KilosortPython $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return $null
}

$pythonExecutable = Find-KilosortPython
if (-not $pythonExecutable) {
    Write-Host ''
    Write-Host 'ERROR: A Python environment containing Kilosort, PyTorch and QtPy was not found.' -ForegroundColor Red
    Write-Host 'Set AP_KILOSORT_PYTHON to the full path of that environment''s python.exe if automatic discovery is not possible.'
    exit 2
}

Write-Host "Project root     : $projectRoot"
Write-Host "Kilosort Python : $pythonExecutable"
if ($Mode -eq 'Diagnose') {
    exit 0
}

Push-Location $programDirectory
try {
    if ($Mode -eq 'Batch') {
        Write-Host 'Starting automatic Kilosort4 batch sorting...'
        Write-Host 'Close the Kilosort4 GUI before running this batch.'
        if ($DryRun) {
            & $pythonExecutable (Join-Path $programDirectory 'batch_sort_all.py') --dry-run
        }
        else {
            & $pythonExecutable (Join-Path $programDirectory 'batch_sort_all.py')
        }
    }
    elseif ($Mode -eq 'Parameters') {
        & $pythonExecutable (Join-Path $programDirectory 'kilosort_parameter_settings.py')
    }
    else {
        & $pythonExecutable -m kilosort --skip_load
    }
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($null -eq $exitCode) { $exitCode = 0 }
exit $exitCode
