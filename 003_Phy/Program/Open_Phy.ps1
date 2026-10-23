param(
    [Parameter(Position = 0)]
    [string]$ResultFolder,

    [switch]$Diagnose
)

$ErrorActionPreference = 'Stop'
$scriptDirectory = [System.IO.Path]::GetDirectoryName($PSCommandPath)
$toolDirectory = [System.IO.Path]::GetDirectoryName($scriptDirectory)
$projectRoot = [System.IO.Path]::GetDirectoryName($toolDirectory)
$clusterRoot = Join-Path $projectRoot '004_Result\Phy\Working_Curated'

function Find-PhyExecutable {
    $candidates = New-Object System.Collections.Generic.List[string]

    if ($env:AP_PHY_EXE) {
        $candidates.Add($env:AP_PHY_EXE)
    }

    $pathCommand = Get-Command phy.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pathCommand) {
        $candidates.Add($pathCommand.Source)
    }

    $candidates.Add((Join-Path $env:USERPROFILE '.local\bin\phy.exe'))
    if ($env:APPDATA) {
        $candidates.Add((Join-Path $env:APPDATA 'uv\data\tools\phy\Scripts\phy.exe'))
    }
    $uvCommand = Get-Command uv.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($uvCommand) {
        try {
            $uvBinDirectory = (& $uvCommand.Source tool dir --bin 2>$null | Select-Object -First 1)
            if ($uvBinDirectory) {
                $candidates.Add((Join-Path $uvBinDirectory 'phy.exe'))
            }
        }
        catch {
            
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
        $candidates.Add((Join-Path $root 'phy\Scripts\phy.exe'))
        if (Test-Path -LiteralPath $root -PathType Container) {
            foreach ($environment in Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue) {
                $candidates.Add((Join-Path $environment.FullName 'Scripts\phy.exe'))
            }
        }
    }

    foreach ($candidate in $candidates) {
        if ($candidate -and -not [IO.Path]::IsPathRooted($candidate)) { $candidate = Join-Path $projectRoot $candidate }
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return $null
}

function Show-ErrorMessage([string]$message) {
    Write-Host ''
    Write-Host "ERROR: $message" -ForegroundColor Red
    if (-not $Diagnose) {
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.MessageBox]::Show(
            $message,
            'Open Phy',
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
    }
}

function Select-ResultFolder([string]$initialDirectory) {
    Add-Type -AssemblyName System.Windows.Forms
    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $dialog.Description = 'Select the Kilosort result folder containing params.py'
    $dialog.ShowNewFolderButton = $false
    if (Test-Path -LiteralPath $initialDirectory -PathType Container) {
        $dialog.SelectedPath = $initialDirectory
    }
    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        return $dialog.SelectedPath
    }
    return $null
}

function Read-DataPath([string]$paramsFile) {
    $content = Get-Content -LiteralPath $paramsFile -Raw
    $match = [regex]::Match(
        $content,
        '(?m)^\s*dat_path\s*=\s*\[?\s*[''"](?<path>[^''"]+)[''"]'
    )
    if ($match.Success) {
        return $match.Groups['path'].Value
    }
    return $null
}

function Resolve-DataFile([string]$configuredPath, [string]$resultDirectory) {
    if ($configuredPath) {
        if ([System.IO.Path]::IsPathRooted($configuredPath)) {
            if (Test-Path -LiteralPath $configuredPath -PathType Leaf) {
                return (Resolve-Path -LiteralPath $configuredPath).Path
            }
        }
        else {
            $relativeCandidate = Join-Path $resultDirectory $configuredPath
            if (Test-Path -LiteralPath $relativeCandidate -PathType Leaf) {
                return (Resolve-Path -LiteralPath $relativeCandidate).Path
            }
        }
    }

    $binaryName = if ($configuredPath) { [System.IO.Path]::GetFileName($configuredPath) } else { $null }
    $binaryRoot = Join-Path $projectRoot '001_Bin\Data'
    if ($binaryName -and (Test-Path -LiteralPath $binaryRoot -PathType Container)) {
        $matches = @(Get-ChildItem -LiteralPath $binaryRoot -Recurse -File -Filter $binaryName -ErrorAction SilentlyContinue)
        if ($matches.Count -eq 1) {
            return $matches[0].FullName
        }
    }

    if ($Diagnose) {
        return $null
    }

    Add-Type -AssemblyName System.Windows.Forms
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = 'The original binary moved. Select its current .bin file'
    $dialog.Filter = 'Kilosort binary (*.bin)|*.bin|All files (*.*)|*.*'
    $dialog.CheckFileExists = $true
    if (Test-Path -LiteralPath $binaryRoot -PathType Container) {
        $dialog.InitialDirectory = $binaryRoot
    }
    if ($binaryName) {
        $dialog.FileName = $binaryName
    }
    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        return $dialog.FileName
    }
    return $null
}

function Write-PortableParams([string]$sourceFile, [string]$binaryFile) {
    $portableFile = Join-Path ([System.IO.Path]::GetDirectoryName($sourceFile)) 'params_portable.py'
    $paramsDirectory = [IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($sourceFile))
    $baseUri = New-Object System.Uri ($paramsDirectory.TrimEnd('\') + '\')
    $binaryUri = New-Object System.Uri ([IO.Path]::GetFullPath($binaryFile))
    $pythonPath = [Uri]::UnescapeDataString($baseUri.MakeRelativeUri($binaryUri).ToString()).Replace("'", "\'")
    $content = Get-Content -LiteralPath $sourceFile -Raw
    $replacement = "dat_path = [r'$pythonPath']"
    $updated = [regex]::Replace(
        $content,
        "(?m)^\s*dat_path\s*=.*$",
        $replacement,
        1
    )
    if ($updated -eq $content) {
        throw "dat_path was not found in $sourceFile"
    }
    [System.IO.File]::WriteAllText($portableFile, $updated, (New-Object System.Text.UTF8Encoding($false)))
    return $portableFile
}

$phyExecutable = Find-PhyExecutable
if (-not $phyExecutable) {
    Show-ErrorMessage @"
Phy was not found on this computer.

The launcher checked PATH and common Conda environment folders. Install Phy in a Conda environment, or set the AP_PHY_EXE environment variable to the full path of phy.exe.
"@
    exit 2
}

if ($Diagnose) {
    Write-Host "Project root : $projectRoot"
    Write-Host "Cluster root : $clusterRoot"
    Write-Host "Phy executable: $phyExecutable"
    exit 0
}

if (-not $ResultFolder) {
    $ResultFolder = Select-ResultFolder $clusterRoot
}
if (-not $ResultFolder) {
    Write-Host 'No result folder was selected. Phy was not started.'
    exit 0
}
$ResultFolder = (Resolve-Path -LiteralPath $ResultFolder).Path

$requiredFiles = @('params.py', 'spike_times.npy', 'spike_clusters.npy', 'templates.npy')
foreach ($fileName in $requiredFiles) {
    $filePath = Join-Path $ResultFolder $fileName
    if (-not (Test-Path -LiteralPath $filePath -PathType Leaf)) {
        Show-ErrorMessage "The selected folder is not a complete Kilosort result. Missing file: $filePath"
        exit 3
    }
}

$paramsFile = Join-Path $ResultFolder 'params.py'
$configuredDataPath = Read-DataPath $paramsFile
$binaryFile = Resolve-DataFile $configuredDataPath $ResultFolder
if (-not $binaryFile) {
    Show-ErrorMessage @"
The binary file referenced by params.py is unavailable, and no replacement was selected.

Original path: $configuredDataPath

Phy needs the corresponding .bin file to open this result correctly.
"@
    exit 4
}

$launchParams = $paramsFile
if (-not $configuredDataPath -or -not (Test-Path -LiteralPath $configuredDataPath -PathType Leaf) -or
    ((Resolve-Path -LiteralPath $configuredDataPath -ErrorAction SilentlyContinue).Path -ne $binaryFile)) {
    $launchParams = Write-PortableParams $paramsFile $binaryFile
    Write-Host "Using relocated binary: $binaryFile"
    Write-Host "Session parameters: $launchParams"
}

Write-Host ''
Write-Host "Opening Phy result: $ResultFolder"
Write-Host "Phy executable   : $phyExecutable"
Write-Host ''

$guard = Join-Path $scriptDirectory 'phy_session_guard.py'
$python = Join-Path ([System.IO.Path]::GetDirectoryName([System.IO.Path]::GetDirectoryName($phyExecutable))) 'python.exe'
if (-not (Test-Path -LiteralPath $guard -PathType Leaf)) {
    Show-ErrorMessage "Phy save guard is missing: $guard"
    exit 5
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    Show-ErrorMessage "Python executable for Phy was not found: $python"
    exit 5
}

& $python $guard --result-folder $ResultFolder --params-file $launchParams --phy-exe $phyExecutable
$exitCode = $LASTEXITCODE

if ($null -eq $exitCode) { $exitCode = 0 }
exit $exitCode
