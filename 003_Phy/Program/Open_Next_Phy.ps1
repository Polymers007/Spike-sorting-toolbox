param(
    [string]$Device
)

$ErrorActionPreference = 'Stop'
$programDirectory = [System.IO.Path]::GetDirectoryName($PSCommandPath)
$toolDirectory = [System.IO.Path]::GetDirectoryName($programDirectory)
$projectRoot = [System.IO.Path]::GetDirectoryName($toolDirectory)
$sortingRoot = Join-Path $projectRoot '004_Result\Kilosort4'
$launcher = Join-Path $programDirectory 'Open_Phy.ps1'

$results = @(Get-ChildItem -LiteralPath $sortingRoot -File -Recurse -Filter 'params.py' |
    Where-Object {
        $folder = $_.Directory
        (Test-Path -LiteralPath (Join-Path $folder.FullName 'spike_times.npy')) -and
        (-not (Test-Path -LiteralPath (Join-Path $folder.FullName 'MANUAL_CURATION_CONFIRMED.json'))) -and
        (-not (Test-Path -LiteralPath (Join-Path $projectRoot ("004_Result\Phy\Working_Curated\{0}\{1}\MANUAL_CURATION_CONFIRMED.json" -f $folder.Parent.Name, $folder.Name)))) -and
        (-not $Device -or $folder.Parent.Name -ieq $Device)
    } |
    Sort-Object FullName)

if ($results.Count -eq 0) {
    Write-Host 'No uncompleted Phy results were found.' -ForegroundColor Green
    exit 0
}

$params = $results[0]
$resultFolder = $params.Directory.FullName
Write-Host "Opening next uncompleted result: $resultFolder"
& $launcher -ResultFolder $resultFolder
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) { exit $exitCode }

Write-Host 'The save guard has checked the session. Use the Curation Manager for reliable queue status.'
