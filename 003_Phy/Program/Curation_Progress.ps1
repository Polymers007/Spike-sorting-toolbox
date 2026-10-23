param(
    [string]$Device
)

$programDirectory = [System.IO.Path]::GetDirectoryName($PSCommandPath)
$toolDirectory = [System.IO.Path]::GetDirectoryName($programDirectory)
$projectRoot = [System.IO.Path]::GetDirectoryName($toolDirectory)
$sortingRoot = Join-Path $projectRoot '004_Result\Kilosort4'
$outputRoot = Join-Path $projectRoot '004_Result\Phy'
$folders = @(Get-ChildItem -LiteralPath $sortingRoot -File -Recurse -Filter 'params.py' |
    ForEach-Object { $_.Directory } |
    Where-Object { -not $Device -or $_.Parent.Name -ieq $Device } |
    Sort-Object FullName)

$rows = foreach ($folder in $folders) {
    $relative = Join-Path $folder.Parent.Name $folder.Name
    $working = Join-Path (Join-Path $outputRoot 'Working_Curated') $relative
    $status = if (Test-Path -LiteralPath (Join-Path $working 'MANUAL_CURATION_CONFIRMED.json')) {
        'ManualConfirmed'
    }
    elseif (Test-Path -LiteralPath (Join-Path $working 'CURATION_SESSION.json')) {
        'ManualReview'
    }
    elseif (Test-Path -LiteralPath (Join-Path $working 'AUTO_CURATION_DONE.json')) {
        'AutoCurated'
    }
    else {
        'Unprocessed'
    }
    [pscustomobject]@{
        Device = $folder.Parent.Name
        Recording = $folder.Name
        Status = $status
    }
}
$rows | Format-Table -AutoSize
Write-Host ''
Write-Host "Total: $($rows.Count)"
Write-Host "Auto curated: $(@($rows | Where-Object Status -eq 'AutoCurated').Count)"
Write-Host "Manual review: $(@($rows | Where-Object Status -eq 'ManualReview').Count)"
Write-Host "Manual confirmed: $(@($rows | Where-Object Status -eq 'ManualConfirmed').Count)"
Write-Host "Unprocessed: $(@($rows | Where-Object Status -eq 'Unprocessed').Count)"
