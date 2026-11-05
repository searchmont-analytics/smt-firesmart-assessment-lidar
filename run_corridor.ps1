# One-click FireSmart corridor run for Windows.
#
#   Right-click > "Run with PowerShell", or from a PowerShell window in this folder:
#       .\run_corridor.ps1
#   Optional:
#       .\run_corridor.ps1 -TileDir "D:\lidar" -Workers 2
#
# Steps: set up Python packages (first run only) -> download missing tiles ->
#        score every house -> build the map. Safe to re-run; finished steps are skipped.

param(
    [string]$TileDir  = "$env:USERPROFILE\Documents\Claude-Code\Searchmont Firesmart Assessment - lidar",
    [string]$TileList = "$PSScriptRoot\tiles_searchmont_corridor.txt",
    [string]$Out      = "$PSScriptRoot\output_corridor",
    [int]$Workers     = 1      # ~3 GB RAM per worker; use 2 on a 16 GB PC
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

Step "Checking Python"
$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command python -ErrorAction SilentlyContinue }
if (-not $py) { throw "Python 3.10+ not found. Install it from https://www.python.org/downloads/ (tick 'Add to PATH') and run this again." }

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Step "Creating a Python environment (first run only)"
    & $py.Source -m venv .venv
    & .venv\Scripts\python.exe -m pip install --upgrade pip
    & .venv\Scripts\python.exe -m pip install -r requirements.txt
}
$venv = ".venv\Scripts\python.exe"

$drive = (Get-Item $TileDir -ErrorAction SilentlyContinue)
if ($drive) {
    $free = (Get-PSDrive ($drive.FullName.Substring(0,1))).Free / 1GB
    Write-Host ("Free space on tile drive: {0:N0} GB (the full corridor needs about 18 GB)" -f $free)
}

Step "Downloading missing tiles"
& $venv scripts\fetch_tiles.py --list $TileList --dir $TileDir
if ($LASTEXITCODE -ne 0) { throw "Some downloads failed. Run this script again to resume." }

Step "Scoring houses (about 30-60 s per tile)"
& $venv scripts\run_corridor.py --tiles "$TileDir\*.laz" --out $Out --config config_ssm2024.yaml --name "Searchmont corridor" --workers $Workers
if ($LASTEXITCODE -ne 0) { throw "Scoring failed; see the messages above." }

Step "Building the map"
$houses = (Get-Content "$Out\summary.json" | ConvertFrom-Json).houses
& $venv scripts\build_map.py --out $Out --html "$Out\searchmont_corridor_map.html" `
    --title "Searchmont FireSmart Map" --h1 "Searchmont wildfire risk by property" --tag "2024 LiDAR - screening only" `
    --intro "$houses houses along the Searchmont highway corridor, each scored on tree canopy and ladder fuel within 10 m and 30 m, from the 2024 Sault Ste Marie airborne LiDAR. Zoom in to see house labels. A starting point for FireSmart conversations, not a substitute for an on-site assessment." `
    --attribution "LiDAR: NRCan CanElevation / Ontario, 2024 - Open Government Licence - Canada"

Copy-Item "$Out\searchmont_corridor_map.html", "$Out\properties.csv" -Destination $TileDir -Force
Step "Done"
Write-Host "Map:  $TileDir\searchmont_corridor_map.html"
Write-Host "Data: $TileDir\properties.csv"
