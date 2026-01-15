Param(
    [string]$Name = "PhyCLNetDemo",
    [string]$Entry = "app/main.py"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "Building $Name from $Entry ..."

# Ensure we run from repo root
Set-Location (Split-Path -Parent $PSCommandPath)
Set-Location ..

if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    Write-Error "pyinstaller not found. Activate your venv and run: pip install pyinstaller"
}

pyinstaller `
    --noconfirm `
    --noconsole `
    --onedir `
    --name $Name `
    --collect-all torch `
    --collect-all PySide6 `
    --add-data "template;template" `
    $Entry

Write-Host "Build finished. Check dist/$Name/"
