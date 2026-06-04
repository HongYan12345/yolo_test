param(
    [int]$SyntheticCount = 3000,
    [int]$ImageSize = 0,
    [int]$Epochs = 100,
    [int]$Batch = 0,
    [int]$Workers = 4,
    [int]$Seed = 42,
    [string]$Model = "yolov8n.pt",
    [string]$RunName = "",
    [string]$TorchCudaIndexUrl = "",
    [switch]$Cpu,
    [switch]$SkipInstall,
    [switch]$RebuildSynthetic,
    [switch]$RebuildYolo,
    [switch]$NoTrain,
    [switch]$NoExport
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

function Write-Section {
    param([string]$Message)

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Get-NvidiaInfo {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        return $null
    }

    try {
        $queryOutput = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $queryOutput) {
            return $null
        }

        $firstLine = @($queryOutput)[0]
        $parts = $firstLine -split ","

        if ($parts.Count -lt 3) {
            return $null
        }

        $memoryMb = 0
        [void][int]::TryParse($parts[1].Trim(), [ref]$memoryMb)

        return [pscustomobject]@{
            Name          = $parts[0].Trim()
            MemoryMb      = $memoryMb
            DriverVersion = $parts[2].Trim()
        }
    }
    catch {
        return $null
    }
}

function Test-DriverAtLeast {
    param(
        [string]$DriverVersion,
        [int]$Major,
        [int]$Minor
    )

    try {
        $version = [version]$DriverVersion

        if ($version.Major -gt $Major) {
            return $true
        }

        if ($version.Major -eq $Major -and $version.Minor -ge $Minor) {
            return $true
        }

        return $false
    }
    catch {
        return $false
    }
}

function Select-TorchCudaIndexUrl {
    param($GpuInfo)

    if ($TorchCudaIndexUrl.Trim().Length -gt 0) {
        return $TorchCudaIndexUrl
    }

    if ($null -eq $GpuInfo) {
        return ""
    }

    $gpuName = $GpuInfo.Name

    # GTX 10 系、老 Quadro/Tesla 默认使用 cu118，对旧驱动和 Pascal 架构更稳。
    $isOldOrPascalGpu = $gpuName -match "GTX\s*10|GTX\s*9|GTX\s*7|GTX\s*6|Quadro\s*(K|M|P)|Tesla\s*(K|M|P)"

    if ($isOldOrPascalGpu) {
        return "https://download.pytorch.org/whl/cu118"
    }

    # CUDA 12.1 PyTorch wheel 通常要求 NVIDIA driver >= 525.60。
    if (Test-DriverAtLeast -DriverVersion $GpuInfo.DriverVersion -Major 525 -Minor 60) {
        return "https://download.pytorch.org/whl/cu121"
    }

    return "https://download.pytorch.org/whl/cu118"
}

function Add-SwitchArg {
    param(
        [System.Collections.ArrayList]$ArgList,
        [string]$Name,
        [bool]$Enabled
    )

    if ($Enabled) {
        [void]$ArgList.Add($Name)
    }
}

Write-Section "MapleStoryAI auto training launcher"

$gpu = Get-NvidiaInfo
$useCpu = [bool]$Cpu

if ($null -eq $gpu) {
    Write-Host "[INFO] NVIDIA GPU was not detected by nvidia-smi. CPU mode will be used."
    $useCpu = $true
}
else {
    Write-Host "[INFO] NVIDIA GPU detected:"
    Write-Host "       name   : $($gpu.Name)"
    Write-Host "       memory : $($gpu.MemoryMb) MB"
    Write-Host "       driver : $($gpu.DriverVersion)"
}

if ($useCpu) {
    $effectiveImageSize = if ($ImageSize -gt 0) { $ImageSize } else { 512 }
    $effectiveBatch = if ($Batch -gt 0) { $Batch } else { 4 }
    $effectiveRunName = if ($RunName.Trim().Length -gt 0) { $RunName } else { "maple_auto_cpu_$effectiveImageSize" }
    $effectiveTorchCudaIndexUrl = ""
}
else {
    $effectiveImageSize = if ($ImageSize -gt 0) { $ImageSize } else { 640 }

    if ($Batch -gt 0) {
        $effectiveBatch = $Batch
    }
    elseif ($gpu.MemoryMb -gt 0 -and $gpu.MemoryMb -lt 7000) {
        $effectiveBatch = 4
    }
    else {
        $effectiveBatch = 8
    }

    $effectiveRunName = if ($RunName.Trim().Length -gt 0) { $RunName } else { "maple_auto_gpu_$effectiveImageSize" }
    $effectiveTorchCudaIndexUrl = Select-TorchCudaIndexUrl -GpuInfo $gpu
}

Write-Host "Project root     : $ProjectRoot"
Write-Host "Selected device  : $(if ($useCpu) { 'cpu' } else { 'cuda:0' })"
Write-Host "Synthetic count  : $SyntheticCount"
Write-Host "Image size       : $effectiveImageSize"
Write-Host "Epochs           : $Epochs"
Write-Host "Batch            : $effectiveBatch"
Write-Host "Workers          : $Workers"
Write-Host "Model            : $Model"
Write-Host "Run name         : $effectiveRunName"

if (-not $useCpu) {
    Write-Host "Torch CUDA index : $effectiveTorchCudaIndexUrl"
}

$trainScript = Join-Path $ProjectRoot "train_1080.ps1"

if (-not (Test-Path $trainScript)) {
    throw "Missing training script: $trainScript"
}

$argsList = [System.Collections.ArrayList]@(
    "-SyntheticCount", $SyntheticCount,
    "-ImageSize", $effectiveImageSize,
    "-Epochs", $Epochs,
    "-Batch", $effectiveBatch,
    "-Workers", $Workers,
    "-Seed", $Seed,
    "-Model", $Model,
    "-RunName", $effectiveRunName
)

if (-not $useCpu) {
    [void]$argsList.Add("-TorchCudaIndexUrl")
    [void]$argsList.Add($effectiveTorchCudaIndexUrl)
}

Add-SwitchArg -ArgList $argsList -Name "-Cpu" -Enabled $useCpu
Add-SwitchArg -ArgList $argsList -Name "-SkipInstall" -Enabled $SkipInstall
Add-SwitchArg -ArgList $argsList -Name "-RebuildSynthetic" -Enabled $RebuildSynthetic
Add-SwitchArg -ArgList $argsList -Name "-RebuildYolo" -Enabled $RebuildYolo
Add-SwitchArg -ArgList $argsList -Name "-NoTrain" -Enabled $NoTrain
Add-SwitchArg -ArgList $argsList -Name "-NoExport" -Enabled $NoExport

Write-Section "Delegating to train_1080.ps1"
Write-Host "Command:"
Write-Host "  powershell -ExecutionPolicy Bypass -File train_1080.ps1 $($argsList -join ' ')"

& $trainScript @argsList
