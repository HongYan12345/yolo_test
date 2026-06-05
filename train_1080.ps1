param(
    [int]$SyntheticCount = 3000,
    [int]$ImageSize = 640,
    [int]$Epochs = 100,
    [int]$Batch = 8,
    [int]$Workers = 4,
    [int]$Seed = 42,
    [string]$Model = "yolov8n.pt",
    [string]$RunName = "maple_yolov8n_640",
    [string]$TorchCudaIndexUrl = "https://download.pytorch.org/whl/cu118",
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

$VenvDir = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$YoloExe = Join-Path $VenvDir "Scripts\yolo.exe"

function Write-Section {
    param([string]$Message)
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Count-Images {
    param([string]$Dir)
    if (-not (Test-Path $Dir)) {
        return 0
    }

    $count = (
        Get-ChildItem $Dir -Recurse -File -Include *.png,*.jpg,*.jpeg,*.bmp,*.webp |
        Measure-Object
    ).Count
    return $count
}

function Get-MissingLabels {
    param(
        [string]$ImageDir,
        [string]$LabelDir
    )

    if (-not (Test-Path $ImageDir)) {
        return @()
    }

    $imageRoot = (Resolve-Path $ImageDir).Path
    if (Test-Path $LabelDir) {
        $labelRoot = (Resolve-Path $LabelDir).Path
    }
    else {
        $labelRoot = Join-Path $ProjectRoot $LabelDir
    }

    $missing = @()
    $images = Get-ChildItem $ImageDir -Recurse -File -Include *.png,*.jpg,*.jpeg,*.bmp,*.webp

    foreach ($image in $images) {
        $relativeImagePath = $image.FullName.Substring($imageRoot.Length).TrimStart("\", "/")
        $relativeLabelPath = [System.IO.Path]::ChangeExtension($relativeImagePath, ".txt")
        $expectedLabelPath = Join-Path $labelRoot $relativeLabelPath

        if (-not (Test-Path $expectedLabelPath)) {
            $missing += $relativeLabelPath
        }
    }

    return $missing
}

function Test-SyntheticLabelsUseRoboflowClassMapping {
    param([string]$LabelDir)

    if (-not (Test-Path $LabelDir)) {
        return $true
    }

    $labelFiles = Get-ChildItem $LabelDir -Recurse -File -Include *.txt

    foreach ($labelFile in $labelFiles) {
        $lines = Get-Content $labelFile.FullName -ErrorAction SilentlyContinue

        foreach ($line in $lines) {
            $trimmed = $line.Trim()
            if ($trimmed.Length -eq 0) {
                continue
            }

            # Roboflow class mapping:
            #   0 Item
            #   1 Mob
            #   2 Platform
            #   3 Player
            #   4 Portal
            #   5 Rope
            # Synthetic generator only creates Mob/Item labels, so synthetic labels
            # should only contain class 0/1.
            if ($trimmed -notmatch "^[01]\s+") {
                return $false
            }
        }
    }

    return $true
}

function Test-YoloDataYamlUseRoboflowClassMapping {
    param([string]$DataYamlPath)

    if (-not (Test-Path $DataYamlPath)) {
        return $true
    }

    $content = Get-Content $DataYamlPath -Raw -ErrorAction SilentlyContinue

    return (
        $content -match "nc:\s*6" -and
        $content -match "0:\s*Item" -and
        $content -match "1:\s*Mob" -and
        $content -match "2:\s*Platform" -and
        $content -match "3:\s*Player" -and
        $content -match "4:\s*Portal" -and
        $content -match "5:\s*Rope"
    )
}

function New-Venv {
    if (Test-Path $PythonExe) {
        Write-Host "[OK] Existing virtual environment: $VenvDir"
        return
    }

    Write-Section "Creating Python virtual environment"

    $created = $false

    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            Write-Host "[INFO] Trying Python 3.11 via py launcher..."
            & py -3.11 -m venv $VenvDir
            $created = $true
        }
        catch {
            Write-Host "[WARN] Python 3.11 venv failed: $($_.Exception.Message)"
        }

        if (-not $created) {
            try {
                Write-Host "[INFO] Trying Python 3.10 via py launcher..."
                & py -3.10 -m venv $VenvDir
                $created = $true
            }
            catch {
                Write-Host "[WARN] Python 3.10 venv failed: $($_.Exception.Message)"
            }
        }
    }

    if (-not $created) {
        if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
            throw "Python was not found. Please install Python 3.10 or 3.11 on the GTX 1080 machine."
        }

        Write-Host "[INFO] Trying default python..."
        & python -m venv $VenvDir
        $created = $true
    }

    if (-not (Test-Path $PythonExe)) {
        throw "Virtual environment creation failed. Missing: $PythonExe"
    }

    Write-Host "[OK] Virtual environment created: $VenvDir"
}

function Test-TorchCuda {
    if (-not (Test-Path $PythonExe)) {
        return $false
    }

    try {
        $output = & $PythonExe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>$null
        if ($LASTEXITCODE -ne 0) {
            return $false
        }

        $lines = @($output)
        if ($lines.Count -ge 1 -and $lines[0].Trim() -eq "True") {
            if ($lines.Count -ge 2) {
                Write-Host "[OK] CUDA torch detected: $($lines[1])"
            }
            return $true
        }

        return $false
    }
    catch {
        return $false
    }
}

function Install-Dependencies {
    if ($SkipInstall) {
        Write-Host "[SKIP] Dependency installation skipped by -SkipInstall"
        return
    }

    Write-Section "Installing training dependencies"

    & $PythonExe -m pip install --upgrade pip

    if (-not $Cpu) {
        $cudaReady = Test-TorchCuda
        if (-not $cudaReady) {
            Write-Host "[INFO] Installing CUDA PyTorch wheel for NVIDIA GPU..."
            Write-Host "[INFO] GTX 1080 is Pascal; CUDA 11.8 wheels are the default for better old-driver compatibility."
            Write-Host "[INFO] Command: pip install torch torchvision torchaudio --index-url $TorchCudaIndexUrl"
            & $PythonExe -m pip install torch torchvision torchaudio --index-url $TorchCudaIndexUrl
        }
        else {
            Write-Host "[OK] CUDA PyTorch already available."
        }
    }
    else {
        Write-Host "[INFO] CPU mode selected. CUDA PyTorch install is skipped."
    }

    & $PythonExe -m pip install -r requirements-train.txt

    if (-not $Cpu) {
        $cudaReadyAfterInstall = Test-TorchCuda
        if (-not $cudaReadyAfterInstall) {
            Write-Host "[WARN] torch.cuda.is_available() is not True." -ForegroundColor Yellow
            Write-Host "[WARN] Training will probably fail with device=0 unless NVIDIA driver/CUDA wheel is fixed." -ForegroundColor Yellow
            Write-Host "[WARN] You can run CPU fallback with: .\train_1080.ps1 -Cpu" -ForegroundColor Yellow
        }
    }

    if (-not (Test-Path $YoloExe)) {
        throw "Ultralytics yolo.exe was not found after installation: $YoloExe"
    }

    Write-Host "[OK] Dependencies installed."
}

function Assert-TrainingInputs {
    Write-Section "Checking training inputs"

    $monsterCount = Count-Images "assets\monsters\raw"
    $itemCount = Count-Images "assets\items\raw"
    $backgroundCount = Count-Images "datasets\backgrounds"
    $realImageCount = Count-Images "datasets\real\images"

    $realLabelCount = 0
    if (Test-Path "datasets\real\labels") {
        $realLabelCount = (Get-ChildItem "datasets\real\labels" -Recurse -File -Include *.txt | Measure-Object).Count
    }

    $missingRealLabels = @(Get-MissingLabels "datasets\real\images" "datasets\real\labels")

    Write-Host "monster raw images : $monsterCount"
    Write-Host "item raw images    : $itemCount"
    Write-Host "background images  : $backgroundCount"
    Write-Host "real images        : $realImageCount"
    Write-Host "real labels        : $realLabelCount"
    Write-Host "missing real labels: $($missingRealLabels.Count)"

    if ($monsterCount -le 0) {
        throw "Missing monster assets. Expected images under assets\monsters\raw."
    }

    if ($itemCount -le 0) {
        throw "Missing item assets. Expected images under assets\items\raw."
    }

    if ($backgroundCount -le 0) {
        throw @"
Missing background screenshots.

Please put empty/low-target map screenshots here before running training:

  datasets\backgrounds\

Recommended first batch: 50 to 200 screenshots.
"@
    }

    if ($realImageCount -le 0) {
        throw @"
Missing real screenshots.

Please put real gameplay screenshots here:

  datasets\real\images\

And YOLO labels here:

  datasets\real\labels\

Validation/test should be real screenshots. Synthetic-only training is not recommended.
"@
    }

    if ($missingRealLabels.Count -gt 0) {
        $examples = ($missingRealLabels | Select-Object -First 10) -join "`n  "
        throw @"
Missing YOLO label files for real screenshots.

Every image under:

  datasets\real\images\

must have a matching .txt label under:

  datasets\real\labels\

Examples of missing label files:

  $examples

Important:
  - Images containing rope/platform/monster/item must be manually labeled.
  - Images containing none of the four classes are allowed, but still need an empty .txt file.
  - Synthetic images generated by tools\make_synthetic_dataset.py only auto-label monster/item as class 2/3.
"@
    }

    Write-Host "[OK] Required inputs found."
}

function Build-SyntheticDataset {
    Write-Section "Generating synthetic dataset"

    $syntheticImageCount = Count-Images "datasets\synthetic\images"

    if ($syntheticImageCount -ge $SyntheticCount -and -not $RebuildSynthetic) {
        $syntheticLabelsOk = Test-SyntheticLabelsUseRoboflowClassMapping "datasets\synthetic\labels"
        if (-not $syntheticLabelsOk) {
            throw @"
Existing synthetic labels do not match the current Roboflow class mapping.

Current class mapping is:

  0 Item
  1 Mob
  2 Platform
  3 Player
  4 Portal
  5 Rope

Synthetic labels should only contain class 0/1 because the synthetic generator only creates Item/Mob.

Please rebuild synthetic and YOLO datasets:

  .\run_train_auto.cmd -RebuildSynthetic -RebuildYolo

or on GTX 1080:

  .\run_train_1080.cmd -RebuildSynthetic -RebuildYolo
"@
        }

        Write-Host "[OK] Synthetic dataset already exists: $syntheticImageCount images"
        Write-Host "[INFO] Use -RebuildSynthetic to regenerate it."
        return
    }

    if ($RebuildSynthetic -and (Test-Path "datasets\synthetic")) {
        Write-Host "[INFO] Removing existing datasets\synthetic because -RebuildSynthetic was specified."
        Remove-Item "datasets\synthetic" -Recurse -Force
    }

    & $PythonExe "tools\make_synthetic_dataset.py" `
        --background-dir "datasets\backgrounds" `
        --monster-dir "assets\monsters\raw" `
        --item-dir "assets\items\raw" `
        --output-dir "datasets\synthetic" `
        --count $SyntheticCount `
        --seed $Seed

    if ($LASTEXITCODE -ne 0) {
        throw "Synthetic dataset generation failed."
    }

    Write-Host "[OK] Synthetic dataset generated."
}

function Build-YoloDataset {
    Write-Section "Building YOLO dataset"

    if ((Test-Path "datasets\yolo\data.yaml") -and -not $RebuildYolo) {
        $yoloDataYamlOk = Test-YoloDataYamlUseRoboflowClassMapping "datasets\yolo\data.yaml"
        if (-not $yoloDataYamlOk) {
            throw @"
Existing datasets\yolo\data.yaml does not use the current Roboflow class mapping.

Current class mapping is:

  0 Item
  1 Mob
  2 Platform
  3 Player
  4 Portal
  5 Rope

Please rebuild the YOLO dataset:

  .\run_train_auto.cmd -RebuildYolo

or on GTX 1080:

  .\run_train_1080.cmd -RebuildYolo
"@
        }

        Write-Host "[OK] YOLO dataset already exists: datasets\yolo\data.yaml"
        Write-Host "[INFO] Use -RebuildYolo to rebuild it."
        return
    }

    $cleanArgs = @()
    if ($RebuildYolo -or -not (Test-Path "datasets\yolo\data.yaml")) {
        $cleanArgs += "--clean"
    }

    & $PythonExe "tools\build_yolo_dataset.py" `
        --synthetic-dir "datasets\synthetic" `
        --real-dir "datasets\real" `
        --output-dir "datasets\yolo" `
        --seed $Seed `
        @cleanArgs

    if ($LASTEXITCODE -ne 0) {
        throw "YOLO dataset build failed."
    }

    if (-not (Test-Path "datasets\yolo\data.yaml")) {
        throw "YOLO data.yaml was not generated."
    }

    Write-Host "[OK] YOLO dataset is ready: datasets\yolo\data.yaml"
}

function Train-Model {
    if ($NoTrain) {
        Write-Host "[SKIP] Training skipped by -NoTrain"
        return
    }

    Write-Section "Starting YOLO training"

    $device = "0"
    if ($Cpu) {
        $device = "cpu"
    }

    $effectiveRunName = $RunName
    if ($RunName -eq "maple_yolov8n_640" -and $ImageSize -ne 640) {
        $effectiveRunName = "maple_yolov8n_$ImageSize"
    }

    & $YoloExe detect train `
        model=$Model `
        data="datasets/yolo/data.yaml" `
        imgsz=$ImageSize `
        epochs=$Epochs `
        patience=20 `
        batch=$Batch `
        device=$device `
        workers=$Workers `
        amp=False `
        flipud=0 `
        project="runs/detect" `
        name=$effectiveRunName `
        exist_ok=True

    if ($LASTEXITCODE -ne 0) {
        throw "YOLO training failed."
    }

    $bestPt = Join-Path $ProjectRoot "runs\detect\$effectiveRunName\weights\best.pt"
    if (-not (Test-Path $bestPt)) {
        throw "Training completed but best.pt was not found: $bestPt"
    }

    Write-Host "[OK] Training completed."
    Write-Host "best.pt: $bestPt"

    if ($NoExport) {
        Write-Host "[SKIP] ONNX export skipped by -NoExport"
        return
    }

    Write-Section "Exporting ONNX"

    & $YoloExe export `
        model=$bestPt `
        format=onnx `
        opset=12 `
        imgsz=$ImageSize `
        simplify=True

    if ($LASTEXITCODE -ne 0) {
        throw "ONNX export failed."
    }

    $bestOnnx = Join-Path $ProjectRoot "runs\detect\$effectiveRunName\weights\best.onnx"
    Write-Host "[OK] ONNX export completed."
    Write-Host "best.onnx: $bestOnnx"
}

Write-Section "MapleStoryAI one-click training for GTX 1080"
Write-Host "Project root     : $ProjectRoot"
Write-Host "Synthetic count  : $SyntheticCount"
Write-Host "Image size       : $ImageSize"
Write-Host "Epochs           : $Epochs"
Write-Host "Batch            : $Batch"
Write-Host "Workers          : $Workers"
Write-Host "Model            : $Model"
Write-Host "Torch CUDA index : $TorchCudaIndexUrl"
Write-Host "CPU mode         : $Cpu"

New-Venv
Install-Dependencies
Assert-TrainingInputs
Build-SyntheticDataset
Build-YoloDataset
Train-Model

Write-Section "All done"
Write-Host "If training finished successfully, check:"
Write-Host "  runs\detect\$RunName\weights\best.pt"
Write-Host "  runs\detect\$RunName\weights\best.onnx"
