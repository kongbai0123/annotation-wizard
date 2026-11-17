param([switch]$CpuOnly, [string]$ModelSource, [switch]$WithDino, [string]$DinoSource)

$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython -PathType Leaf)) {
    & py -3.13 -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.13 is required.' }
}

& $projectPython -m pip install --disable-pip-version-check -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Could not install the annotation dependencies.' }
$build = if ($CpuOnly) { 'cpu' } else { 'cu128' }
& $projectPython -m pip install --disable-pip-version-check --index-url "https://download.pytorch.org/whl/$build" "torch==2.7.1+$build" "torchvision==0.22.1+$build"
if ($LASTEXITCODE -ne 0) { throw 'Could not install Torch and TorchVision.' }
& $projectPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'The Python environment has conflicting dependencies.' }

$modelArgs = @((Join-Path $PSScriptRoot 'prepare_model.py'))
if ($ModelSource) { $modelArgs += @('--from-dir', $ModelSource) }
& $projectPython @modelArgs
if ($LASTEXITCODE -ne 0) { throw 'Could not prepare the SAM 2.1 model.' }

if ($WithDino -or $DinoSource) {
    $dinoArgs = @((Join-Path $PSScriptRoot 'prepare_grounding.py'))
    if ($DinoSource) { $dinoArgs += @('--from-dir', $DinoSource) }
    & $projectPython @dinoArgs
    if ($LASTEXITCODE -ne 0) { throw 'Could not prepare the MM Grounding DINO model.' }
}

& (Join-Path $PSScriptRoot 'build-app.ps1')
