[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet('infer','train','probe')][string]$Stage,
    [Alias('Input')][string]$InputDirectory,
    [Parameter(Mandatory=$true)][string]$Destination,
    [ValidateRange(1,1000)][int]$Epochs = 30,
    [string]$ProtectedReferences,
    [ValidateRange(1,86400)][int]$TimeoutSeconds
)

$ErrorActionPreference = 'Stop'
$Stage = $Stage.ToLowerInvariant()
$root = $PSScriptRoot

function Get-LocalSha256([string]$Path) {
    $algorithm = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Path)
    try {
        return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '').ToLowerInvariant()
    } finally {
        $stream.Dispose()
        $algorithm.Dispose()
    }
}

function Get-RegularEntry([string]$Path, [bool]$Directory) {
    $entry = Get-Item -LiteralPath $Path -Force
    if ($entry.PSIsContainer -ne $Directory -or
        ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Expected a regular local input: $Path"
    }
    return $entry
}

function Test-ContainedPath([string]$Path, [string]$Directory) {
    $prefix = $Directory.TrimEnd([char[]]@('\','/')) + [IO.Path]::DirectorySeparatorChar
    return $Path.Equals($Directory, [StringComparison]::OrdinalIgnoreCase) -or
           $Path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
}

function Copy-StageDirectory([string]$Source, [string]$Target, [string[]]$Extensions) {
    $directory = Get-RegularEntry $Source $true
    New-Item -ItemType Directory -Path $Target | Out-Null
    foreach ($child in Get-ChildItem -LiteralPath $directory.FullName -Force) {
        if ($child.PSIsContainer) {
            Copy-StageDirectory $child.FullName (Join-Path $Target $child.Name) $Extensions
        } else {
            $file = Get-RegularEntry $child.FullName $false
            if ($file.Extension.ToLowerInvariant() -notin $Extensions) {
                throw "Unexpected input file type: $($file.Name)"
            }
            Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $Target $file.Name)
        }
    }
}

$destinationPath = (Get-RegularEntry $Destination $true).FullName
if (Get-ChildItem -LiteralPath $destinationPath -Force | Select-Object -First 1) {
    throw 'The job result directory must be empty.'
}
if ($Stage -ne 'train' -and $PSBoundParameters.ContainsKey('Epochs')) {
    throw 'Epochs applies to the training stage.'
}
if (-not $TimeoutSeconds) {
    $TimeoutSeconds = switch ($Stage) {
        'probe' { 90 }
        'infer' { 7200 }
        'train' { 21600 }
    }
}

$listeners = [System.Collections.Generic.List[Net.Sockets.TcpListener]]::new()
$work = $null
$canary = $null
$failure = $null
try {
    $isolation = if ($env:ANNOTATION_WIZARD_ISOLATION_HOME) {
        $env:ANNOTATION_WIZARD_ISOLATION_HOME
    } else {
        Join-Path $root 'training\door-box-sandbox-20261117\isolation'
    }
    $isolation = (Get-RegularEntry $isolation $true).FullName
    $runtime = (Get-RegularEntry (Join-Path $isolation 'runtime') $true).FullName
    $python = (Get-RegularEntry (Join-Path $runtime 'python.exe') $false).FullName
    $source = (Get-RegularEntry (Join-Path $root 'launcher\IsolatedProcess.cs') $false).FullName
    $profile = switch ($Stage) {
        'probe' { 'AnnotationWizardGenericProbeV1' }
        'infer' { 'AnnotationWizardGenericInferenceV1' }
        'train' { 'AnnotationWizardGenericTrainingV1' }
    }
    Add-Type -Path $source
    $sid = [IsolatedProcess]::ProfileSid($profile)
    $exemptions = & CheckNetIsolation.exe LoopbackExempt -s
    if ($LASTEXITCODE -ne 0 -or ($exemptions -join "`n").Contains($sid)) {
        throw 'The AppContainer loopback exemption check failed.'
    }

    $tempRoot = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) "Packages\$profile\AC\Temp"
    $work = Join-Path $tempRoot "run-$([Guid]::NewGuid().ToString('N'))"
    New-Item -ItemType Directory -Path $work -Force | Out-Null
    if ($Stage -ne 'probe') {
        $live = @{stage=$Stage; run_folder=(Split-Path -Leaf $work)} | ConvertTo-Json -Compress
        [IO.File]::WriteAllText((Join-Path $destinationPath 'live.json'), $live)
    }
    Copy-Item -LiteralPath (Join-Path $root 'generic_isolation_probe.py') -Destination $work
    $arguments = @()
    $stagedManifest = $null
    $access = @($runtime)
    $privatePaths = [System.Collections.Generic.List[string]]::new()
    $canary = Join-Path $destinationPath ".host-only-$([Guid]::NewGuid().ToString('N')).txt"
    [IO.File]::WriteAllText($canary, 'Generic job host-only ACL probe.')
    $privatePaths.Add($canary)

    if ($Stage -ne 'probe') {
        if (-not $InputDirectory) { throw 'An input directory is required for inference or training.' }
        $inputPath = (Get-RegularEntry $InputDirectory $true).FullName
        if ((Test-ContainedPath $destinationPath $inputPath) -or (Test-ContainedPath $inputPath $destinationPath)) {
            throw 'Keep the job input and result directories separate.'
        }
        $manifestName = if ($Stage -eq 'infer') { 'manifest.json' } else { 'dataset.json' }
        $inputManifest = (Get-RegularEntry (Join-Path $inputPath $manifestName) $false).FullName
        $privatePaths.Add($inputManifest)
        $stagedInput = Join-Path $work 'input'
        New-Item -ItemType Directory -Path $stagedInput | Out-Null
        $stagedManifest = Join-Path $stagedInput $manifestName
        Copy-Item -LiteralPath $inputManifest -Destination $stagedManifest
        Copy-StageDirectory (Join-Path $inputPath 'images') (Join-Path $stagedInput 'images') @('.png','.jpg','.jpeg')

        $modelRoot = if ($env:ANNOTATION_WIZARD_MODELS_HOME) {
            (Get-RegularEntry $env:ANNOTATION_WIZARD_MODELS_HOME $true).FullName
        } else {
            Join-Path $root 'models'
        }
        $dino = Join-Path $modelRoot 'MM-Grounding-DINO-Tiny'
        $qwen = Join-Path $modelRoot 'Qwen3-VL-4B-Instruct'
        if ($Stage -eq 'infer') {
            $access += @((Get-RegularEntry $dino $true).FullName, (Get-RegularEntry $qwen $true).FullName)
            $scripts = @('generic_inference.py', 'generic_confidence.py', 'generic_qc.py', 'grounding.py')
            $arguments = @((Join-Path $work 'generic_inference.py'), '--manifest', $stagedManifest,
                '--dino-dir', $dino, '--qwen-dir', $qwen, '--output', (Join-Path $work 'inference.json'))
            $jobConfig = (Get-Content -Raw -Encoding utf8 -LiteralPath $stagedManifest | ConvertFrom-Json).config
            if ($jobConfig.settings.shadow_sam -is [bool] -and $jobConfig.settings.shadow_sam) {
                $samSource = (Get-RegularEntry (Join-Path $modelRoot 'sam2.1-hiera-tiny') $true).FullName
                $sam = Join-Path $work 'sam'
                New-Item -ItemType Directory -Path $sam | Out-Null
                foreach ($file in @('config.json','model.safetensors','preprocessor_config.json','processor_config.json')) {
                    $entry = Get-RegularEntry (Join-Path $samSource $file) $false
                    Copy-Item -LiteralPath $entry.FullName -Destination $sam
                }
                $scripts += @('segmentation.py', 'mask_cleanup.py', 'prepare_model.py')
                $arguments += @('--sam-dir', $sam)
            }
        } else {
            Copy-StageDirectory (Join-Path $inputPath 'labels') (Join-Path $stagedInput 'labels') @('.txt')
            $scripts = @('demo\auto_yolo_worker.py', 'generic_eval.py')
            foreach ($model in @((Join-Path $dino 'model.safetensors'),
                                  (Join-Path $qwen 'model-00001-of-00002.safetensors'))) {
                if (Test-Path -LiteralPath $model -PathType Leaf) { $privatePaths.Add($model) }
            }
            $arguments = @((Join-Path $work 'auto_yolo_worker.py'), '--dataset', $stagedInput,
                '--output', (Join-Path $work 'training'), '--isolation', (Join-Path $work 'isolation.json'),
                '--epochs', "$Epochs", '--evaluate-checkpoint')
        }
        foreach ($script in $scripts) {
            Copy-Item -LiteralPath (Join-Path $root $script) -Destination $work
        }
    }

    if ($ProtectedReferences) {
        $inventory = (Get-RegularEntry $ProtectedReferences $false).FullName
        if ($inputPath -and (Test-ContainedPath $inventory $inputPath)) {
            throw 'The reference inventory must be outside the input copied to the worker.'
        }
        $privatePaths.Add($inventory)
        $entries = Get-Content -Raw -Encoding utf8 -LiteralPath $inventory | ConvertFrom-Json
        foreach ($path in $entries) {
            if ($path -isnot [string] -or -not [IO.Path]::IsPathRooted($path)) {
                throw 'Protected reference paths must be absolute local file names.'
            }
            $entry = Get-RegularEntry $path $false
            if ($inputPath -and (Test-ContainedPath $entry.FullName $inputPath)) {
                throw 'Protected references must be outside the input copied to the worker.'
            }
            $privatePaths.Add($entry.FullName)
        }
    }
    foreach ($path in $privatePaths) {
        $opened = [IO.File]::OpenRead($path)
        $opened.Dispose()
    }
    foreach ($path in $access) {
        & icacls.exe $path /grant:r "*${sid}:(OI)(CI)RX" /Q | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Unable to grant AppContainer read access: $path" }
    }

    $env:TILE_PROFILE = $profile
    $env:TEMP = $work
    $env:TMP = $work
    $env:HF_HOME = Join-Path $work 'hf'
    $env:TORCH_HOME = Join-Path $work 'torch'
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    $env:HF_HUB_DISABLE_TELEMETRY = '1'
    $env:USE_HUB_KERNELS = 'NO'
    $env:HF_DATASETS_OFFLINE = '1'
    $env:YOLO_OFFLINE = 'true'
    $env:YOLO_AUTOINSTALL = 'false'
    $env:WANDB_MODE = 'disabled'
    $addresses = [ordered]@{loopback_ipv4=[Net.IPAddress]::Loopback}
    if ([Net.Sockets.Socket]::OSSupportsIPv6) { $addresses['loopback_ipv6'] = [Net.IPAddress]::IPv6Loopback }
    $adapter = Get-NetIPAddress -AddressFamily IPv4 -AddressState Preferred |
        Where-Object { $_.PrefixOrigin -in 'Dhcp','Manual' -and -not $_.IPAddress.StartsWith('127.') } |
        Select-Object -First 1
    if ($adapter) { $addresses['host_adapter'] = [Net.IPAddress]::Parse($adapter.IPAddress) }
    $probe = @{private_files=$privatePaths.ToArray(); listeners=@()
        launcher_sha256=(Get-LocalSha256 $PSCommandPath)
        token_launcher_sha256=(Get-LocalSha256 $source)}
    if ($stagedManifest) {
        $probe['manifest_sha256'] = Get-LocalSha256 $stagedManifest
    }
    foreach ($name in $addresses.Keys) {
        $address = $addresses[$name]
        $listener = [Net.Sockets.TcpListener]::new($address, 0)
        $listener.Start()
        $listeners.Add($listener)
        $port = $listener.LocalEndpoint.Port
        $control = [Net.Sockets.TcpClient]::new($address.AddressFamily)
        try {
            $control.Connect($address, $port)
            $listener.AcceptTcpClient().Dispose()
        } finally {
            $control.Dispose()
        }
        $probe.listeners += @{name=$name; ip=$address.ToString(); port=$port}
    }
    $env:AW_GENERIC_PROBE = $probe | ConvertTo-Json -Compress -Depth 5
    $tokens = @($python, '-I', '-B', '-X', 'utf8', (Join-Path $work 'generic_isolation_probe.py'), $Stage) + $arguments
    $command = ($tokens | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $exitCode = [IsolatedProcess]::Run($profile, $python, $command, $work, $TimeoutSeconds)
    if ($exitCode -ne 0) { throw "The isolated $Stage worker exited with code $exitCode." }
    $proof = Get-Content -Raw -Encoding utf8 -LiteralPath (Join-Path $work 'isolation.json') | ConvertFrom-Json
    if ($proof.schema -ne 'annotation_wizard.generic_isolation.v1' -or $proof.stage -ne $Stage -or
        $proof.profile -ne $profile -or $proof.appcontainer_token -ne $true -or
        $null -eq $proof.os_network_capabilities -or $proof.os_network_capabilities -ne 0 -or
        $proof.network_denied -ne $true -or $proof.labels_denied -ne $true -or
        ($Stage -ne 'probe' -and $proof.cuda -ne $true)) {
        throw 'Isolation proof does not validate. No model result can be published.'
    }
    if ($Stage -eq 'infer') {
        Copy-Item -LiteralPath (Join-Path $work 'inference.json') -Destination $destinationPath
    } elseif ($Stage -eq 'train') {
        Copy-Item -LiteralPath (Join-Path $work 'training') -Destination $destinationPath -Recurse
    }
    Write-Output "AppContainer $Stage complete. Results: $destinationPath"
} catch {
    $failure = $_.Exception.Message
    throw
} finally {
    foreach ($listener in $listeners) { $listener.Stop() }
    foreach ($name in @('isolation.json','worker.log')) {
        if ($work -and (Test-Path -LiteralPath (Join-Path $work $name) -PathType Leaf)) {
            Copy-Item -LiteralPath (Join-Path $work $name) -Destination $destinationPath
        }
    }
    if ($failure) {
        Add-Content -LiteralPath (Join-Path $destinationPath 'worker.log') -Encoding utf8 -Value "Launcher error: $failure"
    }
    if ($canary -and (Test-Path -LiteralPath $canary -PathType Leaf)) {
        Remove-Item -LiteralPath $canary
    }
    if ($work -and (Test-Path -LiteralPath $work -PathType Container)) {
        $resolved = (Resolve-Path -LiteralPath $work).Path
        $expectedParent = (Resolve-Path -LiteralPath $tempRoot).Path
        if ((Split-Path -Parent $resolved) -eq $expectedParent -and
            (Split-Path -Leaf $resolved) -match '^run-[a-f0-9]{32}$' -and
            -not (Get-Item -LiteralPath $resolved).Attributes.HasFlag([IO.FileAttributes]::ReparsePoint)) {
            Remove-Item -LiteralPath $resolved -Recurse -Force -ErrorAction Continue
        }
    }
}
