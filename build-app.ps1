$ErrorActionPreference = 'Stop'
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path -LiteralPath $compiler -PathType Leaf)) { throw 'Windows .NET Framework 4.x compiler is unavailable.' }

$source = Join-Path $PSScriptRoot 'launcher\AnnotationWizard.cs'
$target = Join-Path $PSScriptRoot 'AnnotationWizard.exe'
$parameters = @('/nologo', '/target:winexe', '/optimize+', '/platform:anycpu', '/codepage:65001',
                '/reference:System.Windows.Forms.dll', "/out:$target", $source)
$icon = Join-Path $PSScriptRoot 'launcher\annotation.ico'
if (Test-Path -LiteralPath $icon -PathType Leaf) { $parameters += "/win32icon:$icon" }
& $compiler @parameters
if ($LASTEXITCODE -ne 0) { throw 'Could not build AnnotationWizard.exe.' }
Write-Host "Built $target"
