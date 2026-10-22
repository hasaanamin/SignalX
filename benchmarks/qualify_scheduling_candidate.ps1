param(
    [string]$OutputRoot = 'output/scheduling_fix_20261020/qualification',
    [string]$NumericalRoot = 'output/native_acquisition_fix_20261020/numerical'
)
# Run from the repository root, with no concurrent scheduled benchmark.
# Inputs/frozen scripts are copied, never repaired or overwritten by this runner.
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $OutputRoot) { throw 'Choose a new qualification directory' }
$taskRepoRoot = (Get-Location).Path
$taskQualification = (New-Item -ItemType Directory -Path $OutputRoot).FullName
$taskOutcomes = [ordered]@{}
function Run-Check {
    param([string]$Name, [string[]]$Arguments, [int[]]$Expected = @(0))
    & py -3.14 @Arguments *> (Join-Path $taskQualification ($Name+'.log'))
    $code = $LASTEXITCODE
    $taskOutcomes[$Name] = @{exit_code=$code; expected=$Expected; matched=($Expected -contains $code); arguments=$Arguments}
    Write-Output ($Name+' exit '+$code)
}
$env:SIGNAL_PROCESSING_SRC = $taskRepoRoot
Run-Check 'source' @('-m','pytest','-q','tests',"--junitxml=$taskQualification/source.xml")
Run-Check 'acceptance' @('-m','pytest','-q','checks',"--junitxml=$taskQualification/acceptance.xml")
$taskNumerical = (New-Item -ItemType Directory -Path (Join-Path $taskQualification 'numerical')).FullName
foreach ($name in @('signal_processing2_positive_validation','signal_processing2_feature_oracles','signal_processing2_filtered_and_lifecycle')) {
    $original = Join-Path $NumericalRoot ($name+'.py')
    if (-not (Test-Path -LiteralPath $original)) { throw ('Missing genuine current-equivalent numerical script: '+$original) }
    $copy = Join-Path $taskNumerical ($name+'.py')
    Copy-Item -LiteralPath $original -Destination $copy
    if ((Get-FileHash $original).Hash -ne (Get-FileHash $copy).Hash) { throw 'Numerical copy mismatch' }
    Run-Check $name @($copy)
    $observations = Get-Content ([IO.Path]::ChangeExtension($copy,'.json')) -Raw | ConvertFrom-Json
    foreach ($property in $observations.PSObject.Properties) {
        if ($property.Value.passed -eq $false) { throw ('Live numerical oracle failure: '+$property.Name) }
    }
}
Run-Check 'memory' @('-m','benchmarks.check_lte_broadcast_retention','--profile','low_snr','--output',"$taskQualification/memory.json")
Run-Check 'failing_sink' @('-m','benchmarks.stream_isolated','--seconds','1','--sample-rate','10000','--fail-sink','--progress-group','4','--acceptance-scope','processing','--output',"$taskQualification/failing_sink.json") @(1)
$taskFault = Get-Content (Join-Path $taskQualification 'failing_sink.json') -Raw | ConvertFrom-Json
if ($taskFault.sink_errors -lt 1 -or $taskFault.processing_scope_passed -ne $false -or -not $taskFault.recovered -or $taskFault.unresolved -ne 0) { throw 'Failure injection did not demonstrate the declared sink failure/recovery' }
Run-Check 'slow_sink' @('-m','benchmarks.stream_isolated','--seconds','3','--sample-rate','10000','--slow-sink','0.1','--progress-group','4','--acceptance-scope','processing','--output',"$taskQualification/slow_sink.json")
$taskRecovery = Get-Content (Join-Path $taskQualification 'slow_sink.json') -Raw | ConvertFrom-Json
if (-not $taskRecovery.processing_scope_passed -or -not $taskRecovery.recovered -or $taskRecovery.unfinished -ne 0) { throw 'Slow sink did not recover under the declared scope' }
$taskBuild = (New-Item -ItemType Directory -Path (Join-Path $taskQualification 'build_source')).FullName
$taskFiles = @(Get-ChildItem signal_processing -Recurse -File -Filter '*.py')
foreach ($file in $taskFiles) {
    if (-not $file.FullName.StartsWith($taskRepoRoot+'\')) { throw 'Production file outside repository' }
    $relative = $file.FullName.Substring($taskRepoRoot.Length+1)
    $destination = Join-Path $taskBuild $relative
    New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($destination)) | Out-Null
    Copy-Item -LiteralPath $file.FullName -Destination $destination
}
if (@(Get-ChildItem (Join-Path $taskBuild 'signal_processing') -Recurse -File -Filter '*.py').Count -ne $taskFiles.Count) { throw 'Build routing mismatch' }
Copy-Item -LiteralPath pyproject.toml,README.md -Destination $taskBuild
Run-Check 'build' @('-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',"$taskQualification/wheels",$taskBuild)
$taskWheels = @(Get-ChildItem (Join-Path $taskQualification 'wheels') -Filter '*.whl')
if ($taskWheels.Count -ne 1) { throw 'Expected exactly one fresh candidate wheel' }
Run-Check 'install' @('-m','pip','install','--no-deps','--target',"$taskQualification/installed",$taskWheels[0].FullName)
Run-Check 'installed_tests' @('-I','benchmarks/verify_lte_install.py',"$taskQualification/installed",$taskRepoRoot)
$taskFiles | ForEach-Object { [pscustomobject]@{path=$_.FullName.Substring($taskRepoRoot.Length+1); sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()} } | Export-Csv -NoTypeInformation (Join-Path $taskQualification 'production_manifest.csv')
$taskOutcomes | ConvertTo-Json -Depth 8 | Out-File -Encoding utf8 (Join-Path $taskQualification 'execution.json')
if (@($taskOutcomes.Values | Where-Object { -not $_.matched }).Count) { exit 1 }
# This is functional/build evidence, NOT scheduled acquisition qualification.
exit 0
