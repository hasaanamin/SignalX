param(
    [Parameter(Mandatory=$true)][string]$AdaptiveInput,
    [Parameter(Mandatory=$true)][string]$LTEInput,
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [string]$NoCellInput,
    [string]$TransitionInput,
    [string]$ActiveInput,
    [string]$ToneInput,
    [int]$Repetitions = 3
)
# Keep timed DSP separate from builds, profiling, editing and regression suites.
$ErrorActionPreference = 'Stop'
if ($Repetitions -lt 1) { throw 'Repetitions must be positive' }
if (Test-Path -LiteralPath $OutputRoot) { throw 'Choose a new output directory' }
$taskTimingRoot = (New-Item -ItemType Directory -Path $OutputRoot).FullName
$taskResults = @()
$env:OPENBLAS_NUM_THREADS = '1'
for ($attempt = 1; $attempt -le $Repetitions; $attempt++) {
    $modes = @('adaptive','lte')
    if ($NoCellInput) { $modes += 'no_cell' }
    if ($TransitionInput) { $modes += 'transition' }
    if ($ActiveInput) { $modes += 'adaptive_active' }
    if ($ToneInput) { $modes += 'tone' }
    foreach ($mode in $modes) {
        $inputPath = if ($mode -eq 'adaptive') { $AdaptiveInput } elseif ($mode -eq 'lte') { $LTEInput } elseif ($mode -eq 'no_cell') { $NoCellInput } elseif ($mode -eq 'adaptive_active') { $ActiveInput } elseif ($mode -eq 'tone') { $ToneInput } else { $TransitionInput }
        $name = $mode+'_'+$attempt
        $resultPath = Join-Path $taskTimingRoot ($name+'.json')
        $logPath = Join-Path $taskTimingRoot ($name+'.log')
        if ($mode -eq 'transition') {
            & py -3.14 -m benchmarks.qualify_lte_transition run --priority-class high --input $inputPath --output $resultPath *> $logPath
        } else {
            & py -3.14 -m benchmarks.qualify_realtime_1920 $mode --input $inputPath --output $resultPath *> $logPath
        }
        $taskExit = $LASTEXITCODE
        $taskResults += [pscustomobject]@{name=$name;exit_code=$taskExit;result=$resultPath}
        Write-Output ($name+' exit '+$taskExit)
    }
}
$taskResults | ConvertTo-Json | Out-File -Encoding utf8 (Join-Path $taskTimingRoot 'execution.json')
if (@($taskResults | Where-Object { $_.exit_code -ne 0 }).Count) { exit 1 }
exit 0
