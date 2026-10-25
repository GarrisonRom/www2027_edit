#requires -Version 7.0
[CmdletBinding()]
param(
    [string]$PythonPath = 'D:\Anaconda\envs\cuda126_env\python.exe',
    [string]$ModelPath = 'E:\12models\modelscope\Qwen2.5-1.5B-Instruct',
    [string]$BatchId,
    [switch]$DryRun,
    [switch]$PreflightOnly,
    [switch]$Worker
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$Workspace = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$ExperimentRoot = Join-Path $Workspace 'exp/runs/exp007_gsu_integrated_guidance'
$Trainer = Join-Path $PSScriptRoot 'diagnose_gsu_integrated_guidance.py'
$Variants = @('B0', 'B1', 'B2', 'B3', 'B2R', 'B3R')

function Get-LocalTimestamp {
    [DateTimeOffset]::UtcNow.ToOffset([TimeSpan]::FromHours(8)).ToString('o')
}

function Write-JsonFile {
    param([string]$Path, [object]$Value, [switch]$Exclusive)
    $json = ConvertTo-Json -InputObject $Value -Depth 40
    if ($Exclusive -and (Test-Path -LiteralPath $Path)) { throw "Refusing to overwrite $Path" }
    [IO.File]::WriteAllText($Path, $json, [Text.UTF8Encoding]::new($false))
}

function Quote-NativeArgument {
    param([string]$Value)
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

function Get-FrozenFiles {
    $relativePaths = @(
        'exp/train/start_gsu_integrated_guidance.ps1',
        'exp/train/diagnose_gsu_integrated_guidance.py',
        'exp/train/diagnose_gsu_memory_guided.py',
        'exp/train/diagnose_gsu_actual_feedback.py',
        'exp/train/diagnose_gsu_retain_protection.py',
        'exp/train/diagnose_gsu_m3.py',
        'exp/train/train_gsu.py',
        'exp/train/train_eval.py',
        'exp/train/tests/test_gsu_integrated_guidance.py',
        'exp/train/tests/test_gsu_memory_guided.py',
        'exp/damage/component_ablation.py',
        'exp/damage/eval_per_item.py',
        'docs/method_versions/GSU-v7-integrated-damage-gradient-path.md',
        'docs/method_versions/GSU-v6-memory-guided.md',
        'exp/runs/exp007_gsu_integrated_guidance/card.md',
        'exp/runs/exp007_gsu_integrated_guidance/README.md',
        'exp/runs/exp006_gsu_memory_guided/card.md',
        'exp/runs/exp006_gsu_memory_guided/README.md',
        'data/faithun/FaithUn_Dataset.json',
        'data/faithun/splits/forget_set_split.txt',
        'data/faithun/splits/retaining_set_split.txt',
        'data/faithun/splits/test_set_split.txt',
        'exp/results/F1.json',
        'exp/results/candidates.json',
        'exp/results/active_paths.json',
        'exp/results/attribution/items.json',
        'exp/results/per_item_G2.jsonl',
        'exp/results/attribution/ffn_G1.json',
        'exp/results/attribution/ffn_G2.json',
        'exp/results/attribution/stats_stage2.json',
        'exp/results/attribution/ablation_G2.jsonl'
    )
    foreach ($relativePath in $relativePaths) {
        $path = Join-Path $Workspace $relativePath
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Frozen input is missing: $path" }
        [ordered]@{ path = $relativePath; sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
    }
}

function Assert-FrozenFiles {
    param([object[]]$Files)
    foreach ($file in $Files) {
        $path = Join-Path $Workspace $file.path
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Frozen input disappeared: $path" }
        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
        if ($actual -ine $file.sha256) { throw "Frozen input changed; stop before the next variant: $path" }
    }
}

foreach ($required in @($PythonPath, $Trainer, (Join-Path $ModelPath 'config.json'))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Required file not found: $required" }
}
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
$ModelPath = (Resolve-Path -LiteralPath $ModelPath).Path

if (-not $BatchId) {
    $stamp = [DateTimeOffset]::UtcNow.ToOffset([TimeSpan]::FromHours(8)).ToString('yyyyMMdd_HHmmss')
    $BatchId = 'batch_' + $stamp + '_' + [Guid]::NewGuid().ToString('N').Substring(0, 6)
}
if ($BatchId -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$') { throw 'Invalid BatchId.' }
$BatchRoot = Join-Path $ExperimentRoot "batches/$BatchId"
$StatusPath = Join-Path $BatchRoot 'status.json'

if (-not $Worker) {
    $jobs = @(
        foreach ($variant in $Variants) {
            $run = Join-Path $ExperimentRoot "runs/run_${BatchId}_${variant}"
            if (Test-Path -LiteralPath $run) { throw "Run already exists: $run" }
            [ordered]@{ variant = $variant; run_dir = $run; updates = 114 }
        }
    )
    $comparison = Join-Path $ExperimentRoot "comparisons/screen_$BatchId"
    if (Test-Path -LiteralPath $comparison) { throw "Comparison already exists: $comparison" }
    $manifest = [ordered]@{
        experiment = 'exp007_gsu_integrated_guidance'
        method_version = 'GSU-v7-integrated-damage-gradient-path'
        batch_id = $BatchId; created = Get-LocalTimestamp; workspace = $Workspace
        python = $PythonPath; model = $ModelPath; seed = 42
        updates_per_variant = 114; planned_total_updates = 684
        variant_order = $Variants; jobs = $jobs; comparison_dir = $comparison
        failure_policy = 'stop immediately; no retry, overwrite, hyperparameter changes or extra seed'
        frozen_files = @(Get-FrozenFiles)
    }
    if ($DryRun) { $manifest | ConvertTo-Json -Depth 40; return }
    if ($PreflightOnly) {
        $tmpBatch = Join-Path ([IO.Path]::GetTempPath()) $BatchId
        if (Test-Path -LiteralPath $tmpBatch) { throw "Temporary preflight path exists: $tmpBatch" }
        $null = New-Item -ItemType Directory -Path $tmpBatch
        try {
            $preflightArgs = @('-u','-B',$Trainer,'--preflight','--model',$ModelPath)
            & $PythonPath @preflightArgs
            if ($LASTEXITCODE -ne 0) { throw "Preflight failed with exit code $LASTEXITCODE" }
        } finally {
            Remove-Item -LiteralPath $tmpBatch -Recurse -Force
        }
        return
    }
    if (Test-Path -LiteralPath $BatchRoot) { throw "Batch already exists: $BatchRoot" }
    $null = New-Item -ItemType Directory -Path $BatchRoot
    Write-JsonFile -Path (Join-Path $BatchRoot 'config.json') -Value $manifest -Exclusive
    [IO.File]::Copy($PSCommandPath, (Join-Path $BatchRoot 'launcher.ps1.snapshot'), $false)
    Write-JsonFile -Path $StatusPath -Value ([ordered]@{
        batch_id = $BatchId; status = 'queued'; created = $manifest.created
        completed_variants = @(); current_variant = $null
    }) -Exclusive
    $shellPath = (Get-Process -Id $PID).Path
    $launchArgs = @('-NoLogo','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass',
        '-File',$PSCommandPath,'-Worker','-BatchId',$BatchId,'-PythonPath',$PythonPath,'-ModelPath',$ModelPath)
    $launchLine = ($launchArgs | ForEach-Object { Quote-NativeArgument $_ }) -join ' '
    try {
        $controller = Start-Process -FilePath $shellPath -ArgumentList $launchLine `
            -WorkingDirectory $Workspace -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $BatchRoot 'controller.stdout.log') `
            -RedirectStandardError (Join-Path $BatchRoot 'controller.stderr.log')
        Write-JsonFile -Path (Join-Path $BatchRoot 'launch.json') -Value ([ordered]@{
            controller_pid = $controller.Id; launched = Get-LocalTimestamp
            shell = $shellPath; arguments = $launchArgs
        }) -Exclusive
        [ordered]@{ batch_id=$BatchId; controller_pid=$controller.Id; batch_dir=$BatchRoot; status_file=$StatusPath } | ConvertTo-Json
    } catch {
        Write-JsonFile -Path $StatusPath -Value ([ordered]@{batch_id=$BatchId;status='failed';ended=Get-LocalTimestamp;error=$_.Exception.Message;phase='dispatch'})
        throw
    }
    return
}

$manifestPath = Join-Path $BatchRoot 'config.json'
$manifest = Get-Content -Raw -Encoding utf8 -LiteralPath $manifestPath | ConvertFrom-Json
if ($manifest.python -ne $PythonPath -or $manifest.model -ne $ModelPath) { throw 'Worker paths differ from frozen batch config.' }
$queued = Get-Content -Raw -Encoding utf8 -LiteralPath $StatusPath | ConvertFrom-Json
if ($queued.status -ne 'queued') { throw 'This batch was already started.' }
$marker = [IO.File]::Open((Join-Path $BatchRoot 'worker.started'),'CreateNew','Write','None'); $marker.Dispose()
$state = [ordered]@{batch_id=$BatchId;status='running';started=Get-LocalTimestamp;controller_pid=$PID;phase='startup';current_variant=$null;current_run_dir=$null;completed_variants=@();last_exit_code=$null}
$mutex = [Threading.Mutex]::new($false,'Local\GSU_exp007_IntegratedGuidance')
$hasMutex = $false

function Write-BatchEvent {
    param([string]$Event)
    $row=[ordered]@{time=Get-LocalTimestamp;event=$Event;phase=$state.phase;variant=$state.current_variant;run_dir=$state.current_run_dir}
    Add-Content -LiteralPath (Join-Path $BatchRoot 'events.jsonl') -Value (ConvertTo-Json -InputObject $row -Compress) -Encoding utf8
    Write-JsonFile -Path $StatusPath -Value $state
}

function Invoke-PythonPhase {
    param([string]$Name,[string[]]$NativeArgs)
    $state.phase=$Name; Write-BatchEvent "started:$Name"
    $out=Join-Path $BatchRoot "$Name.stdout.log"; $err=Join-Path $BatchRoot "$Name.stderr.log"
    & $PythonPath @NativeArgs 1>> $out 2>> $err
    $state.last_exit_code=$LASTEXITCODE
    if ($LASTEXITCODE -ne 0) { throw "Phase $Name failed ($LASTEXITCODE); inspect $err" }
    Write-BatchEvent "completed:$Name"
}

try {
    try { $hasMutex=$mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $hasMutex=$true }
    if (-not $hasMutex) { throw 'Another exp007 batch is active.' }
    $active=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'diagnose_gsu_integrated_guidance\.py.+--variant' })
    if ($active.Count -gt 0) { throw "An exp007 training process is already active: $($active.ProcessId -join ', ')" }
    Set-Location -LiteralPath $Workspace
    $env:PYTHONIOENCODING='utf-8'; $env:PYTHONUNBUFFERED='1'; $env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'
    Assert-FrozenFiles $manifest.frozen_files
    $gpuCheck='import json,sys,torch; ok=torch.cuda.is_available(); bf=ok and torch.cuda.is_bf16_supported(); print(json.dumps(dict(cuda_available=ok,bf16_supported=bf,torch=torch.__version__,gpu=torch.cuda.get_device_name(0) if ok else None))); sys.exit(0 if bf else 2)'
    Invoke-PythonPhase -Name 'environment' -NativeArgs @('-u','-B','-c',$gpuCheck)
    Invoke-PythonPhase -Name 'self_test' -NativeArgs @('-u','-B',$Trainer,'--self_test')
    Invoke-PythonPhase -Name 'unit_tests' -NativeArgs @('-u','-B','-m','unittest','discover','-s','exp/train/tests','-p','test_gsu_integrated_guidance.py','-v')
    Invoke-PythonPhase -Name 'preflight' -NativeArgs @('-u','-B',$Trainer,'--preflight','--model',$ModelPath)
    foreach ($job in $manifest.jobs) {
        Assert-FrozenFiles $manifest.frozen_files
        $state.current_variant=$job.variant; $state.current_run_dir=$job.run_dir
        Invoke-PythonPhase -Name $job.variant -NativeArgs @('-u','-B',$Trainer,'--variant',$job.variant,'--model',$ModelPath,'--run_dir',$job.run_dir)
        $runStatus=Get-Content -Raw -Encoding utf8 -LiteralPath (Join-Path $job.run_dir 'status.json') | ConvertFrom-Json
        if ($runStatus.status -ne 'completed') { throw "Run incomplete: $($job.run_dir)" }
        $state.completed_variants=@($state.completed_variants)+@($job.variant)
        Write-BatchEvent "verified:$($job.variant)"
    }
    Assert-FrozenFiles $manifest.frozen_files
    $state.current_variant=$null; $state.current_run_dir=$null
    $compareArgs=@('-u','-B',$Trainer,'--compare')
    $compareArgs+=@($manifest.jobs | ForEach-Object { $_.run_dir })
    $compareArgs+=@('--comparison_dir',$manifest.comparison_dir)
    Invoke-PythonPhase -Name 'comparison' -NativeArgs $compareArgs
    $state.status='completed'; $state.phase='completed'; $state['comparison_dir']=$manifest.comparison_dir; $state['ended']=Get-LocalTimestamp
    Write-BatchEvent 'batch_completed'
} catch {
    $state.status='failed'; $state['error']=$_.Exception.Message; $state['ended']=Get-LocalTimestamp
    Write-BatchEvent 'batch_failed'; Write-Error -Message $_.Exception.Message -ErrorAction Continue
    exit 1
} finally {
    if ($hasMutex) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
