param(
    [Parameter(Mandatory=$true)][string]$Method,
    [string]$Model='E:/12models/modelscope/Qwen2.5-1.5B-Instruct',
    [string]$Python='D:/Anaconda/envs/cuda126_env/python.exe',
    [string]$RunDir,
    [string]$Config,
    [switch]$Smoke,
    [switch]$Plan,
    [switch]$Generation,
    [switch]$MmluFull,
    [switch]$LoraAdapted,
    [switch]$TrainOnly
)
$ErrorActionPreference='Stop'
if (-not (Test-Path -LiteralPath $Python)) { throw "Missing interpreter: $Python" }
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $baselineArgs=@('-B','-m','baselines.train','--method',$Method)
    if ($LoraAdapted) { $baselineArgs+='--lora-adapted' }
    if ($Plan) { $baselineArgs+='--plan' }
    else {
        if (-not $RunDir) { $RunDir=Join-Path $PSScriptRoot ('runs/'+$Method+'_'+(Get-Date -Format 'yyyyMMdd_HHmmss')) }
        $baselineArgs+=@('--model',$Model,'--run-dir',$RunDir)
        if ($Config) { $baselineArgs+=@('--config',$Config) }
        if ($Smoke) { $baselineArgs+='--smoke' }
        if ($Generation) { $baselineArgs+='--generation' }
        if ($MmluFull) { $baselineArgs+='--mmlu-full' }
        if ($TrainOnly) { $baselineArgs+='--train-only' }
    }
    & $Python @baselineArgs
    if ($LASTEXITCODE -ne 0) { throw "Baseline exited with code $LASTEXITCODE" }
} finally { Pop-Location }
