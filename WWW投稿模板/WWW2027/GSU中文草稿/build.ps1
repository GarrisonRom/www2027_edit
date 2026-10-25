$ErrorActionPreference = 'Stop'
$taskTexBin = 'C:\Users\RoboSAMA\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe'
$taskClass = Join-Path $PSScriptRoot '../ACM-acmart-template-v2.19/acmart-2.19'
$taskOldTexinputs = $env:TEXINPUTS
Push-Location -LiteralPath $PSScriptRoot
try {
    $env:TEXINPUTS = (Resolve-Path -LiteralPath $taskClass).Path + ';' + $taskOldTexinputs
    for ($taskPass = 1; $taskPass -le 2; $taskPass++) {
        & $taskTexBin -interaction=nonstopmode -halt-on-error -file-line-error main.tex
        if ($LASTEXITCODE -ne 0) { throw "pdfLaTeX failed on pass $taskPass" }
    }
} finally {
    $env:TEXINPUTS = $taskOldTexinputs
    Pop-Location
}
