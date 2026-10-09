$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$modelPath = Join-Path $taskRoot 'models\Model-7.6B-Q4_K_M.gguf'
while (-not (Test-Path -LiteralPath $modelPath)) { Start-Sleep -Seconds 10 }
$modelHash = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($modelHash -ne '4f635fc0c0bb88d50ccd9cf1f1e5892b5cb085ff88fe0d8e1148fd9a8a836bc2') { throw 'Model SHA256 mismatch' }
$projectorPath = Join-Path $taskRoot 'models\mmproj-model-f16.gguf'
$projectorHash = (Get-FileHash -LiteralPath $projectorPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($projectorHash -ne 'efa4f7d96aa0f838f2023fc8d28e519179b16f1106777fa9280b32628191aa3e') { throw 'Projector SHA256 mismatch' }
$serverPath = Join-Path $taskRoot 'runtime\vulkan\llama-server.exe'
$arguments = @('-m', $modelPath, '--mmproj', $projectorPath, '--host', '127.0.0.1', '--port', '18086', '-c', '12288', '-np', '1', '-ngl', 'all', '-b', '512', '-ub', '128', '-fa', 'on', '-ctk', 'f16', '-ctv', 'f16', '--no-warmup', '--fit', 'off', '--mtmd-batch-max-tokens', '256')
$logRoot = Join-Path $taskRoot 'results'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$arguments | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logRoot 'server_arguments.json') -Encoding utf8
$process = Start-Process -FilePath $serverPath -ArgumentList $arguments -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logRoot 'server.stdout.log') -RedirectStandardError (Join-Path $logRoot 'server.stderr.log')
$process.Id | Set-Content -LiteralPath (Join-Path $logRoot 'server.pid')
Write-Output "Started local server PID $($process.Id)"
