param(
    [int]$Runs = 5,
    [string]$Exe = ".\build-m5-gpu\gfss_gpu_m5_fastsetup_persistent_solver_bench.exe",
    [int]$InnerIterations = 5,
    [int]$MaxOuter = 8,
    [int]$BlockY = 4,
    [int]$TargetNodes = 12,
    [int]$MinNodes = 4,
    [double]$OuterTolerance = 1e-6,
    [string]$CsvPath = ""
)

$ErrorActionPreference = "Stop"

if ($Runs -lt 3) {
    throw "Runs must be >= 3; use at least 5 for optimization decisions."
}
if (-not (Test-Path -LiteralPath $Exe)) {
    throw "Benchmark executable not found: $Exe"
}

function Get-Scalar([string]$Text, [string]$Name) {
    $pattern = [regex]::Escape($Name) + '=([-+0-9.eE]+)'
    $m = [regex]::Match($Text, $pattern)
    if (-not $m.Success) {
        throw "Missing metric '$Name' in benchmark output."
    }
    return [double]::Parse($m.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
}

function Get-Bool([string]$Text, [string]$Name) {
    $pattern = [regex]::Escape($Name) + '=(true|false)'
    $m = [regex]::Match($Text, $pattern)
    if (-not $m.Success) {
        throw "Missing boolean '$Name' in benchmark output."
    }
    return $m.Groups[1].Value -eq 'true'
}

function Get-Median([double[]]$Values) {
    $sorted = @($Values | Sort-Object)
    $n = $sorted.Count
    if (($n % 2) -eq 1) {
        return [double]$sorted[[int](($n - 1) / 2)]
    }
    return 0.5 * ([double]$sorted[$n / 2 - 1] + [double]$sorted[$n / 2])
}

function Get-Mean([double[]]$Values) {
    return ($Values | Measure-Object -Average).Average
}

function Get-StdDev([double[]]$Values) {
    if ($Values.Count -lt 2) { return 0.0 }
    $mean = Get-Mean $Values
    $sum2 = 0.0
    foreach ($x in $Values) {
        $d = $x - $mean
        $sum2 += $d * $d
    }
    return [math]::Sqrt($sum2 / ($Values.Count - 1))
}

$benchArgs = @(
    $InnerIterations,
    $MaxOuter,
    $BlockY,
    $TargetNodes,
    $MinNodes,
    $OuterTolerance.ToString('G17', [Globalization.CultureInfo]::InvariantCulture)
)

$rows = @()
Write-Host "GFSS M5 process-isolated production benchmark"
Write-Host "runs=$Runs exe=$Exe"
Write-Host "args=$($benchArgs -join ' ')"

for ($i = 1; $i -le $Runs; ++$i) {
    # Native invocation creates a fresh process for every run, so CUDA context,
    # allocator state, CPU caches and hierarchy construction are not reused.
    $outputLines = & $Exe @benchArgs 2>&1
    $exitCode = $LASTEXITCODE
    $text = ($outputLines | Out-String)

    if ($exitCode -ne 0) {
        Write-Host $text
        throw "Benchmark run $i failed with exit code $exitCode."
    }

    $oracle = Get-Bool $text 'hierarchy_oracle_accept'
    $converged = Get-Bool $text 'converged'
    $breakdown = Get-Bool $text 'breakdown'
    if (-not $oracle -or -not $converged -or $breakdown) {
        Write-Host $text
        throw "Benchmark run $i failed numerical acceptance."
    }

    $row = [pscustomobject]@{
        run = $i
        cpu_setup_ms = Get-Scalar $text 'cpu_production_hierarchy_setup_ms'
        gpu_setup_ms = Get-Scalar $text 'runtime_gpu_setup_ms'
        solve_ms = Get-Scalar $text 'persistent_solve_wall_ms'
        e2e_ms = Get-Scalar $text 'production_end_to_end_required_ms'
        gpu_inner_ms = Get-Scalar $text 'gpu_inner_solve_ms'
        fp64_residual_ms = Get-Scalar $text 'accurate_FP64_residual_ms'
        a1_ms = Get-Scalar $text 'setup_actual_A1_offdiagonal_ms'
        cached_a1p1_ms = Get-Scalar $text 'setup_cached_A1P1_ms'
        a2_ms = Get-Scalar $text 'setup_A2_ms'
        final_true_residual = Get-Scalar $text 'final_true_relative_residual'
    }
    $rows += $row

    Write-Host ([string]::Format(
        [Globalization.CultureInfo]::InvariantCulture,
        "run={0} setup_ms={1:F3} gpu_setup_ms={2:F3} solve_ms={3:F3} e2e_ms={4:F3} A1_ms={5:F3} A1P1_ms={6:F3} A2_ms={7:F3} residual={8:E3}",
        $row.run, $row.cpu_setup_ms, $row.gpu_setup_ms, $row.solve_ms,
        $row.e2e_ms, $row.a1_ms, $row.cached_a1p1_ms, $row.a2_ms,
        $row.final_true_residual))
}

if ($CsvPath -ne '') {
    $rows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $CsvPath
    Write-Host "csv=$CsvPath"
}

$metricNames = @(
    'cpu_setup_ms',
    'gpu_setup_ms',
    'solve_ms',
    'e2e_ms',
    'gpu_inner_ms',
    'fp64_residual_ms',
    'a1_ms',
    'cached_a1p1_ms',
    'a2_ms'
)

Write-Host ""
Write-Host "summary=median_first min max mean stdev cv_pct"
foreach ($name in $metricNames) {
    [double[]]$values = @($rows | ForEach-Object { [double]($_.$name) })
    $median = Get-Median $values
    $mean = Get-Mean $values
    $std = Get-StdDev $values
    $min = ($values | Measure-Object -Minimum).Minimum
    $max = ($values | Measure-Object -Maximum).Maximum
    $cv = if ($mean -ne 0.0) { 100.0 * $std / $mean } else { 0.0 }
    Write-Host ([string]::Format(
        [Globalization.CultureInfo]::InvariantCulture,
        "metric={0} median={1:F6} min={2:F6} max={3:F6} mean={4:F6} stdev={5:F6} cv_pct={6:F3}",
        $name, $median, $min, $max, $mean, $std, $cv))
}

[double[]]$residuals = @($rows | ForEach-Object { [double]$_.final_true_residual })
Write-Host ([string]::Format(
    [Globalization.CultureInfo]::InvariantCulture,
    "residual_max={0:E12} residual_min={1:E12}",
    ($residuals | Measure-Object -Maximum).Maximum,
    ($residuals | Measure-Object -Minimum).Minimum))
