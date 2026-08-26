param()

$ErrorActionPreference = "Stop"

function Replace-Required {
    param(
        [string]$Text,
        [string]$Pattern,
        [string]$Replacement,
        [string]$Description
    )
    $updated = [regex]::Replace($Text, $Pattern, $Replacement)
    if ($updated -eq $Text) {
        throw "Patch failed: $Description"
    }
    return $updated
}

$root = Split-Path -Parent $PSScriptRoot
$sourcePath = Join-Path $root "src/gpu_pcg.cu"
$headerPath = Join-Path $root "include/gfss/gpu_solver.hpp"
$benchPath = Join-Path $root "benchmarks/gpu_pcg_bench.cpp"
$testPath = Join-Path $root "tests/gpu_pcg_test.cpp"

# Header: make the distinction between recursive and independently audited residual explicit.
$header = Get-Content $headerPath -Raw
$header = $header.Replace("std::size_t residual_replacements{0};", "std::size_t residual_audits{0};")
if ($header -notmatch "audited_relative_residual") {
    $header = $header.Replace(
        "double reported_relative_residual{0.0};",
        "double reported_relative_residual{0.0};`r`n    double audited_relative_residual{0.0};")
}
$header = $header.Replace(
    "// are excluded from solve_ms; PCG kernels, cuBLAS reductions, and their host",
    "// are excluded from solve_ms; PCG kernels, cuBLAS reductions, residual audits, and their host")
Set-Content -Path $headerPath -Value $header -Encoding utf8

$source = Get-Content $sourcePath -Raw
$source = $source.Replace("residual_replacements", "residual_audits")

# Remove the periodic-restart policy if this script is applied to that revision.
$source = [regex]::Replace(
    $source,
    '(?m)^\s*constexpr std::size_t residual_replacement_period = 50U;\r?\n',
    '')
$source = [regex]::Replace(
    $source,
    '(?s)\s*const bool periodic_replacement =\s*result\.iterations % residual_replacement_period == 0U;',
    '')
$source = $source.Replace(
    "if (recursive_candidate || periodic_replacement) {",
    "if (recursive_candidate) {")

# Replace the mutating reliable-restart block with an audit-only block.  d_z is
# stale at this point and is safe scratch: it is recomputed from recursive d_r
# immediately afterwards if the audit does not converge.
$auditBlock = @'
                const bool recursive_candidate =
                    result.reported_relative_residual <= relative_tolerance;

                if (recursive_candidate) {
                    launch_pcg_matvec(mesh, block_y, nodes, d_x, d_q);
                    ++result.matvecs;
                    ++result.residual_audits;

                    residual_from_b_ax_pcg_kernel<<<vector_blocks, vector_threads>>>(
                        n, d_b, d_q, d_z);
                    check_cuda_pcg(cudaGetLastError(), "GPU PCG residual audit launch");

                    const float rr_audit = dot_pcg(handle, n, d_z, d_z);
                    result.audited_relative_residual =
                        std::sqrt(static_cast<double>(rr_audit) /
                                  static_cast<double>(bnorm2));

                    if (!std::isfinite(rr_audit) ||
                        !std::isfinite(result.audited_relative_residual)) {
                        throw std::runtime_error("GPU PCG audited residual became non-finite");
                    }
                    if (result.audited_relative_residual <= relative_tolerance) {
                        result.converged = true;
                        break;
                    }
                }

                launch_pcg_jacobi(mesh, block_y, nodes, d_r, d_z);
'@

$pattern = '(?s)\s*const bool recursive_candidate =\s*result\.reported_relative_residual <= relative_tolerance;.*?\n\s*launch_pcg_jacobi\(mesh, block_y, nodes, d_r, d_z\);'
$source = Replace-Required $source $pattern ("`r`n" + $auditBlock.TrimEnd()) "replace reliable restart with audit-only check"

# The initial residual is exact because x=0 and r=b.
if ($source -notmatch 'result\.audited_relative_residual =\s*result\.reported_relative_residual;') {
    $source = $source.Replace(
        "result.reported_relative_residual =`r`n            bnorm2 > 0.0f ? std::sqrt(static_cast<double>(rr) / static_cast<double>(bnorm2)) : 0.0;",
        "result.reported_relative_residual =`r`n            bnorm2 > 0.0f ? std::sqrt(static_cast<double>(rr) / static_cast<double>(bnorm2)) : 0.0;`r`n        result.audited_relative_residual = result.reported_relative_residual;")
    $source = $source.Replace(
        "result.reported_relative_residual =`n            bnorm2 > 0.0f ? std::sqrt(static_cast<double>(rr) / static_cast<double>(bnorm2)) : 0.0;",
        "result.reported_relative_residual =`n            bnorm2 > 0.0f ? std::sqrt(static_cast<double>(rr) / static_cast<double>(bnorm2)) : 0.0;`n        result.audited_relative_residual = result.reported_relative_residual;")
}

# If max_iterations is reached without a successful candidate audit, always
# audit the final x once. This makes the returned audited residual meaningful.
if ($source -notmatch 'GPU PCG final residual audit launch') {
    $finalAudit = @'
        if (!result.converged && bnorm2 > 0.0f) {
            constexpr int final_vector_threads = 256;
            const int final_vector_blocks = (n + final_vector_threads - 1) / final_vector_threads;
            launch_pcg_matvec(mesh, block_y, nodes, d_x, d_q);
            ++result.matvecs;
            ++result.residual_audits;
            residual_from_b_ax_pcg_kernel<<<final_vector_blocks, final_vector_threads>>>(
                n, d_b, d_q, d_z);
            check_cuda_pcg(cudaGetLastError(), "GPU PCG final residual audit launch");
            const float rr_audit = dot_pcg(handle, n, d_z, d_z);
            result.audited_relative_residual =
                std::sqrt(static_cast<double>(rr_audit) /
                          static_cast<double>(bnorm2));
        }

'@
    $source = Replace-Required $source '(\s*)check_cuda_pcg\(cudaDeviceSynchronize\(\), "cudaDeviceSynchronize\(PCG solve\)"\);' ("`r`n" + $finalAudit + '        check_cuda_pcg(cudaDeviceSynchronize(), "cudaDeviceSynchronize(PCG solve)");') "insert final residual audit"
}
Set-Content -Path $sourcePath -Value $source -Encoding utf8

# Benchmark: expose recursive vs audited residual separately.
$bench = Get-Content $benchPath -Raw
$bench = $bench.Replace("residual_replacements=", "residual_audits=")
$bench = $bench.Replace("result.residual_replacements", "result.residual_audits")
if ($bench -notmatch 'gpu_audited_relative_residual') {
    $bench = $bench.Replace(
        '<< "reported_relative_residual=" << result.reported_relative_residual << ''\n''',
        '<< "recursive_relative_residual=" << result.reported_relative_residual << ''\n''`r`n                  << "gpu_audited_relative_residual=" << result.audited_relative_residual << ''\n''')
}
$bench = $bench.Replace(
    "WARNING: PCG did not reach the requested recomputed-residual tolerance",
    "WARNING: PCG did not reach the requested audited-residual tolerance")
Set-Content -Path $benchPath -Value $bench -Encoding utf8

# Test: audit count replaces replacement count, and convergence must be backed by
# the GPU audit as well as the independent CPU true-residual check.
$test = Get-Content $testPath -Raw
$test = $test.Replace("residual_replacements", "residual_audits")
$test = $test.Replace("gpu.matvecs == gpu.iterations + gpu.residual_audits", "gpu.matvecs == gpu.iterations + gpu.residual_audits")
if ($test -notmatch 'audited_relative_residual <= 1.0e-5') {
    $test = $test.Replace(
        'require(gpu.reported_relative_residual <= 1.0e-5,`r`n            "GPU PCG recursive residual must meet requested tolerance");',
        'require(gpu.reported_relative_residual <= 1.0e-5,`r`n            "GPU PCG recursive residual must meet requested tolerance");`r`n    require(gpu.audited_relative_residual <= 1.0e-5,`r`n            "GPU PCG audited residual must meet requested tolerance");')
    $test = $test.Replace(
        'require(gpu.reported_relative_residual <= 1.0e-5,`n            "GPU PCG recursive residual must meet requested tolerance");',
        'require(gpu.reported_relative_residual <= 1.0e-5,`n            "GPU PCG recursive residual must meet requested tolerance");`n    require(gpu.audited_relative_residual <= 1.0e-5,`n            "GPU PCG audited residual must meet requested tolerance");')
}
$test = $test.Replace(
    '<< " reported_rel=" << gpu.reported_relative_residual',
    '<< " recursive_rel=" << gpu.reported_relative_residual`r`n              << " audited_rel=" << gpu.audited_relative_residual')
Set-Content -Path $testPath -Value $test -Encoding utf8

Write-Host "Applied PCG audit-only experiment patch." -ForegroundColor Green
Write-Host "Review with: git --no-pager diff -- include/gfss/gpu_solver.hpp src/gpu_pcg.cu benchmarks/gpu_pcg_bench.cpp tests/gpu_pcg_test.cpp"
