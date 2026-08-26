param()

$ErrorActionPreference = "Stop"

function Read-NormalizedText {
    param([string]$Path)
    $text = [System.IO.File]::ReadAllText($Path)
    return $text.Replace("`r`n", "`n").Replace("`r", "`n")
}

function Write-Utf8NoBom {
    param(
        [string]$Path,
        [string]$Text
    )
    $clean = $Text.TrimEnd("`r", "`n") + "`n"
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $clean, $utf8)
}

function Replace-RequiredLiteral {
    param(
        [string]$Text,
        [string]$Old,
        [string]$New,
        [string]$Description
    )
    if (-not $Text.Contains($Old)) {
        throw "Patch failed: $Description"
    }
    return $Text.Replace($Old, $New)
}

$root = Split-Path -Parent $PSScriptRoot
$sourcePath = Join-Path $root "src/gpu_pcg.cu"
$headerPath = Join-Path $root "include/gfss/gpu_solver.hpp"
$benchPath = Join-Path $root "benchmarks/gpu_pcg_bench.cpp"
$testPath = Join-Path $root "tests/gpu_pcg_test.cpp"

# Header ----------------------------------------------------------------------
$header = Read-NormalizedText $headerPath
$header = Replace-RequiredLiteral $header `
    "    std::size_t residual_replacements{0};" `
    "    std::size_t residual_audits{0};" `
    "rename residual replacement counter"

if (-not $header.Contains("audited_relative_residual")) {
    $header = Replace-RequiredLiteral $header `
        "    double reported_relative_residual{0.0};" `
        "    double reported_relative_residual{0.0};`n    double audited_relative_residual{0.0};" `
        "add audited residual result"
}

$header = $header.Replace(
    "// are excluded from solve_ms; PCG kernels, cuBLAS reductions, and their host",
    "// are excluded from solve_ms; PCG kernels, cuBLAS reductions, residual audits, and their host")
Write-Utf8NoBom $headerPath $header

# CUDA solver -----------------------------------------------------------------
$source = Read-NormalizedText $sourcePath
$source = $source.Replace("residual_replacements", "residual_audits")

$oldCandidateBlock = @'
                const bool recursive_candidate =
                    result.reported_relative_residual <= relative_tolerance;
                const bool periodic_replacement =
                    result.iterations % residual_replacement_period == 0U;

                if (recursive_candidate || periodic_replacement) {
                    launch_pcg_matvec(mesh, block_y, nodes, d_x, d_q);
                    ++result.matvecs;
                    ++result.residual_audits;

                    residual_from_b_ax_pcg_kernel<<<vector_blocks, vector_threads>>>(
                        n, d_b, d_q, d_r);
                    check_cuda_pcg(cudaGetLastError(), "GPU PCG reliable residual launch");

                    rr = dot_pcg(handle, n, d_r, d_r);
                    result.reported_relative_residual =
                        std::sqrt(static_cast<double>(rr) / static_cast<double>(bnorm2));

                    if (!std::isfinite(rr) ||
                        !std::isfinite(result.reported_relative_residual)) {
                        throw std::runtime_error("GPU PCG recomputed residual became non-finite");
                    }
                    if (result.reported_relative_residual <= relative_tolerance) {
                        result.converged = true;
                        break;
                    }

                    // Residual replacement breaks exact recurrence consistency.
                    // Restart with the freshly preconditioned residual rather
                    // than pretending the previous search direction remains
                    // conjugate to the new residual.
                    launch_pcg_jacobi(mesh, block_y, nodes, d_r, d_z);
                    rho = dot_pcg(handle, n, d_r, d_z);
                    if (!(rho > 0.0f) || !std::isfinite(rho)) {
                        throw std::runtime_error(
                            "GPU PCG reliable restart produced invalid preconditioned residual");
                    }
                    check_cuda_pcg(cudaMemcpyAsync(d_p, d_z, vector_bytes,
                                                   cudaMemcpyDeviceToDevice),
                                   "cudaMemcpyAsync(PCG reliable restart p=z)");
                    continue;
                }

                launch_pcg_jacobi(mesh, block_y, nodes, d_r, d_z);
'@

$newCandidateBlock = @'
                const bool recursive_candidate =
                    result.reported_relative_residual <= relative_tolerance;

                if (recursive_candidate) {
                    // Audit the current solution without mutating the recursive
                    // residual or search direction. d_z is scratch here and is
                    // recomputed by Jacobi immediately after a failed audit.
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

$source = Replace-RequiredLiteral $source `
    $oldCandidateBlock.TrimEnd() `
    $newCandidateBlock.TrimEnd() `
    "replace reliable restart with non-mutating audit"

$source = Replace-RequiredLiteral $source `
    "            constexpr std::size_t residual_replacement_period = 50U;`n" `
    "" `
    "remove periodic residual replacement period"

if (-not $source.Contains("result.audited_relative_residual = result.reported_relative_residual;")) {
    $oldInitialResidual = @'
        result.reported_relative_residual =
            bnorm2 > 0.0f ? std::sqrt(static_cast<double>(rr) / static_cast<double>(bnorm2)) : 0.0;
'@
    $newInitialResidual = @'
        result.reported_relative_residual =
            bnorm2 > 0.0f ? std::sqrt(static_cast<double>(rr) / static_cast<double>(bnorm2)) : 0.0;
        result.audited_relative_residual = result.reported_relative_residual;
'@
    $source = Replace-RequiredLiteral $source `
        $oldInitialResidual.TrimEnd() `
        $newInitialResidual.TrimEnd() `
        "initialize audited residual"
}

if (-not $source.Contains("GPU PCG final residual audit launch")) {
    $syncLine = '        check_cuda_pcg(cudaDeviceSynchronize(), "cudaDeviceSynchronize(PCG solve)");'
    $finalAudit = @'
        if (!result.converged && bnorm2 > 0.0f) {
            constexpr int final_vector_threads = 256;
            const int final_vector_blocks =
                (n + final_vector_threads - 1) / final_vector_threads;
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

        check_cuda_pcg(cudaDeviceSynchronize(), "cudaDeviceSynchronize(PCG solve)");
'@
    $source = Replace-RequiredLiteral $source `
        $syncLine `
        $finalAudit.TrimEnd() `
        "insert final residual audit"
}
Write-Utf8NoBom $sourcePath $source

# Benchmark -------------------------------------------------------------------
$bench = Read-NormalizedText $benchPath
$bench = $bench.Replace("residual_replacements=", "residual_audits=")
$bench = $bench.Replace("result.residual_replacements", "result.residual_audits")
if (-not $bench.Contains("gpu_audited_relative_residual=")) {
    $bench = Replace-RequiredLiteral $bench `
        '                  << "reported_relative_residual=" << result.reported_relative_residual << ''\n''' `
        ('                  << "recursive_relative_residual=" << result.reported_relative_residual << ''\n''' + "`n" +
         '                  << "gpu_audited_relative_residual=" << result.audited_relative_residual << ''\n''') `
        "add audited residual benchmark output"
}
$bench = $bench.Replace(
    "WARNING: PCG did not reach the requested recomputed-residual tolerance",
    "WARNING: PCG did not reach the requested audited-residual tolerance")
Write-Utf8NoBom $benchPath $bench

# Test ------------------------------------------------------------------------
$test = Read-NormalizedText $testPath
$test = $test.Replace("residual_replacements", "residual_audits")
$test = $test.Replace("residual replacement", "residual audit")
$test = $test.Replace("residual replacements", "residual audits")

if (-not $test.Contains("gpu.audited_relative_residual <= 1.0e-5")) {
    $oldRequire = @'
    require(gpu.reported_relative_residual <= 1.0e-5,
            "GPU PCG recursive residual must meet requested tolerance");
'@
    $newRequire = @'
    require(gpu.reported_relative_residual <= 1.0e-5,
            "GPU PCG recursive residual must meet requested tolerance");
    require(gpu.audited_relative_residual <= 1.0e-5,
            "GPU PCG audited residual must meet requested tolerance");
'@
    $test = Replace-RequiredLiteral $test `
        $oldRequire.TrimEnd() `
        $newRequire.TrimEnd() `
        "require audited residual convergence"
}

if (-not $test.Contains('<< " audited_rel=" << gpu.audited_relative_residual')) {
    $test = Replace-RequiredLiteral $test `
        '              << " reported_rel=" << gpu.reported_relative_residual' `
        ('              << " recursive_rel=" << gpu.reported_relative_residual' + "`n" +
         '              << " audited_rel=" << gpu.audited_relative_residual') `
        "print audited residual in test"
}
Write-Utf8NoBom $testPath $test

Write-Host "Applied PCG audit-only experiment patch." -ForegroundColor Green
Write-Host "Run: git diff --check" -ForegroundColor Cyan
Write-Host "Then review: git --no-pager diff -- include/gfss/gpu_solver.hpp src/gpu_pcg.cu benchmarks/gpu_pcg_bench.cpp tests/gpu_pcg_test.cpp"
