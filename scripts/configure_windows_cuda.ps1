$ErrorActionPreference = "Stop"

function Resolve-Nvcc {
  $candidates = @()

  $cmd = Get-Command nvcc.exe -ErrorAction SilentlyContinue
  if ($cmd) {
    $candidates += $cmd.Source
  }

  if ($env:CUDA_PATH) {
    $candidates += (Join-Path $env:CUDA_PATH "bin\nvcc.exe")
  }

  $toolkitRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
  if (Test-Path $toolkitRoot) {
    $candidates += Get-ChildItem $toolkitRoot -Directory -ErrorAction SilentlyContinue |
      Sort-Object Name -Descending |
      ForEach-Object { Join-Path $_.FullName "bin\nvcc.exe" }
  }

  foreach ($candidate in $candidates | Select-Object -Unique) {
    if ($candidate -and (Test-Path $candidate)) {
      return (Resolve-Path $candidate).Path
    }
  }

  return $null
}

$nvcc = Resolve-Nvcc
if (-not $nvcc) {
  Write-Host "CUDA compiler (nvcc.exe) was not found." -ForegroundColor Red
  Write-Host "Checked PATH, CUDA_PATH, and the standard NVIDIA CUDA Toolkit installation directory."
  Write-Host ""
  Write-Host "Useful diagnostics:"
  Write-Host "  Get-Command nvcc.exe -ErrorAction SilentlyContinue"
  Write-Host "  `$env:CUDA_PATH"
  Write-Host "  Get-ChildItem 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' -Directory"
  exit 1
}

Write-Host "Using CUDA compiler: $nvcc"
& $nvcc --version

$nvccCMake = $nvcc -replace '\\', '/'

cmake -S . -B build -G "Visual Studio 18 2026" -A x64 `
  -DGFSS_ENABLE_CUDA=ON `
  -DGFSS_BUILD_TESTS=ON `
  "-DCMAKE_CUDA_COMPILER=$nvccCMake"

cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure

Write-Host ""
Write-Host "GPU information:"
& .\build\Release\gfss.exe info
