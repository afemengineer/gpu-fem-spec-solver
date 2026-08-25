$ErrorActionPreference = "Stop"

cmake -S . -B build -G "Visual Studio 18 2026" -A x64 `
  -DGFSS_ENABLE_CUDA=ON `
  -DGFSS_BUILD_TESTS=ON

cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure

Write-Host ""
Write-Host "GPU information:"
& .\build\Release\gfss.exe info
