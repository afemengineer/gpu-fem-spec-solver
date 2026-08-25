param(
  [string]$CudaArch = "75"
)

$ErrorActionPreference = "Stop"

function Assert-LastExitCode([string]$Step) {
  if ($LASTEXITCODE -ne 0) {
    throw "$Step failed with exit code $LASTEXITCODE."
  }
}

function Resolve-VcVars64 {
  $candidates = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  )

  $vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
  if (Test-Path $vswhere) {
    $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($installPath) {
      $candidates = @((Join-Path $installPath "VC\Auxiliary\Build\vcvars64.bat")) + $candidates
    }
  }

  foreach ($candidate in $candidates | Select-Object -Unique) {
    if (Test-Path $candidate) {
      return (Resolve-Path $candidate).Path
    }
  }

  return $null
}

function Import-MsvcEnvironment {
  if ((Get-Command cl.exe -ErrorAction SilentlyContinue) -and
      (Get-Command nmake.exe -ErrorAction SilentlyContinue)) {
    Write-Host "MSVC environment already initialized."
    return
  }

  $vcvars = Resolve-VcVars64
  if (-not $vcvars) {
    throw "Could not locate vcvars64.bat. Install the MSVC x64 C++ Build Tools workload."
  }

  Write-Host "Initializing MSVC x64 environment: $vcvars"

  # A .bat file cannot modify its parent PowerShell process directly. Run it in
  # cmd.exe, dump the resulting environment, and import those variables here.
  $lines = & cmd.exe /d /s /c "`"call `"$vcvars`" >nul && set`""
  Assert-LastExitCode "Visual Studio environment initialization"

  foreach ($line in $lines) {
    $separator = $line.IndexOf('=')
    if ($separator -le 0) { continue }

    $name = $line.Substring(0, $separator)
    $value = $line.Substring($separator + 1)
    Set-Item -Path "Env:$name" -Value $value
  }

  if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw "vcvars64.bat completed, but cl.exe is still unavailable."
  }
  if (-not (Get-Command nmake.exe -ErrorAction SilentlyContinue)) {
    throw "vcvars64.bat completed, but nmake.exe is still unavailable."
  }
}

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

if (-not (Get-Command cmake.exe -ErrorAction SilentlyContinue)) {
  throw "cmake.exe was not found. Install CMake and reopen the terminal."
}

Import-MsvcEnvironment

$nvcc = Resolve-Nvcc
if (-not $nvcc) {
  throw "CUDA compiler nvcc.exe was not found in PATH, CUDA_PATH, or the standard NVIDIA Toolkit directory."
}

Write-Host "Using C++ compiler: $((Get-Command cl.exe).Source)"
Write-Host "Using NMake:       $((Get-Command nmake.exe).Source)"
Write-Host "Using CUDA:        $nvcc"
Write-Host "CUDA architecture: sm_$CudaArch"
& $nvcc --version
Assert-LastExitCode "nvcc --version"

$buildDir = Join-Path $PWD "build-cuda"
$cachePath = Join-Path $buildDir "CMakeCache.txt"

if (Test-Path $cachePath) {
  $cache = Get-Content $cachePath -Raw
  if ($cache -notmatch 'CMAKE_GENERATOR:INTERNAL=NMake Makefiles') {
    Write-Host "Removing build-cuda because it was configured with a different generator..."
    Remove-Item -Recurse -Force $buildDir
  }
}

$nvccCMake = $nvcc -replace '\\', '/'

Write-Host ""
Write-Host "Configuring GFSS..."
& cmake.exe -S . -B build-cuda -G "NMake Makefiles" `
  -DCMAKE_BUILD_TYPE=Release `
  -DGFSS_ENABLE_CUDA=ON `
  -DGFSS_BUILD_TESTS=ON `
  "-DCMAKE_CUDA_COMPILER=$nvccCMake" `
  "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch"
Assert-LastExitCode "CMake configure"

Write-Host ""
Write-Host "Building GFSS..."
& cmake.exe --build build-cuda
Assert-LastExitCode "CMake build"

Write-Host ""
Write-Host "Running tests..."
& ctest.exe --test-dir build-cuda --output-on-failure
Assert-LastExitCode "CTest"

$gfssExe = Join-Path $buildDir "gfss.exe"
if (-not (Test-Path $gfssExe)) {
  throw "Build completed but $gfssExe was not produced."
}

Write-Host ""
Write-Host "GPU information:"
& $gfssExe info
Assert-LastExitCode "gfss info"
