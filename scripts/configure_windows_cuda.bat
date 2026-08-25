@echo off
setlocal

cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DGFSS_ENABLE_CUDA=ON -DGFSS_BUILD_TESTS=ON
if errorlevel 1 exit /b %errorlevel%

cmake --build build --config Release
if errorlevel 1 exit /b %errorlevel%

ctest --test-dir build -C Release --output-on-failure
if errorlevel 1 exit /b %errorlevel%

build\Release\gfss.exe info
