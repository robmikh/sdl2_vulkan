@echo off
setlocal

set CMAKE_BUILD_TYPE=Debug

pushd %~dp0..\..\build\
set "BUILD_PATH=%CD%"
popd

cmake . "-B%BUILD_PATH%" -GNinja "-DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE%" "-DCMAKE_INSTALL_PREFIX=%BUILD_PATH%"
ninja -C "%BUILD_PATH%"

REM Copy SDL2 depencencies
IF NOT EXIST %BUILD_PATH%\SDL2.dll copy %SDL2%\lib\x64\SDL2.dll %BUILD_PATH%
