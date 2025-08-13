@echo off

@REM call "%USERPROFILE%\anaconda3\Scripts\activate.bat"

echo ===============================
echo   [MODEL 1] assistant
echo ===============================
call conda activate assistant
echo current conda: %CONDA_DEFAULT_ENV%
python assistant/assistant_s2_5.py
call conda deactivate
echo current conda: %CONDA_DEFAULT_ENV%


echo ==============================
echo   [MODEL 2] triposr
echo ==============================
setlocal enabledelayedexpansion
call conda activate TripoSR
echo current conda: %CONDA_DEFAULT_ENV%

set "max_name="
set "max_num=0"

:: frame_*.jpg 파일 순회
for %%f in (output\crop\frame_*.jpg) do (
    set "filename=%%~nf"
    set "num=!filename:~6!"

    :: 문자열 비교 (자릿수가 같다면 문자열 비교도 정수 비교처럼 작동)
    if "!num!" GTR "!max_num!" (
        set "max_num=!num!"
        set "max_name=%%f"
    )
)

if defined max_name (
    echo Latest Image: !max_name!
    python TripoSR\triposr.py "!max_name!" --output-dir output\
) else (
    echo ❌ No valid frame_*.jpg found!
)


call conda deactivate

@REM echo =============================
@REM echo   [MODEL 3] sampler_env 실행
@REM echo =============================
@REM call conda activate sampler_env
@REM echo current conda: %CONDA_DEFAULT_ENV%
@REM python run_sampler.py
@REM call conda deactivate

echo.
echo Done with every model!
pause
