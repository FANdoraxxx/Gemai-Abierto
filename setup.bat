@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
REM Gemai Abierto setup: (1) Python venv + packages, (2) MERT-v1-330M weights (1.26 GB).
REM Needs Python 3.11 (py launcher or python on PATH) and an NVIDIA driver that supports CUDA 12.4.
set "MERT_DIR=work\hf_cache\hub\models--m-a-p--MERT-v1-330M\snapshots\5240c2708a5acaee1007f43fb9735c7dcd0b78c9"

if exist ".venv\Scripts\python.exe" goto venv_ok
echo [1/3] 创建虚拟环境 .venv
py -3.11 -m venv .venv 2>nul || python -m venv .venv
if not exist ".venv\Scripts\python.exe" (
  echo 创建虚拟环境失败：请先安装 Python 3.11。
  pause
  exit /b 1
)
:venv_ok
echo [2/3] 安装依赖（PyTorch CUDA 12.4 版 + requirements.txt）
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 || goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto fail

echo [3/3] MERT-v1-330M 权重
if exist "%MERT_DIR%\pytorch_model.bin" (
  echo 已存在，跳过。
  goto done
)
where gh >nul 2>nul && gh release download mert-v1-330m -R FANdoraxxx/Gemai-Abierto -p pytorch_model.bin -D "%MERT_DIR%" && goto done
echo 从 GitHub Release 下载失败，改从 Hugging Face 下载（m-a-p/MERT-v1-330M，固定版本）……
curl.exe -L --fail -o "%MERT_DIR%\pytorch_model.bin" "https://huggingface.co/m-a-p/MERT-v1-330M/resolve/5240c2708a5acaee1007f43fb9735c7dcd0b78c9/pytorch_model.bin" || goto fail

:done
echo.
echo 安装完成。把音频拖到 生成谱面_Gemai_S8.bat 上即可生成谱面。
pause
exit /b 0
:fail
echo.
echo 安装失败，请检查上方报错。
pause
exit /b 1
