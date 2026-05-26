@echo off
REM Скрипт для установки зависимостей LightGCN
echo ============================================================
echo Installing PyTorch and PyTorch Geometric dependencies...
echo ============================================================

REM Проверяем версию Python
python --version

REM Устанавливаем зависимости
echo.
echo [1] Installing PyTorch (CPU version - быстрее)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo [2] Installing PyTorch Geometric...
pip install torch_geometric

echo.
echo [3] Installing tqdm (if not already installed)...
pip install tqdm

echo.
echo ============================================================
echo Installation complete! Testing imports...
echo ============================================================
python -c "import torch; import torch_geometric; print('✓ SUCCESS: All dependencies installed!')"

pause
