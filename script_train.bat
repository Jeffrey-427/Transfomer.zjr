@echo off
call conda activate base
python src\train.py --config configs\base\config.yaml
pause
