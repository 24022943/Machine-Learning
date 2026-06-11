@echo off
python -m pytest tests -v --cov=carbon_utils --cov=scenario_projection --cov-report=term-missing
pause
