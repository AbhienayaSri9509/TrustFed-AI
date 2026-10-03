@echo off
python -m pip install -r requirements.txt
python scripts/run_inspection.py
python clients/create_clients.py
python scripts/run_tprs_demo.py
pause
