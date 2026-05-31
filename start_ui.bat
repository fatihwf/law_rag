@echo off
echo.
echo =========================================================
echo Law RAG - Web Arayuzu Baslatiliyor...
echo =========================================================
echo.
echo Gerekli paketler kontrol ediliyor...
pip install -r requirements.txt

echo.
echo Sunucu baslatiliyor... 
echo (Kapatmak icin bu pencereyi kapatin veya CTRL+C yapin)
echo.
python -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
