@echo off
title Hukuk RAG Sistemi
color 0A

echo ========================================================
echo [1/3] Kutuphaneler kontrol ediliyor...
echo ========================================================
python -m pip install -r requirements.txt --quiet
echo Kutuphaneler hazir!

echo.
echo ========================================================
echo [2/3] Dokumanlar isleniyor ve veritabanina ekleniyor...
echo ========================================================
python ingest.py
echo Dokumanlar veritabanina kaydedildi!

echo.
echo ========================================================
echo [3/3] Sohbet arayuzu baslatiliyor...
echo ========================================================
python -m streamlit run app.py
pause
