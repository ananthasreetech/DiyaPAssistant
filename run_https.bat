@echo off
echo Starting Diya with HTTPS...
echo.

REM Generate cert if it doesn't exist yet
IF NOT EXIST cert.pem (
    echo Installing cryptography package...
    pip install cryptography -q
    echo Generating SSL certificate...
    python gen_cert.py
    IF ERRORLEVEL 1 (
        echo ERROR: Certificate generation failed. See above.
        pause
        exit /b 1
    )
)

echo.
echo =============================================
echo  Open on your mobile Chrome:
echo  Find your IP by running: ipconfig
echo  Then open: https://YOUR_IP:8501
echo  Tap "Advanced" then "Proceed" on warning
echo =============================================
echo.

streamlit run app.py --server.sslCertFile=cert.pem --server.sslKeyFile=key.pem
