# 启动 uvicorn 服务（后台运行）
Start-Process "G:\bypaddle\venv\Scripts\uvicorn.exe" -ArgumentList "13-web:app --host 0.0.0.0 --port 8011 --reload"

# 启动 ngrok 服务（后台运行）
Start-Process "G:\bypaddle\ngrok.exe" -ArgumentList "http 8011"
