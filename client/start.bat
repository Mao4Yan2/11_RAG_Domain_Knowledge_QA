@echo off
chcp 65001 >nul
echo ========================================
echo   RAG 知识库问答系统 ...
echo ========================================

REM -------- 这里改成你的实际路径 --------
set CONDA_ENV=D:\86139\envs\RAG_env
set PROJECT_DIR=E:\_Project\11_RAG_Domain_Knowledge_QA\client
REM -------------------------------------

echo 激活虚拟环境: %CONDA_ENV%
call conda activate %CONDA_ENV%

echo 进入项目目录: %PROJECT_DIR%
cd /d "%PROJECT_DIR%"

echo 启动 Streamlit 应用...
echo 提示: 浏览器会自动打开 http://localhost:8501
streamlit run app.py

pause