@echo off
title=ColorChanger
goto col
:col
echo.
echo 颜色代码：(输入错误则无效)
echo ----------------------------------------------------------------
echo 0=黑   1=蓝   2=深绿   3=深青   4=红   5=紫   6=黄   7=白   8=灰 
echo 9=淡蓝    A=淡绿    B=青    C=淡红    D=淡紫    E=淡黄    F=亮白
echo ----------------------------------------------------------------
echo.
echo 请输入代码并按Enter键.
set/p BG=输入背景色:
set/p WO=输入字体色:
color %BG%%WO%
echo 您对当前颜色是否满意？
echo 满意="y",不满意="n".(注意大小写)
set/p YN=(y/n)?:
cls
If %YN%==y goto con 
If %YN%==n goto col
:con
cls
title=Start "" "*.*"
set/p DIR=请输入要运行的BAT或CMD路径:
call "%DIR%"
echo =================================================================
echo 结束!按任意键退出.
echo =================================================================
pause > nul