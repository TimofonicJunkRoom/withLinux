@echo off
color 17
title=请输入倒计时时间再按回车，单位是秒。
set/p TIM=时间:
for /L %%a in (
%TIM%,-1,0
) do (
title=还剩余%%a 秒
echo %TIM%s后提醒，期间请不要关闭本程序;建议最小化本窗口。
echo 还剩余 %%a 秒
ping -n 2 localhost 1>nul 2>nul
cls
)
start .\_.bat
exit
