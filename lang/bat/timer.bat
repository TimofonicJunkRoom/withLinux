@echo off
color 17
title=�����뵹��ʱʱ���ٰ��س�����λ���롣
set/p TIM=ʱ��:
for /L %%a in (
%TIM%,-1,0
) do (
title=��ʣ��%%a ��
echo %TIM%s�����ѣ��ڼ��벻Ҫ�رձ�����;������С�������ڡ�
echo ��ʣ�� %%a ��
ping -n 2 localhost 1>nul 2>nul
cls
)
start .\_.bat
exit
