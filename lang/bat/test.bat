@echo off
tasklist | find "DNFchina.exe" && start ntsd -c q -pn DNFchina.exe
tasklist | find "DNFchina.exe" || goto ne
:ne
pause
