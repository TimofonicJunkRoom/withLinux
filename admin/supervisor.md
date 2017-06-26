Supervisor
===


http://www.liaoxuefeng.com/article/0013738926914703df5e93589a14c19807f0e285194fe84000

http://supervisord.org/subprocess.html#subprocess-environment

## Tips

when supervisor tells you that it cannot find the command to execute, then
you could try to use absolute path in the configuration file. For instance,

```
[program:gogs]
command=/home/lumin/gogs/gogs web
directory=/home/lumin/gogs/
user=lumin
environment=HOME="/home/lumin/",USER="lumin",PATH="/usr/bin:/bin:/usr/local/games:/usr/games"
```

In the above example, if you write `command=./gogs ...` or something else
it would not work and end up with that error.
