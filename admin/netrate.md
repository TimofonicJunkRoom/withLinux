mtu

```
iptables -A INPUT -s x.x.x.x -m limit --limit 20/s -j ACCEPT
iptables -A INPUT -s x.x.x.x -j DROP
```

http://blog.163.com/bdweizhong@yeah/blog/static/963698182013431119108/
http://shixm.iteye.com/blog/2072149
http://www.linuxdiyf.com/linux/20214.html
