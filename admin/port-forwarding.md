Port forwarding
===

# Via SSH

```
-C compression
-N do not execute remote command
-f go background
-g allow remote hosts to connect to local forwarded ports.
```

```
# Forward packets to hostname:6666 to 10.170.1.1:22
ssh -L 6666:10.170.1.1:22 master@hostname -C -f -N -g
```

```
# Forward packages to 10.170.1.1:22 to hostname:6666
ssh -R 6666:10.170.1.1:22 master@hostname -C -f -N -g
```
