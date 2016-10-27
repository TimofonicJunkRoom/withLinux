Docker.io
===

# installation

There is official Debian package for `docker.io`.  
```
apt install docker.io
adduser $USER docker     # grant the docker group permission to avoid sudo
```

# docker image preparation by oneself

1. Setup a basic system via debootstrap
```
$ debootstrap jessie ./jessie http://127.0.0.1/debian
$ cd ./jessie
$ tar zcvf ../jessie.stage3.tar.gz .
```
alternatively, one can download archlinux bootstrap tarball or gentoo
stage3 tarball or something else in favour of other distributions.

2. import it into docker
```
$ cat jessie.stage3.tar | sudo docker import - jessie:19700101
```

# import docker image from docker
```
$ docker search alpine
$ docker pull alpine
```

# docker run

* Hello world test
```
$ sudo docker run ubuntu:14.04 /bin/echo "Hello World!"
```

* Call a shell from container.
```
$ sudo docker run -t -i ubuntu:14.04 /bin/bash
```

* re-enter a previously created container
```
docker ps -as                 # show all containers
docker start -ai 01234567     # container ID
```

# deletion
```
docker ps -as # list containers
docker rm 1234567 # remove container
docker images -a # list images
docker rmi 1234567 # remove image
```

# Deploy archlinux containers and expose ssh port to the wild
```
docker run -t -i archlinux:20161001 /bin/bash
(docker)# echo 'root:toor' | chpasswd
(docker)# pacman -S openssh
(docker)# /usr/bin/sshd -D # make sure it works
docker commit 01234567 archlinux:ssh

docker run -d -p 22:22 archlinux:ssh /usr/bin/sshd -D
OR
docker run -d -P archlinux:ssh /usr/bin/sshd -D # choose random port mapping
```
