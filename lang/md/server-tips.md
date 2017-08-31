Title: 服务器使用提示
Date: 2000-01-01 00:00

## Operating System & Basic Services & Basic Usage

服务器系统为Debian Sid，基本特性与Ubuntu近似。如果您需要在个人计算机上安装
Linux并期望在科学计算方面糙快猛，可以考虑`Archlinux`。

出于安全考虑，不发放`Root`权限，并且设置`iptables`屏蔽几乎所有端口输入。理论上所有框架均可以在非特权模式下安装到用户自身家目录。

每个用户的默认Shell可能是`bash`。可以尝试学习使用`fish`或者`zsh`任意一个现代化Shell来提升操作效率。

其余各项Linux基础知识请参见[鸟哥的私房菜](http://linux.vbird.org)。

### 部分重要基础命令

部分主要Linux基础命令，以下列出的工具参考了Debian的`important`,`required`和`standard`三种Priority软件包，并不包括管理员工具。

* 自救技能查看帮助 `man`

* 基础文件操作和POSIX权限

```shell
列出当前目录内容 `ls -lh`，查看文件详情`stat`
创建文件 `touch`，编辑文件 `nano` 或 `vim` 或 `emacs`。
删除文件 `rm` (Danger)，删除文件夹 `rmdir` 移动文件 `mv` (Danger)
复制文件 `cp` 软链接   `ln`
打印文本文件： `cat`，查看文件头 `head`，查看文件尾 `tail` 文件查看器： less
排序： `sort`，去重 `uniq`，文件行数统计 `wc`
下载 `wget URL`
POSIX文件权限：`chmod`, `chown`， `chgrp`
同步磁盘缓存： `sync`
判断文件类型： file  计算md5哈希： `md5sum`
归档，压缩和解压缩： `tar`, `gzip`, `bzip2`, `xz`
```

* Shell与进程

```
本地终端：`bash`, 远程终端：`ssh`
查看用户进程 `ps ux`，结束进程 `kill PID`，资源监视 `top`
现代化资源监视 `htop` , 屏蔽SIGHUP：`nohup`
```


* 进阶生产力

```
版本控制：`git`
流编辑器：`sed`
文件查找工具：`find`
内容查找工具：`grep`
```

* 编译器和解释器们以及编译系统

```shell
Awk解释器：awk
C编译器： `gcc`, `clang`
C++编译器： `g++`， `clang++`
CUDA编译器：`nvcc`
Go编译器：`go`
Julia:
Lua解释器：`lua`， `luajit`
Python解释器：`python`， `python3`
Perl解释器：`perl`
Ruby解释器：`ruby`
Rust编译器：`rustc`
Scala: `scala`

编译系统: `make` `cmake` `setup.py`
```

### 基本设施

* IP

IP地址因为校园网设置原因会不定期变化，跟本机设置完全没有任何关系，此题无解。

* CUDA

服务器目前CUDA版本为`8.0.44`， 使用`nvidia-smi`查看显卡使用状态。

编译CUDA程序必须根据相应的编译系统调整默认编译器为Clang-3.8或者GCC-5，否则必失败。

* HDF5

服务器HDF5版本为1.10.0，使用1.8.0版本的程序可能出现兼容问题。

* OpenBLAS / Intel MKL

使用时请务必确认线程数环境变量正确设置，否则可能导致不必要的严重性能损耗。

* Python

服务器有`python 2.7`和`python 3.5`, `python 3.6`。默认符号链接指向为：

```sh
$ python        # python2.7
$ python3       # python3.5
```

平时建议使用`ipython`或`ipython3`进行工作。因为安全原因，不建议远程使用`jupyter notebook`。

当系统和家目录中散布存在着同一个包的不同版本时，注意调整`PYTHONPATH`以避免惊喜。

对于仅使用Python标准库的程序可以考虑使用Pypy进行加速。对于更高性能需求的
程序建议考虑C/C++模块，SWIG，Cython等解决方案。

* Git

请务必使用Git管理好自己的源代码，服务器一定不是100%可靠的。

* 用户之间的文件共享

简单chmod修改文件权限，其他用户即可访问。具体以后再写。

* 显示图像

正常情况下显示服务器上的的图像需要进行X转发，这个比较费流量而且在网络恶劣的
情况下使用效果保证会非常糟糕。网络情况良好的时候可以使用如下命令打开SSH的X
转发功能，然后就可以在命令行中调用自己喜欢的图像查看工具查看。

```
$ ssh -X user@hostname
```

在不适合转发的情况下，也有一些工具可以把图像转换成彩色Unicode字符，从而
直接以字符形式，低分辨率地将图像打印到终端。比如

```
$ catimg xxx.jpg
```

类似工具可以在StackOverflow上查到很多。

## Machine Learning Softwares

* Caffe

服务器系统自带CUDA版本Caffe（但未使用cuDNN），并带有Python3接口。
需要cuDNN（提速大概至少4倍？）请自行重新编译。需要matcaffe请自行编译。Caffe编译请使用OpenBLAS，默认编译器使用Clang-3.8或者GCC-5。

在python REPL里使用如下代码来测试你的Caffe导入路径及版本
```python
import caffe
caffe.__path__
caffe.__version__
```

* Matlab

直接通过`matlab`启动，目前版本为`R2016b`。

* PyTorch

系统目录下已经安装Python3版本的PyTorch，如果需要其他版本请自行使用`pip`或者`pip3`安装。

在python REPL里使用如下代码来测试你的PyTorch导入路径及版本
```python
import torch
torch.__path__
torch.__version__
```

* TensorFlow

系统目录下已经安装Python3版本的Tensorflow，如果需要其他版本请自行使用`pip`或者`pip3`安装。

在python REPL里使用如下代码来测试你的TF导入路径及版本
```python
import tensorflow as tf
tf.__path__
tf.__version__
```

* 其他框架及软件

...

## Question

Dr. はな
