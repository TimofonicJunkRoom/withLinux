==============
服务器使用提示
==============

:Author: Copyright (C) 2017 Hakase Hana
:Date: Oct 20 2017

System and Environment Overview
===============================

服务器系统为不更新的 ``Debian Sid`` ，基本特性与 ``Ubuntu`` 近似。
如果您需要在个人计算机上安装Linux并期望在科学计算方面糙快猛，
可以考虑 ``Archlinux`` 。

出于安全考虑，不发放 ``Root`` 权限，并且设置 ``iptables`` 屏蔽
``ssh`` 外所有端口输入。理论上所有机器学习，深度学习，计算机视觉，
以及任何其他常规科学计算程序，均可以在非特权模式下使用，具体安装
方式因情况而异。

系统默认设置每个用户的默认Shell是 ``bash`` 。
对于需要长时间在终端下工作的情况，可以尝试学习使用 ``fish``或者
``zsh`` 任意一个现代化Shell来提升操作效率。

其余各项Linux基础知识请参见 *鸟哥的私房菜* http://linux.vbird.org 。

Basic and Important Command Lines
=================================

这里列举部分主要Linux基础命令，以下列出的工具参考了Debian的
``important`` , ``required`` 和 ``standard`` 三类Priority的软件集合，
并去除了管理员部分，只保留用户常用部分。

* 最重要的自救技能是，主动使用 ``man`` 查看帮助或者 ``xxx --help``

* 基础文件操作和POSIX权限相关

.. code:: bash

  列出当前目录内容 ls -lh
  查看文件详情     stat 
  创建文件         touch
  编辑文件         nano 或 vim 或 emacs
  删除文件         rm (Dangerous)
  删除文件夹       rmdir
  移动文件或者重命名文件 mv (Dangerous)
  复制文件         cp
  软链接           ln
  打印文本文件     cat
  查看文件头       head
  查看文件尾       tail
  文件查看器       less
  排序             sort
  去重             uniq
  文件行数统计     wc
  下载             wget <URL>
  POSIX文件权限    chmod chown chgrp
  同步磁盘缓存     sync
  判断文件类型     file
  计算md5哈希      md5sum
  归档，(解)压缩   tar gzip bzip2 xz

* Shell与进程相关

.. code:: bash

  本地终端         bash
  远程终端         ssh
  查看用户进程     ps ux
  结束进程         kill PID
  资源监视         top
  现代化资源监视   htop
  屏蔽SIGHUP       nohup

* 进阶生产力，其中版本控制非常重要

.. code:: bash

  版本控制         git
  流编辑器         sed
  文件查找工具     find
  内容查找工具     grep

* 编译器们，解释器们，以及编译系统

.. code:: bash

  Awk解释器        awk
  C编译器          gcc clang
  C++编译器        g++ clang++
  CUDA编译器       nvcc
  Go编译器         go
  Julia:
  Lua解释器        lua luajit
  Python解释器     python python3
  Perl解释器       perl
  Ruby解释器       ruby
  Rust编译器       rustc
  Scala            scala

  编译系统         make cmake

Infrastructure
==============

* IP Address

IP地址因为校园网PPPoE所以会不定期变化，跟本机设置完全没有任何关系，
此题无解。

* CUDA

服务器目前CUDA版本为 ``8.0.44`` ， 使用 ``nvidia-smi`` 查看显卡使用状态。

编译CUDA程序 **必须** 根据相应的编译系统调整默认编译器为 ``Clang-3.8``
或者 ``GCC-5`` ，否则必定编译失败。

* HDF5

服务器HDF5版本为1.10.0，使用1.8.0版本的程序可能出现兼容问题。

* OpenBLAS / Intel MKL

使用时请务必确认线程数环境变量正确设置，否则可能导致计算性能严重不符合预期。

* Python

服务器同时有 ``python 2.7`` 和 ``python 3.5`` , ``python 3.6`` 。
默认符号链接指向为：

.. code:: bash

  $ python        # python2.7
  $ python3       # python3.5

平时建议使用 ``ipython`` 或 ``ipython3`` 进行工作。
出于安全设置原因，远程使用 ``jupyter notebook`` 是不可行的。

当系统和家目录中散布存在着同一个包的不同版本时，请一定要
注意调整 ``PYTHONPATH`` 以避免惊喜。此变量有时也能帮助检查BUG。

对于仅使用Python标准库的程序可以考虑使用Pypy进行加速。对于更高性能需求的
程序建议考虑C/C++模块，SWIG，Cython等解决方案。

* Git

请务必使用Git管理好自己的源代码，服务器绝对不是100%可靠的。

* 用户之间的文件共享

目前家目录权限设置为 ``0770`` 。需要通过chmod修改文件的 ``g`` 权限，
其他用户即可访问。

* 显示图像

正常情况下使用看图软件查看服务器上的的图像需要首先进行ssh的X转发。
然而使用X转发的带宽要求和延迟要求非常高，耗费流量。X转发操作不一定
支持Windows。

.. code:: bash

  $ ssh -X user@hostname

在网络条件恶劣，或者无法进行X转发的情况下，也有一些工具可以把图像
转换成彩色Unicode字符，从而直接以字符形式，低分辨率地将图像打印到终端。
比如神奇的

.. code:: bash

  $ catimg xxx.jpg

类似工具可以在StackOverflow上查到很多。

Machine/Deep Learning Softwares
===============================

* Caffe

服务器系统自带CUDA版本Caffe（但未使用cuDNN），并带有Python3接口。
需要cuDNN（提速大概至少4倍？）请务必自行重新编译。
需要matcaffe请自行重新编译。
Caffe编译建议使用OpenBLAS，默认编译器建议使用Clang-3.8或者GCC-5。

在python REPL里使用如下代码来测试你的Caffe导入路径及版本

.. code:: python

  import caffe
  caffe.__path__
  caffe.__version__

* Matlab

直接通过 ``matlab`` 启动，目前版本为 ``R2016b`` 。

* PyTorch

系统目录下已经安装Python3版本的PyTorch，如果需要其他版本请自行使用
``pip`` 或者 ``pip3`` 安装。

在python REPL里使用如下代码来测试你的PyTorch导入路径及版本

.. code:: python

  import torch
  torch.__path__
  torch.__version__

* TensorFlow

系统目录下已经安装Python3版本的Tensorflow，如果需要其他版本请自行使用
``pip`` 或者 ``pip3`` 安装。

在python REPL里使用如下代码来测试你的TF导入路径及版本

.. code:: python

  import tensorflow as tf
  tf.__path__
  tf.__version__

* 其他框架及软件

略
