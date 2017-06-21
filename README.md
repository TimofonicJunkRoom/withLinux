## With Linux - Quite Messy Collection of Personal Notes

> KISS: Keep It Simple, Stupid  

Linux tricks, hints, hacks and many others based on my personal experience with
[Debian GNU/Linux](https://www.debian.org). This project is a personal knowledge
base, and some of the hints and hacks there are just recored by keywords.  

Lookup keywords within it with the perl utility `ack`. Or use the script
`search` to search for some tips as long as created an xapian database.
  
Items marked with `(-)` are still works in progress.  
Items marked with `(el)` means external link.  
  
Installing Linux  
---
1. [Install linux into an USB drive](./admin/install-linux-into-usb-stick.txt)  
2. [Tips about Drivers and firmware](./admin/dri)  
  1. [Graphic card](./admin/dri/graphic_card_driver.txt)  
  2. [Wireless network card](./admin/dri/wireless.txt)  
3. [Turning off graphic card](./admin/turn-off-gpu.txt)  
4. [Tips about Solid State Drive (SSD)](./admin/ssd.md)  
  1. [Gentoo wiki/SSD (el) ](https://wiki.gentoo.org/wiki/SSD)  
5. [Some packages related to EFI/UEFI](./admin/efi.md)  
6. [My old installation log for a HP server](./admin/hpserver)  
7. [Bootstrap minimal/stage3 Debian system](./admin/bootstrap)    
8. [Simple ArchLinux install note](./admin/arch.md)  
9. [mdadm note for creating software md5](./admin/mdadm.md)  
10. [Trying Gentoo is worthwhile (-) ](./admin/gentoo.md)  
1. [Gentoo: Sakaki's EFI Install Guide (el) ](https://wiki.gentoo.org/wiki/Sakaki%27s_EFI_Install_Guide)  
1. [Gentoo Prefix (el) ](https://wiki.gentoo.org/wiki/Project:Prefix)  
1. [ArchLinux's application list (el) ](https://wiki.archlinux.org/index.php/List_of_applications)  
1. [My dotfiles](./dotfile) and [its deploy script](./deploy)  
1. [(/boot + LUKS (LVM (/ + /home))) Linux installation](./admin/lvm-over-luks.md)  
1. [Alpine linux note](./admin/alpine.md)  
1. [OpenBSD note](./admin/openbsd.md)  
1. [Debian Sid root on ZFS](./admin/sidonzfs.md)  

Utilities / Miscellaneous  
---
1. [Example usage of netcat](./util/netcat.txt)  
2. [bash tricks](./util/bash_tricks.txt)  
3. [Commands for fun](./util/funny_commands.txt)  
6. [GPG short note](./util/short_gpg.md)  
7. [Data recovery tools](./util/data-recover.txt)  
8. [Remove data safely](./util/remove-data-safely.txt)  
9. [Encrypting disk with LUKS](./util/disk-crypt.txt)  
10. [Example usage of avconv, an alternative to ffmpeg](./util/avconv.txt)  
12. [Using vim, convert code into html](./util/vim_html.txt)  
13. [ffmpeg, resize picture and video](./util/ffmpeg_resize_picture.txt)  
14. [imagemagick, resize, trans-format, rotate picture](./util/imagemagick.txt)  
15. [xset, change keyboard input delay/rate under X](./util/keyrate)  
16. [ssh-agent, let it memorize your ssh password](./util/ssh-agent.txt)  
17. [Busybox, all-in-one software, developer works](http://www.ibm.com/developerworks/cn/linux/l-busybox/index.html)  
18. [JPEG integrity check](./util/jpeg-int.md)  
19. Wine  
  1. [(el) Arch: wine](https://wiki.archlinux.org/index.php/Wine)  
  1. [Terraria with Wine32, on Linux](./util/terraria.md)  
20. [Git ... content tracker](http://www.git-scm.com/)  
  1. [set up git ... github help](https://help.github.com/articles/set-up-git)  
  2. [a git tutor ... git immersion](http://gitimmersion.com/)  
  3. [git ... merge 2 repos into 1](./util/git_merge_repo.txt)  
1. [iceweasel and chromium ... cache config, of iceweasel (firefox)](./util/iceweasel-cache.txt)  
23. Log Analysis  
  1. awk programming language  
  2. apt-get search {visitors,awstat}  
24. [A long list of Linux utilities](util/util_list.md)  
1. [GPG best practices (el) ](https://help.riseup.net/en/security/message-security/openpgp/best-practices#self-signatures-should-not-use-sha1)  
1. [tmux / gnu screen note](./util/tmux.md)  
1. [(el) Task](http://taskwarrior.org) .. command line todo manager .. task-tutorial(5)  
  1. [nots on this task tool](./util/task.md)  
1. [GNU Utils - Powerful](./util/gnuutils.md)  
1. [Encrypt directory with ECryptfs](./util/ecryptfs.md)  
1. Document converter: pandoc
1. [Flash Solution for Debian](./util/flash.md)  
1. [package:fbi -- frame buffer image viewer, pdf is also browsable in FB]
1. zhcon -- Chinese character in tty
1. [cpio note](./util/cpio.md)  
1. fish/fishshell -- powerful shell (`apt install fish; chsh; fish_config`)  
1. [Vim](./util/vim.md)  
1. [Linux-Perf](./util/perf.md)  
1. [Terminology -- fancy terminal emulator](./util/terminology.md)  
1. [`fold -s txt` -- automatically wrap lines in your file]  
1. [kill all processes of user X `pkill -u X`]  

Graphical Interface Stuff (Xorg)  
---
1. Gnome3 (gnome-shell)  
  1. [change the height of top bar ... modify](./util/topbar.md)  
  2. [against BUG: alacarte empties gnome menu ... wheezy](./util/alacarte.txt)  
  3. X11 default display manager ... /etc/X11/default-display-manager  
  1. [gnome scaling factor](./util/gnome-scale.md)  
2. LXDE  
  1. [set shortcut keys under LXDE](./util/lxde-shortcut)  
3. Desktop Managers, X11 login program  
  1. [gdm3] and [kdm], too heavy  
  2. [sddm] for plasma5  
  3. [lightdm] and [slim] are light and small  
  4. [jwm], [cwm], [twm], [i3-wm], [dwm]  
5. Graphical Virtual terminals  
  1. [gnome-terminal], [guake], [uxterm], [lilyterm], [sakura], [st]  
1. Special task under graphical environment  
  1. [Input greek letters](./util/greek.md)  
1. package: numix-gtk-theme numix-icon-theme arc-theme  
2. [Xorg+dwm notes](./util/xorgdwm.md)  
1. [Extend your screen via network](./util/extend-screen-with-vnc.md)  

Internet Specific  
---
5. ifup and ifdown setting ... see interfaces(5)
2. [vpn ... set up on debian](./admin/vpn.txt)  
3. [ping ... no reply to icmp ping](./admin/ping.txt)  
4. [port ... what is running on a port](./admin/port.txt)  
6. [share the network with other machine ... one of cases](./admin/share-network-between-linux.txt)  
7. [share-ipv4-with-ipv6-tunnel ...](./admin/share-ipv4-with-ipv6-tunnel.txt)  
8. [openvpn ... simple utilization](./admin/openvpn.txt)  
9. [802.1X auth ... ref link](./admin/802auth)  
10. [iptables howto ... ubuntu help](https://help.ubuntu.com/community/IptablesHowTo)  
1. IPv6
  1. [miredo note ... IPv6 Net with Teredo](./admin/miredo.md)
  1. [isatap ... IPv6 with ISATAP](./admin/isatap.txt)
  1. [(el) microsoft: teredo](https://technet.microsoft.com/en-us/library/bb457042.aspx)  
13. [script for listing available wifi SSIDs](./admin/wifilist)  
14. HTTP status code reference  
  1. RFC2616  
  2. [list of http status codes ... wikipedia](http://en.wikipedia.org/wiki/List_of_HTTP_status_codes)  
15. Sharing files over network  
  1. [nc](./util/netcat.txt)  
  2. scp, ssh, rsync.  
  3. samba server.  
  4. http server, like apache2. OR [python](admin/python.txt)   
  5. ftp server, like vsftpd.  
  6. NFS  
16. Dynamic Name Service   
  1. [list of DNS record types ... useful with dig(1)](http://en.wikipedia.org/wiki/List_of_DNS_record_types)  
  2. [comparison of DNS server software](http://http://en.wikipedia.org/wiki/Comparison_of_DNS_server_software)  
1. [Restrict connections with Iptables](./admin/iptables-restrict-connections.md)  
1. [access intranet with ddns](./admin/intranet-ddns.md)  
1. [UFW -- uncomplicated firewall](./admin/ufw.md)  

System Management / Server  
---
1. [APT/DPKG](./admin/apt-dpkg.md)    
1. [tzdata ... change the system time zone](./admin/time_zone_change.txt)  
2. [grub2 ... location of config files](./admin/grub_config_file_location.txt)  
3. [Config runlevels (sysvinit) ...](./admin/runlevel.txt)  
4. [Font in tty ...](./admin/tty-font.txt)  
5. [Networkmanager ... can't change network settings?](./admin/networkmanager.txt)  
7. [Google earth ... install on amd64](./admin/gearth.txt)  
8. [grub ... boot with kali iso image](./admin/grub_kali_iso.txt)  
9. [time ... hardware time](./admin/hardwaretime.txt)  
10. [Silence ... no beep from machine](./admin/silent_beep.txt)  
11. [SMART, hard disk smart attributes](./admin/smart.md)  
12. [Steam ... install steam on amd64](./admin/steam.txt)  
13. [Sudo ...](./admin/sudo.txt)  
14. [Wine ... on amd64](./admin/wine.md)  
16. [font ... install font](./admin/font.txt)  
  1. [some open source fonts](./admin/font_list.txt)  
17. [systemd ... some reference link](./admin/systemd_link.txt)  
20. [sshd security ... sshguard](./admin/sshguard.txt)  
21. [sshd SFTP chroot ... ](./admin/ssh.md)  
22. [Ext2/Ext3/Ext4 on Windows](http://blog.csdn.net/hansel/article/details/7557033)  
  1. [ext2Fsd](http://www.ext2fsd.com/)  
  2. [Ext2Read](http://sourceforge.net/projects/ext2read/)  
  3. [Ext2IFS](http://www.fs-driver.org/)  
  4. [DiskInternals Linux Reader for Windows](http://www.diskinternals.com/linux-reader)  
1. [vsftpd ... a simple setup](./admin/BriefVsftpd.tex)  
2. [dnsmasq ... local cache name server](./admin/dnsmasq.md)  
3. [apache2 (2.2.22) ... simple setup](./admin/apache2.md)  
4. [bind9 (debian)... DNS (cache) - config file](./admin/named.conf.options)  
5. [hostapd ... dev work](http://www.ibm.com/developerworks/cn/linux/l-wifiencrypthostapd/index.html)  
6. [udhcpd from busybox]
7. [gitlab ... official deb setup](./admin/gitlab.md)  
1. [multiple ssh connection at the same time](admin/multiple-ssh.txt)  
1. [(el) WorldPress in Debian](https://wiki.debian.org/WordPress)  
1. [prevent `resolv.conf` from being changed](admin/static_resolv_conf.md)  
1. [Augment existing ext4 filesystem](./admin/ext4-extend-partition.md)  
1. [LVM simple note](./admin/lvm.md)  
1. [systemd note](./admin/systemd.md)  
1. [ZFS note](./admin/zfs.md)  
1. [Some RAID0 Restoring Experiment](./admin/raid0rescue.md)  
1. sshfs -- mount remote host to local directory
1. [keyboard setup](./admin/kbd.md)  
1. [Archlinux Build System (ABS)](./admin/abs.md)  

Linux Kernel Hacks / Operating System / Embedding / Hardware
---
4. [linux frozen ... handle with SysRq](./kernel/sysrq.txt)  
1. [Compile linux kernel... ](./kernel/compile.txt)  
2. [MIT Xv6 Original](https://pdos.csail.mit.edu/6.828/2011/xv6.html)  
4. [Cook a minimum bootable linux / initramfs ](./minisys/main.pdf)  
1. [hard disk link power manage ... sample hack](./kernel/hddpower.sh)  
2. [cpu freqency tweak ... sample hack](./kernel/cpufreq.sh)  
3. [backlight](./kernel/backlight.txt)  
1. [my OK6410 installation log](kernel/ok6410/main.pdf)  
1. [OpenPOWER fundation](http://openpowerfoundation.org/)  
1. cpuid -- dump cpu info  
1. hwinfo, lshw  to gather detailed system hardware information.  
1. i2c-tools (modprobe eeprom; decode-dimms) to obtain CAS latency.  
1. [VM note](./kernel/vm.md)  
1. [Emulating systems with QEMU](./kernel/qemu-emulate.md)  
1. [SATA Hard disk hotplugging](./kernel/hotplug.md)  
1. [syscall list -- man syscalls]

Virtualization  
---
1. Docker.io, the linux container  
  1. [note](./virt/docker.md)  
4. X86 dosbox emulator, (dosbox)  
1. QEMU/KVM  
  1. [QEMU/KVM -- nographic mode](./virt/qemu-nographic.md)  
1. Fake Virtualizations  
  1. traditional chroots  
  1. schroot: [Archwiki:schroot](https://wiki.archlinux.org/index.php/Install_bundled_32-bit_system_in_64-bit_system)[ Debianwiki:Schroot](https://wiki.debian.org/Schroot)    
    1. [schroot note](./virt/schroot.md)  
1. Virtualbox  
  1. [Extremely slow network transmission speed within loop between host and guest](./virt/virt-slow-transmission.md)  
  1. [Virtualbox Shared Directory](./virt/vbox-shared-dir.md)  

Cluster / HighPerf  
---
1. [storm local setup ... apache inqubator, storm](./parallel/setup-local-storm.txt)  
2. [PBS - Cluster Job Management](./parallel/pbs)  
3. [parallel computing note](./parallel/parallel.intro.txt)  
4. [(el) Debian wiki: High Performance Computing](https://wiki.debian.org/HighPerformanceComputing)  
  3. HTCondor (most convenient one in case to manage a single machine)   
    1. [My HTCondor wrapper script](./parallel/condor)  
    1. [my HTCondor note](/parallel/condor.md)  
  4. Other job control systems: Torque, SGE (sub grid engine), Slurm  
1. [Note on OpenMP and OpenBLAS](./parallel/omp_oblas.md)  
1. [BLAS](./parallel/blas.md)  
1. [LAPACK](./parallel/lapack.md)  

Kali / Security / Penetration  
---
1. [kali tools short list](./util/kali-tools.txt)  
2. [nmap note](./util/nmap.txt)  
3. [nping note](./util/nping.txt)  
4. [preventing buffer overflow](http://www.ibm.com/developerworks/cn/linux/l-sp/part4/index.html)  
5. [External resources](./util/resource.md) 
6. [ARP spoof](util/arpspoof.txt)  
7. [AirCrack](util/aircrack.txt)  
8. [aircrack-ng](util/aircrack-ng.txt)  
1. Disk Encryption  
  1. [(el) Gentoo: dm-crypt](https://wiki.gentoo.org/wiki/Dm-crypt)  
  1. [(el) Gentoo: dm-crypt LUKS](https://wiki.gentoo.org/wiki/DM-Crypt_LUKS)  
  1. [(el) Arch: dm-crypt](https://wiki.archlinux.org/index.php/Dm-crypt/System_configuration)  
  1. [(el) Debian: encrypt setup](http://madduck.net/docs/cryptdisk/)  
1. macchanger -- changes you mac address  
1. steghide -- steghide (1) -- forensics/steganography tool  
1. encryption with openssl -- see openssl (1), enc (1)  

[Debian GNU/Linux Specific](https://www.debian.org/)  
---
1. [(el) Bits from Debian](https://bits.debian.org/)  
1. [setup unofficial debian archive](./debian/unofficial_archive.txt)  
2. [setup debian mirror (el) ](https://www.debian.org/mirror/ftpmirror)  
1. Packaging and Policy  
  1. [Debian Policy Manual]
  1. [(el) Debian Science Policy Manual](http://debian-science.alioth.debian.org/debian-science-policy.html)  
  1. [(el) Debian Lua Package Policy Manual](http://pkg-lua.alioth.debian.org/policy.html)  
  1. [Debian Developer Reference]
  1. [Debian New Maintainer's Guide]
  1. [Mentors FAQ](https://wiki.debian.org/DebianMentorsFaq)  
  1. [python library style guide](https://wiki.debian.org/Python/LibraryStyleGuide)  
  1. [rpath issue](https://wiki.debian.org/RpathIssue)  
  1. [library packaging guide](https://www.netfort.gr.jp/~dancer/column/libpkg-guide/libpkg-guide.html)  
  1. [Debian Python Policy (el)](https://www.debian.org/doc/packaging-manuals/python-policy/)  
  1. [(el) Upstream Guide](https://wiki.debian.org/UpstreamGuide)  
  1. [Hardening (wiki)](https://wiki.debian.org/Hardening)  
  1. [How to get backtrace (wiki)](https://wiki.debian.org/HowToGetABacktrace)  
1. [Computer language benchmark Game (el)](http://benchmarksgame.alioth.debian.org/)[ its conclusion](http://benchmarksgame.alioth.debian.org/dont-jump-to-conclusions.html)    
1. [GPG: Keysigning](./debian/gpg.md)  
1. [GNU license list](https://www.gnu.org/licenses/license-list.html)  
1. [wikipedia: comparison of licenses](https://en.wikipedia.org/wiki/Comparison_of_free_and_open-source_software_licenses)  
1. [Debian CD/DVD hint](./debian/debiancd.md)  
1. [ftpmaster-removal](https://wiki.debian.org/ftpmaster_Removals)  
1. [Package Transition](https://wiki.debian.org/PackageTransition)  

[Looking For Help](http://google.com)  
---
wiki, doc, tutorial, and some interesting sites.

1. [Wikipedia](http://wikipedia.org)  
8. [Linux kernel document](https://www.kernel.org/doc), also shiped within kernel.tar.xz:/doc  
3. [Arch wiki](https://wiki.archlinux.org)  
2. [Debian wiki](https://wiki.debian.org)  
4. [Gentoo wiki](https://wiki.gentoo.org/wiki/Main_Page)  
9. [Gentoo doc](http://www.gentoo.org/doc)  
6. [Funtoo wiki](https://wiki.funtoo.org)  
5. [stackoverflow](http://stackoverflow.com)  
7. [IBM Developer works](https://www.ibm.com/developerworks/cn/linux/)  
10. [FreeBSD doc ... sometimes helps](https://www.freebsd.org/docs.html)  
12. [vbird.org ... detailed, complete linux guide](http://linux.vbird.org)  
13. [from windows to Linux, beginner ... IBM developer works](http://www.ibm.com/developerworks/cn/linux/l-roadmap/index.html)  
14. Search Engines  
  1. google  
  2. shodan  
1. [Awesome List](https://github.com/sindresorhus/awesome)  
1. [Matrix67 blog](http://www.matrix67.com/blog/)  
1. [TUNA Wiki](https://wiki.tuna.tsinghua.edu.cn/Homepage)  
1. [(el) Debian Admin](https://debian-administration.org/)  

Open Source Licenses
---
1. [gpl-faq](http://www.gnu.org/licenses/gpl-faq.html#NoticeInSourceFile)  

Programming, including Tool Languages  
---
1. Overview  
  1. [Osamu Aoki's Fun to Program](https://people.debian.org/~osamu/fun2prog.html)  
1. Compilation Tool Chain, GCC, Make, CMake, etc
  5. [GNU Make](http://www.gnu.org/software/make/manual/make.html)  
  6. [CMake](http://www.cmake.org/cmake-tutorial/)  
2. C Programming Language  
  1. [printf ... colourful text in terminal](./lang/c/printf_colour.c)  
  2. [crack a C program ... simple example](./lang/c/hexed/simple-hex-edit-binaries.txt)[(foo.c)](./lang/c/hexed/foo.c)[foo2.c](./lang/c/hexed/foo2.c)  
  3. [print source file name and source line number](./lang/c/file_line_.c)  
  4. [berkely db simple demo ... ](./lang/c/db.c) and [makefile](./lang/c/Makefile)  
  1. [libarchive example (el) ](https://github.com/libarchive/libarchive/wiki/Examples)  
3. C++  
  1. [glog demo program](./lang/cpp/glog.cpp) and its [makefile](./lang/cpp/glog.mk)  
  2. [print source file name and source line number](./lang/cpp/file_line_.cpp)  
  3. [a BLAS demo in cpp](./lang/cpp/blas.cpp) and [its makefile](./lang/cpp/Makefile)  
  4. [template demo](./lang/cpp/template.cpp) and [its makefile](./lang/cpp/Makefile)  
  5. [stl demo](./lang/cpp/stl.cpp)  
  6. [Qt helloworld](./lang/cpp_qt/)  
    1. [Qt blog (el) ](http://blog.51cto.com/zt/20/1/)  
  1. [Very brilliant reference site (el)](http://www.cplusplus.com/)  
  1. [protobuf demo](./lang/cpp/protobuf)  
1. tex/LaTeX  
  1. [use opentype font in tex](./lang/tex/tex-opentype-font.txt)  
  2. [xelatex ... chinese support, xeCJK](./lang/tex/xelatex.cjk.tex)  
  3. [pdflatex ... eng chs mixed sample tex](./lang/tex/eng_chs_mixed_sample.tex)  
  4. [finding the right font](http://www.tug.dk/FontCatalogue/)  
  5. [insert svg picture in latex](lang/tex/tex_svg.md)  
  1. [(el) NASA: latex help](http://www.giss.nasa.gov/tools/latex/)  
  1. [My presentation example](./lang/tex/presentation.tex)  
1. [Python3](./lang/py3)  
  1. [convert list into string](./lang/py3/list2str.md)  
  1. [my logging example in py3](lang/py3/logging_example.py)  
  1. [my glog-like logging lib in py3](./lang/py3/lumin_log_demo.py)  
  1. [HDF5 in python3: h5py demo](lang/py3/hdf5.py)  
  1. [Scipy/Numpy](./lang/py3/numpy.md)  
  1. [Lumin's Task Queue Daemon](./lang/py3/tq)  
  1. [flask hello world](./lang/py3/flask.hello.py)  
  1. [simple parallelization in python3](./lang/py3/para.py)  
  1. [python3 tutorial note](./lang/py3/tutorial/main.md)  
  1. [extend python3 with C lib](./lang/py3/extend/)  
  1. [my YouDao translation in terminal](./lang/py3/ydtrans/README.md)  
  1. [Decorator simple example](./lang/py3/decorator.py)  
  1. [simple use of cython3](./lang/py3/cython3/demo.py)  
  1. [(el) cython3 document](http://docs.cython.org/index.html)  
  1. [Curses hello world in python3](./lang/py3/pycurses.py)  
  1. [PLY - python lex yacc](./lang/py3/plytest.py)  
  1. [Python3 spider for image downloading](./lang/py3/spider/spider.py)  
  1. [Python3 advanced (webdriver) scrapper for image downloading](./lang/py3/spider/xpider.py)  
  1. [PyGame](./lang/py3/pygame/)  
  1. [python3 ffi/ctypes](./lang/py3/ffi/)  
  1. [jupyter-notebook (el)](http://jupyter.readthedocs.io/en/latest/running.html)  
1. [Octave/Matlab](./lang/oct)  
  1. [Simple FFT (recursive algorithm) in octave](./lang/oct/dsp/main.m)  
  1. [Notable differences between Octave and Matlab](./lang/oct/diff.md)  
1. [Lua](http://www.lua.org/)  
  1. [learn lua in 15 minutes](http://tylerneylon.com/a/learn-lua/)  
  1. [My lua logging module](./lang/lua/logging/lumin_log.lua)[demo](./lang/lua/logging/demo.lua)  
  1. [Torch7 interface note](./lang/lua/torch/main.md)  
  1. [lua embedding (el) ](http://www.ibm.com/developerworks/cn/linux/l-lua.html)  
  1. [2D Game engine -- love2d]
  1. [Programming in Lua (el)](https://www.lua.org/pil/contents.html)  
  1. [GC in lua](./lang/lua/gc.lua)  
1. Light weigt languages
  1. [Sed/Ed]  
  1. [AWK](./lang/awk/)  
    1. [GAWK Manual](http://www.gnu.org/software/gawk/manual)  
  1. [GNU BC/DC]     
1. [Doxygen ... Automatic document generation for C++ code](./lang/doxygen/)  
1. [Java]
  7. [java helloworld on linux](./lang/java)  
1. Maxima  
  1. [My maxima Note](./lang/maxima.md)  
1. [Go](./lang/go)  
1. [Perl](./lang/perl)  
  1. [Perl introduction](./lang/perl/intro.pl)  
1. Rust  
  1. [Learn RUST by example](http://rustbyexample.com/)  
  1. [My GLOG-like logging util in Rust](./lang/rust/logging/)  
1. [Julia]  
  1. [start.jl](./lang/jl/start.jl)  
1. [C#/Mono]  
  1. [C# hello world](./lang/csharp/hello.cs)  
1. [HTML/CSS/JS]
  1. [hello world html sample](./lang/sample.html)  
  2. [My html template, based on Debian apache default page](./lang/html/index.html)  
1. [GNU Plot]
  1. [visualizing data with gnu plot (el) ](http://www.ibm.com/developerworks/cn/linux/l-gnuplot/index.html)  
1. [SQL](./lang/sql/main.md)    
1. SHELL/BASH  
  1. [Lumin's log lib in bash](./lang/lumin_log.sh)  
  1. [BLAS Selector script using dialog](./lang/sh/blasselector.sh)  
1. CUDA  
  1. [cuda example](./lang/cuda/)  
  1. [cuda intro cu](./lang/cuda/src/intro.cu)  
1. [IBM CPlex]
1. Online judges
  1. http://poj.org/  
  1. https://leetcode.com  
1. Collections  
  1. My Logging Libraries, see [Rainbowlog](https://github.com/CDLuminate/rainbowlog)  
    1. [In C](https://github.com/CDLuminate/cda/tree/master/lib)  
    1. [In Lua](./lang/lua/logging/lumin_log.lua)  
    1. [In BASH Shell](./lang/lumin_log.sh)  
    1. [In Python3](./lang/py3/lumin_log_demo.py)  

Tiny Bits about Artificial Intelegence  
---
1. Random Mathematical Bits  
  1. [Wikipedia: Embedding (el) ](https://en.wikipedia.org/wiki/Embedding)  
1. Wolfram  
  1. [Introduction to Wolfram language (el)](http://www.wolfram.com/language/elementary-introduction/)  
  1. [Wolfram Alpha (el)](http://www.wolframalpha.com/)  
  1. [Wolfram Mathworld (el)](http://mathworld.wolfram.com/)  
1. Machine Learning  
  1. [stanford:Ng's opencourse (el) ](http://openclassroom.stanford.edu/MainFolder/CoursePage.php?course=MachineLearning)  
1. Some Deep Learning Frameworks  
  1. [caffe (cxx, python, matlab)](https://github.com/BVLC/caffe)  
  1. [torch7 (lua)](https://github.com/torch/distro) [-- torch cheatsheet --](https://github.com/torch/torch7/wiki/Cheatsheet)  
  1. [theano (python)](http://deeplearning.net/software/theano/) [-- my theano note --](ai/theano/main.md)  
    1. [theano issue, no recursion is supported](./ai/theano/fac.py)  
  1. [tensorflow (python)](https://github.com/tensorflow/tensorflow)  
  1. [mxnet (cxx, julia)](https://github.com/dmlc/mxnet)  
  1. [leaf (rust)](https://github.com/autumnai/leaf) [-- Deep learning frameworks benchmark by leaf --](http://autumnai.com/deep-learning-benchmarks)  
  1. [Chainer, Define-by-Run instead of Define-and-Run (python)](https://github.com/pfnet/chainer)  
1. Deep learning
  1. http://deeplearning.stanford.edu/  
  1. https://devblogs.nvidia.com/parallelforall/deep-learning-nutshell-core-concepts/  
1. Preprint
  1. [Arxiv](http://arxiv.org/)  
1. Datasets  
  1. [MS COCO](https://github.com/CDLuminate/cocofetch)  
  1. [Imagenet](http://image-net.org/index)  
1. [WikiCFP](http://www.wikicfp.com/cfp/)  
  
## LICENSE
```
MIT LICENSE.
COPYRIGHT (C) 2014,2015,2016,2017 Lumin
```  
---
Started on 2014/06/28 by Lumin Zhou  
  
Seek for UNIX, look into UNIX, follow the beats of UNIX, play with UNIX,
work with UNIX, learn from UNIX, but neither become an UNIX, nor marry UNIX.
