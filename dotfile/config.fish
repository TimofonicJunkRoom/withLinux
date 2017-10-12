                                                                # - Variables -
export GIT_EDITOR=vim
#export _JAVA_AWT_WM_NONREPARENTING=1
set PATH $PATH $HOME/.local/bin $HOME/bin

                                                                  # - Helpers -
function aup
   sudo apt update
   sudo apt list --upgradable
end

alias julia="jl"
alias ip="ip -c"

abbr -a aug sudo apt upgrade
abbr -a gpa "git push; git push --all; git push --tags"
abbr -a off systemctl poweroff
abbr -a rg ranger
abbr -a sv sudo supervisorctl
abbr -a bt bluetoothctl
abbr -a nb jupyter notebook --no-browser
abbr -a px ps -ux

abbr -a p ipython3
abbr -a pp pypy3 -m IPython
abbr -a bp bpython3
abbr -a jqt jupyter-qtconsole
abbr -a ap autopep8 -i

# Julia 0.6 Official Binary Tarball
function jl
	/home/lumin/Downloads/julia-903644385b/bin/julia $argv
end
# Pypy3
function pypy3
	/home/lumin/Downloads/pypy3-v5.8.0-linux64/bin/pypy3 $argv
end

function ips
	ip -s -c -h a
end

# omf install grc
# fix grc behaviour
alias ls="ls --color"
alias findmnt="grc findmnt"
alias lsblk="grc lsblk"
alias lsmod="grc lsmod"
alias lspci="grc lspci"
alias stat="grc stat"
alias env="grc env"
alias lsof="grc lsof"
alias uptime="grc uptime"
alias ss="grc ss"
alias iptables="grc iptables"
alias id="grc id"
alias df="grc df -h"
alias mtr="mtr -t"

                                                         # - debian packaging - 
function dquilt
	quilt --quiltrc=$HOME/.quiltrc-dpkg $argv
end

export DPKG_COLORS=always
export DEBEMAIL="cdluminate@gmail.com"
export DEBFULLNAME="Zhou Mo"
export DEB_BUILD_OPTIONS=parallel=4
export BUILDER=pbuilder

export QUILT_SERIES=debian/patches/series
export QUILT_PATCHES=debian/patches

                                                                # - Greetings -
# Don't break Rsync, Scp or something alike. You can use this script to
# test whether fish is writting something to terminal.
# foo.fish:
# | echo extra stuff
# Then $ fish foo.fish
#
# https://github.com/fish-shell/fish-shell/issues/3473#issuecomment-254599357
# https://superuser.com/questions/679498/specifying-an-alternate-shell-with-rsync
if status --is-interactive
   cat ~/withlinux/dotfile/rose_ascii.co.uk
   fortune-zh
   grc uptime
end
