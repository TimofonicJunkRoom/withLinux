                                                                # - Variables -
export GIT_EDITOR=vim
#export _JAVA_AWT_WM_NONREPARENTING=1

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

# Julia 0.6 Official Binary Tarball
function jl
	/home/lumin/Downloads/julia-903644385b/bin/julia $argv
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

                                                         # - debian packaging - 
function dquilt
	quilt --quiltrc=$HOME/.quiltrc-dpkg $argv
end

export DEBEMAIL="cdluminate@gmail.com"
export DEBFULLNAME="Zhou Mo"
export DEB_BUILD_OPTIONS=parallel=4
export BUILDER=pbuilder

export QUILT_SERIES=debian/patches/series
export QUILT_PATCHES=debian/patches

                                                                # - Greetings -
fortune-zh
#cowsay -f stegosaurus "苦海无边，回头是岸"
# 不忘初心，方得始终 -- L.J.R.
grc uptime
