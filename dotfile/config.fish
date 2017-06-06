
fortune-zh

function jl
	/home/lumin/Downloads/julia-6445c82d00/bin/julia $argv
end

function gitpushall
	git push; git push --all; git push --tags
end

set -g theme_powerline_fonts no
set -g theme_display_user yes

# start debian packaging block

function dquilt
	quilt --quiltrc=$HOME/.quiltrc-dpkg $argv
end

export DEBEMAIL="cdluminate@gmail.com"
export DEBFULLNAME="Zhou Mo"
export DEB_BUILD_OPTIONS=parallel=4
export BUILDER=pbuilder

export QUILT_SERIES=debian/patches/series
export QUILT_PATCHES=debian/patches

# end debian packaging block

export GIT_EDITOR=vim


cowsay -f stegosaurus "苦海无边，回头是岸"
# 不忘初心，方得始终 -- L.J.R.

