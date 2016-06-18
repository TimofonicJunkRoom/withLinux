#!/usr/bin/julia
# vim syntax file available at: https://github.com/JuliaLang/julia-vim
# @reference file:///usr/share/doc/julia-doc/html/index.html
# @package julia-doc

# customize rc file
msg = "\$ echo Greetings > ~/.juliarc.jl"
println (msg)

# command line arguments
for x in ARGS;
  println (x);
end

# variables
x = 10
println(x)
π = 3.1415926
println(π)

# integer and floating numbers
x = int128(32453243)
print(bin(132415345), '\n')
y = float64(1.01)
println(y)
println(typeof(y))
println("Word size of your system is ", WORD_SIZE)
a = 0x1 # Uint8
b = 0b10 # Uint8
println("Int128 typemin ", typemin(Int128), " typemax ", typemax(Int128))
c = 2.5e-4
println(sizeof(float16(4.)))
println(bits(0.0))
println(bits(-0.0))
println(1/Inf)
println(1/0)
println(0/0)
println(eps(Float64))
println(BigInt(typemax(Int64)) + 1) # GNU GMP, MPFR
println(BigFloat(2.0^(1000/3))) # GNU GMP, MPFR
d = 3
println(2d^3-.5d+1+5^2d-1.3(d-1)d)

# Math operators
# TODO

quit()
