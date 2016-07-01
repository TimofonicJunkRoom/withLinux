#!/usr/bin/julia
# vim syntax file available at:
#   https://github.com/JuliaLang/julia-vim
# @reference file:///usr/share/doc/julia-doc/html/index.html
# @package julia-doc
# julia version 0.4.6

# customize rc file
msg = "\$ echo Greetings > ~/.juliarc.jl"
println(msg)

# command line arguments
for x in ARGS;
  println(x);
end

## Getting started
1 + 2
# variable 'ans' is only available in interactive
# shell.
println("invoke julia --help")

# variables, UTF-8 supported
x = 10
println(x)
x = "Hello world"
println(x)
π = 3.1415926
println(π)
# in shell you can type \alpha+TAB to obtain real alpha
# integer and floating numbers
println(pi)

# integer and float
println(typeof(1)) # on amd64 it's Int64
println(WORD_SIZE) # on amd64 it's 64
println(typeof(3000000000)) # automated type
0x1 # 0x01 UInt8
0x123 # 0x0123 UInt16
0x1234567 # 0x01234567  UInt32
0x123456789abcdef # 0x0123456789abcdef UInt64
0b10 # 0x02 UInt8
0o10 # 0x08 UInt8
println((typemin(Int32), typemax(Int32)))
for T in [Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128]
   println("$(lpad(T,7)): [$(typemin(T)),$(typemax(T))]")
end

x = typemax(Int64)
x + 1 == typemin(Int64) # overflow

Float32(-1.5) # -1.5f0
0x.4p-1 # 0.125 Float64
sizeof(Float16(4.)) # 2
10_000, 0.000_000_005, 0xdead_beef, 0b1011_0010
#> (10000,5.0e-9,0xdeadbeef,0xb2)

0.0 == -0.0 # true
println(bits(0.0))
# "0000000000000000000000000000000000000000000000000000000000000000"
println(bits(-0.0))
# "1000000000000000000000000000000000000000000000000000000000000000"

1/Inf # 0.0 
1/0 # Inf 
-5/0 # -Inf 
0.000001/0 # Inf 
0/0 # NaN 
500 + Inf # Inf 
500 - Inf # -Inf 
Inf + Inf # Inf 
Inf - Inf # NaN 
Inf * Inf # Inf 
Inf / Inf # NaN 
0 * Inf # NaN

(typemin(Float16),typemax(Float16)) #(-Inf16,Inf16) 
(typemin(Float32),typemax(Float32)) #(-Inf32,Inf32) 
(typemin(Float64),typemax(Float64)) #(-Inf,Inf)

println(eps(Float64)) # machine epsilon
x = 1.25f0 #1.25f0
nextfloat(x) #1.2500001f0
prevfloat(x) #1.2499999f0
bits(prevfloat(x)) #"00111111100111111111111111111111"
bits(x) #"00111111101000000000000000000000" 
bits(nextfloat(x)) #"00111111101000000000000000000001"

1.1 + 0.1 # 1.2000000000000002

println(BigInt(typemax(Int64)) + 1) # GNU GMP, MPFR
println(BigFloat(2.0^(1000/3))) # GNU GMP, MPFR
parse(BigInt, "123456789012345678901234567890") + 1
parse(BigFloat, "1.23456789012345678901")
BigFloat(2.0^66) / 3
factorial(BigInt(40))

d = 3
println(2d^3-.5d+1+5^2d-1.3(d-1)d)
println((d-1)d) # but d(d-1) raises error

zero(Float32) # 0.0f0 
zero(1.0) # 0.0
one(Int32) # 1
one(BigFloat) 

# Math operators
println(isnan(0/0))
println(isnan(NaN))

-0.0 == 0.0 # true
isequal(-0.0, 0.0) # false
println(1 < 2 <= 2 < 3 == 3 > 2 >= 1 == 1 < 3 != 5)

# Complex and Rational numbers
# TODO

# Strings
# TODO

# Functions
# TODO
function f(x,y)
   x+y
end
f(x, y) = x + y
println(map(round, [1.2, 3.5, 1.7]))

# Control flow
# TODO

# Scope of Variables
# TODO

# Types
# Methods
# Constructors
# Conversion and Promotion
# Interfaces
# Modules
# Documentation
# Metaprogramming
# Multi-dimensional Arrays
# Linear algebra
# Networking and Streams
# Parallel Computing
# Date and DateTime
# Running External Programs

# Calling C and Fortran Code
# TODO
# Requires no glue code, but you compile C or fortran
# code with -shared and -fPIC options.
t = ccall( (:time, "libc.so.6"), Int32, ())
println(t)

# Interacting With Julia
# Embedding Julia
# Packages
# Profiling
# Performance Tips
# Style Guide
# Frequently Asked Questions
# Noteworthy Differences from other Languages

# Unicode Input
# Essentials
# Collections and Data Structures
# Mathematics
# ...

quit()
