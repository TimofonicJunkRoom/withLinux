Some solutions to the Euler project
===

# [1](https://projecteuler.net/problem=1)

```math
\sum_i ia + \sum_j jb  \text{ ,where } i \neq nb , j \neq ma
```

```julia
julia> @time sum([ i for i in filter(x -> (x%3==0) || (x%5==0), 1:999) ])
0.009719 seconds (1.21 k allocations: 29.043 KB)
233168
```

equivalent to 
```math
\sum_i ia + \sum_j jb - \sum_k kab
```
where kab is the repeated numbers among ia and jb.

```julia
julia> sum([ 3:3:999; 5:5:999; -(15:15:999) ])
233168
```

# [18](https://projecteuler.net/problem=18) / [67](https://projecteuler.net/problem=67)

```
In a triangle like this:

      a
     b c
    d e f
    
the best way to find the anwser is not to get the maximum from the summaries of
all possible branches from top to bottom.

There is such a recursive pattern

a + max( b+max(b,c), c+max(e,f)
```

`tri.txt`
```
75
95 64
17 47 82
18 35 87 10
20 04 82 47 65
19 01 23 75 03 34
88 02 77 73 07 63 67
99 65 04 28 06 16 70 92
41 41 26 56 83 40 80 70 33
41 48 72 33 47 32 37 16 94 29
53 71 44 65 25 43 91 52 97 51 14
70 11 33 28 77 73 17 78 39 68 17 57
91 71 52 38 17 14 91 43 58 50 27 29 48
63 66 04 68 89 53 67 30 73 16 69 87 40 31
04 62 98 27 23 09 70 98 73 93 38 53 60 04 23
```

```julia
import Base.zero
zero(::SubString{String}) = 0 # Julia 0.5

ZeroString(::SubString{String}) = 0
ZeroString(x::Int64) = x

A = readdlm("tri.txt")
A = ZeroString.(A)

function myreduction(m)
	if size(m)[1] == 1
		return m[1,1]
	else
		mprime = m[1:(end-1), :]
		for k in 1:(size(m, 1)-1)
			mprime[size(mprime, 1), k] += max(
									 m[size(m, 1),k], m[size(m, 1),k+1])
		end
		return myreduction(mprime)
	end
end

println(myreduction(A))
```

# [69](https://projecteuler.net/problem=69)

Euler totient function looks like
```math
\varphi(n) = n \prod_{p|n} ( 1 - \frac{1}{p} )
```

To find the solution n* which maximizes our object function
```math
\text{max} \frac{n}{\varphi(n)} = \frac{1}{ \prod\limits_{p|n} (1-\frac{1}{p}) }, n \leq 1000000
```

is equivalent to
```math
\text{min} \prod_{p|n} (1-\frac{1}{p}), n \leq 1000000
```

Distinct prime factors $`p_i \in \{p|n\}`$ are always positive integers that are larger than 1,
hence $`0 < 1-\frac{1}{p} < 1`$ always holds. To minimize the above object function, we need
as many distince prime factors as possible from the number n*. Now we comprehend this problem
as to figure out a integer n* where n* <= 1000000 and has the most distinct prime factors
among the ingeters less or equal to itself.

Let's think about this problem in the reverse direction. The most ideal integer for this problem
should ship all possible primes, e.g. $`n^* =\prod([2,3,5,7,11,\ldots])`$. Moreover, there are infinite
number of primes, and the constraint $`n\leq 1000000`$ is exactly telling us when we should stop
the infinite production.

Code (Julia)
```julia
for i in 1:20
    @printf "%d\t%8d\t%s\n" i prod(primes(i)) "$(prod(primes(i))<1_000_000)"
end
```

Output
```
1	       1	true
2	       2	true
3	       6	true
4	       6	true
5	      30	true
6	      30	true
7	     210	true
8	     210	true
9	     210	true
10	     210	true
11	    2310	true
12	    2310	true
13	   30030	true
14	   30030	true
15	   30030	true
16	   30030	true
17	  510510	true
18	  510510	true    *
19	 9699690	false
20	 9699690	false
```