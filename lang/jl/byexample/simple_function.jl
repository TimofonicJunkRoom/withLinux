
# classical function
function volume_sphere(r)
   # $$ V = \frac{4}{3} \pi r^3 $$
   return (4*pi*r^3)/3
end

vol = volume_sphere(3.0)
@printf "volume = %0.3f\n" vol

# simpler function definition
linear(w, x, b) = (w * x + b)

a = [1, 2]; b = [3, 4]'; c = 0;
println(linear(b, a, c))

# specify argument datatype
function recv_Int64(a::Int64)
   a, a*2  # return multiple values with tuple
end

res1, res2 = recv_Int64(Int64(4))
@printf "res1 %ld res2 %ld\n" res1 res2
