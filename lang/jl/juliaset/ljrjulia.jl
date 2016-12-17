#const C = -0.62772 -0.42193im
const C = 0.285 + 0.0im
const xmin = -2.
const xmax = 2.
const ymin = -2.
const ymax = 2.
const samples = 2000
@printf("%s", "Dump configuration
 -> C       = $(C)
 -> xrange  = [ $(xmin), $(xmax) ]
 -> yrange  = [ $(ymin), $(ymax) ]
 -> samples = $(samples)\n")

function gen_im!(file::String, xmin::Real, xmax::Real, ymin::Real, ymax::Real, samples::Int, C::Complex)
    y = ((ymin-ymax)/samples)
    x = ((xmax-xmin)/samples)
    f = open(file, "w")
    imgrange = ymax:y:ymin-y
    relrange = xmin:x:xmax-x
    write(f, "P2\n# Julia Set image\n$(samples) $(samples)\n255\n")
    @simd for im in imgrange
        for re in relrange
            z = complex(re, im)
            n = 255
            while abs(z) < 10 && n >= 5
                z = z*z + C
                n -= 5
            end
            write(f, "$n ")
        end
        write(f, "\n")
    end
    close(f)
end

gen_im!("julia.pgm", xmin, xmax, ymin, ymax, 200, C)
t = @elapsed gen_im!("julia.pgm", xmin, xmax, ymin, ymax, samples, C)
println("\n$(t)s\n")
