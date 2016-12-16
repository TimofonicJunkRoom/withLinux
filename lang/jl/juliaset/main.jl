#!/usr/bin/julia

# Calculate Julia Set with Julia.
#
# The resulting image will be saved in PPM format to file result.ppm
# This ppm file can be viewed directly, or you can convert it into
# another format like this:
#   $ ffmpeg -i result.ppm -vcodec png output.png
# This command provides performance information.
#   $ sudo /usr/bin/time perf stat julia main.jl
#
# https://en.wikipedia.org/wiki/Julia_set      for "julia set"
# http://mathworld.wolfram.com/JuliaSet.html   for "julia set"
# https://en.wikipedia.org/wiki/Netpbm_format  for "ppm picture format"
# 
# Copyright © Zhou Mo <cdluminate AT gmail.com>
# MIT License

# configure
#C = -0.62772 -0.42193im
C = 0.285 + 0im
xmin = -1.8
xmax = 1.8
ymin = -1.8
ymax = 1.8
samples = 2000
@printf("%s", "Dump configuration
 -> C       = $(C)
 -> xrange  = [ $(xmin), $(xmax) ]
 -> yrange  = [ $(ymin), $(ymax) ]
 -> samples = $(samples)
")

# helper functions
function getPPMHeader(width, height, magic="P2")
   # Magic "P2" for gray PPM picture
   return "$(magic) $(width) $(height) 255
   "
end

function getCount(z, c, maxiter=255, threshold=2)
   n = 0
   while abs(z)<threshold && n < maxiter
      z = z*z + c
      n += 1
   end
   return n
end

function normalize(mat)
   return (mat - minimum(mat)) / (maximum(mat) - minimum(mat))
end

function fakecolorize(mat)
   I, J = size(mat)
   ret = zeros(UInt8, I, J*3)
   for i = 1:I
      for j = 1:J
         # ret[i, (j-1)*3+1:j*3] = mat[i,j]
         needle = round(UInt8, mat[i,j])
         value = [ 0, 0, 0 ]
         if needle <= 63
            value = [ 0, 4*needle, 255 ]
         elseif needle <= 127
            value = [ 0, 255, 511-4*needle ]
         elseif needle <= 191
            value = [ 4*needle-511, 255, 0 ]
         else
            value = [ 255, 1023- 4*needle, 0 ]
         end
         ret[i, (j-1)*3+1:j*3] = value
      end
   end
   return ret
end

# calculate
output = zeros(samples, samples)
@printf("Calculating Julia Set\n")
for (i,x) in enumerate(linspace(xmin, xmax, samples))
   @printf(" -> progress: %.2f\n", i/samples)
   for (j,y) in enumerate(linspace(ymin, ymax, samples))
      # FIXME: why should I rotate the resulting image like this?
      output[j,i] = getCount(x + y * im, C)
   end
end

# post process
@printf("Post processing\n")
output = round(UInt8, 255*normalize(output))

# save picture
#@printf("Saving gray result to result.gray.ppm\n")
#f = open("result.gray.ppm", "w+")
#write(f, getPPMHeader(samples, samples))
#writedlm(f, output, " ")
#close(f)

@printf("Saving color result to result.color.ppm\n")
fc = open("result.color.ppm", "w+")
write(fc, getPPMHeader(samples, samples, "P3"))
writedlm(fc, fakecolorize(output), " ")
close(fc)

@printf("Done\n")
