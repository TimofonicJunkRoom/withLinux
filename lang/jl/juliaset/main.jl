# Calculate Julia Set with Julia.
#
# https://en.wikipedia.org/wiki/Julia_set      for "julia set"
# http://mathworld.wolfram.com/JuliaSet.html   for "julia set"
# https://en.wikipedia.org/wiki/Netpbm_format  for "ppm picture format"
#
# The resulting image will be saved in PPM format to file result.ppm
# This ppm file can be viewed directly, or you can convert it into
# another format like this:
#  $ ffmpeg -i result.ppm -vcodec png output.png
# This command provides performance information.
#  $ sudo /usr/bin/time perf stat julia main.jl
# 
# Copyright © Zhou Mo <cdluminate AT gmail.com>
# MIT License

# configure
C = -0.62772 -0.42193im
#C = 0.285 + 0im
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
function getPPMHeader(width, height)
   # Magic "P2" for gray PPM picture
   return "P2
   $(width) $(height)
   255
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

# calculate
output = zeros(samples, samples)
@printf("Calculating Julia Set\n")
for (i,x) in enumerate(linspace(xmin, xmax, samples))
   for (j,y) in enumerate(linspace(ymin, ymax, samples))
      # FIXME: why should I rotate the resulting image like this?
      output[j,i] = getCount(x + y * im, C)
   end
end

# post process
@printf("Post processing\n")
output = round(UInt8, 255*normalize(output))

# save picture
@printf("Saving result to result.ppm\n")
f = open("result.ppm", "w+")
write(f, getPPMHeader(samples, samples))
writedlm(f, output, " ")
close(f)

@printf("Done\n")
