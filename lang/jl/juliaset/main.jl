# Calculate Julia Set with Julia.
#
# The resulting image will be saved in PPM format to file result.ppm
# This ppm file can be viewed directly, or you can convert it into
# another format like this: $ ffmpeg -i result.ppm -vcodec png output.png
# 
# Copyright © Zhou Mo <cdluminate AT gmail.com>
# MIT License

# configure
C = -0.62772 -0.42193im
xmin = -1.8
xmax = 1.8
ymin = -1.8
ymax = 1.8
samples = 2000

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

# prepare
@printf("Preparing Matrices\n")
Z = complex(zeros(samples, samples))
output = zeros(samples, samples)
for (i,x) in enumerate(linspace(xmin, xmax, samples))
   for (j,y) in enumerate(linspace(ymin, ymax, samples))
      # FIXME: why should I rotate the resulting image like this?
      Z[j,i] = x + y * im
   end
end

# calculate
@printf("Calculating Julia Set\n")
# FIXME: why doesn't @parallel for i = 1:samples work ?
for i = 1:samples
   #@printf(" -> iteration %d\n", i)
   for j = 1:samples
      output[i,j] = getCount(Z[i,j], C)
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
