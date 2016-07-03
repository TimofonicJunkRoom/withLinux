
function ret = zexpand(mat33, idx1, idx2, idx3)
   ret = zeros(9, 9);
   ret([idx1, idx2, idx3], [idx1, idx2, idx3]) = mat33;
end