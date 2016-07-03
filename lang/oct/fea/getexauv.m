
function ret = getexauv(points, idx1, idx2, idx3)
    ret = zexpand(getauv(points, idx1, idx2, idx3), idx1, idx2, idx3);
end