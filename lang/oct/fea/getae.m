%% Get A^{(e)}
function ret = getae (x1, y1, x2, y2, x3, y3)
   ret = (y2-y3)*x1 + (y3-y1)*x2 + (y1-y2)*x3;
   ret = ret ./ 2;
   ret = abs(ret);
end