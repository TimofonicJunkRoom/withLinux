%% Get T^{(e)}
function ret = gette (x1, y1, x2, y2, x3, y3)
   ret = [ y2-y3, y3-y1, y1-y2; x3-x2, x1-x3, x2-x1 ];
   ret = ret ./ (2 * getae(x1, y1, x2, y2, x3, y3));
end
