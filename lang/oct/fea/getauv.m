
function ret = getauv (points, a, b, c)
    x1 = points(a, 1);
    y1 = points(a, 2);
    x2 = points(b, 1);
    y2 = points(b, 2);
    x3 = points(c, 1);
    y3 = points(c, 2);
    ret = gette(x1, y1, x2, y2, x3, y3);
    ret = ret.' * ret;
    ret = ret .* getae(x1, y1, x2, y2, x3, y3);
end