function [s_cdt]=CDT(x,J,x_cdt,rm_edge)

if (size(J,2) == 1) 
    J=J';
end
if (size(x,2) == 1) 
    x=x';
end
if (size(x_cdt,2) == 1) 
    x_cdt=x_cdt';
end

J=J/sum(J); 
cJ=cumsum(J);

s_cdt=interp1(cJ,x,x_cdt,'pchip');

if rm_edge
    s_cdt(1) = [];
    s_cdt(end) = [];
end
 
end
