function [s_cdt]=CDT(x,J,x_cdt,rm_edge)

if (size(J,2) == 1) 
    J=J';
end

J=J/sum(J); 
cJ=cumsum(J);

if rm_edge
    eps = 1e-2;
    i1=find(cJ<eps); i2=find(abs(1-cJ)<eps);
    i1(end)=[]; i2(1)=[];
    i=[i1(:);i2(:)];
    cJ(i)=[]; x(i)=[];
end

s_cdt=interp1(cJ,x,x_cdt,'pchip');
 
end
