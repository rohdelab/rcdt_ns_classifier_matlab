function rcdt_mat = RCDT(x_I, I, x_c, theta_seq, rm_edge)

eps=1e-8;
pR = radon(I,theta_seq);

for th=1:length(theta_seq)
    p = pR(:,th); p = p/sum(p);
    
    x = linspace(x_I(1),x_I(2),length(p));
    x_cdt=linspace(x_c(1),x_c(2),length(p));
    pCDT(:,1)=CDT(x, p+eps, x_cdt, rm_edge);
       
    rcdt_mat(:,th) = pCDT;
end
              
end


