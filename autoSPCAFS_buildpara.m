function paramCell = autoSPCAFS_buildpara(p_num,lambda_candi,mu_candi,s_candi,m_candi)
n1 = length(p_num);
n2 = length(lambda_candi);
n3 = length(mu_candi);
n4 = length(s_candi);
n5 = length(m_candi);
nP = n1*n2*n3*n4*n5;
paramCell = cell(nP, 1);
idx=0;

for i1=1:n1
    for i2=1:n2
        for i3=1:n3
            for i4 = 1:n4
                for i5 = 1:n5
                    param = [];
                    param.p = p_num(i1);
                    param.lambda = lambda_candi(i2);
                    param.mu = mu_candi(i3);
                    param.s = s_candi(i4);
                    param.m = m_candi(i5);
                    idx = idx + 1;
                    paramCell{idx} = param;
                end
            end
        end
    end
end
end