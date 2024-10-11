function [W,WResult] = InterationW_autoSPCAFS(X,gamma,m,u,p,s)
%X: data matrix(dim*num)
%gamma: regularization parameter
%m: projection dimension of W (dim*m)

num = size(X,2);
dim = size(X,1);

INTER_W = 50;
% INTER_W = 15;
Q = eye(dim);

H = eye(num)-(1/num)*ones(num);
St = X*H*X';
St = -St;

% p=1; % L_2p

WResult=zeros(INTER_W,1);
w1=WResult;
w2=WResult;
bb=zeros(INTER_W,1);
for i = 1:INTER_W
    try
    tempStQ = (St+gamma*Q);
    tempStQ = (tempStQ+tempStQ')/2;
    [vec,val] = eig(tempStQ);
    catch
        ;
    end
    clear tempStQ
    [~,di] = sort(diag(val));
    clear val
    
    W = vec(:,di(1:m));
    clear vec di

    %     w_2psum = sum((sum(W.^2,2)+eps).^(p/2));
    w_2psum = sum((sqrt(sum(W.^2,2)+eps)).^p);%column must be ortho,||row||2==1 if and only if m=d 
    w_2sum = sum(sum(W.^2,2));
    a = 1/((s*w_2psum/ w_2sum).^(1/(1-u))+1);
    bb(i) = a;

    tempQ = a^u + s*0.5*p * (1-a)^u* (sqrt(sum(W.^2,2)+eps)).^(p-2);
    Q = diag(tempQ);

    w1(i) = trace(W'*St*W); %  Tr(W'*St*W)
    w2(i) = gamma*(a^u * w_2sum+s*(1-a)^u * w_2psum);% gama*||W||_21
    %     w2(i) = gamma*(sum(sqrt(sum(W.^2,2)));% gama*||W||_21
    WResult(i) = w1(i)+w2(i);

    if i > 1 && abs(WResult(i-1)-WResult(i)) < 0.000001
        break;
    end;

end;
end