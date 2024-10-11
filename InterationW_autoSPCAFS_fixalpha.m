function [W,WResult] = InterationW_autoSPCAFS_fixalpha(X,gamma,m,u,p)
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

for i = 1:INTER_W
    
    tempStQ = (St+gamma*Q);
    [vec,val] = eig(tempStQ);
    [~,di] = sort(diag(val));
    W = vec(:,di(1:m));

%     w_2psum = sum((sum(W.^2,2)+eps).^(p/2));
   w_2psum= sum((sqrt(sum(W.^2,2)+eps)).^p);
    w_2sum = sum(sum(W.^2,2));
%      a = 1/((w_2psum/ w_2sum).^(1/(1-u))+1);
    a = 0.5;
    tempQ = a^u + 0.5*p * (1-a)^u* (sqrt(sum(W.^2,2)+eps)).^(p-2);
    Q = diag(tempQ);

    w1(i) = trace(W'*St*W); %  Tr(W'*St*W)
    w2(i) = gamma*(a^u * w_2sum+(1-a)^u * w_2psum);% gama*||W||_21
%     w2(i) = gamma*(sum(sqrt(sum(W.^2,2)));% gama*||W||_21
    WResult(i) = w1(i)+w2(i);

    if i > 1 && abs(WResult(i-1)-WResult(i)) < 0.000001
        break;
    end;
    
end;
end