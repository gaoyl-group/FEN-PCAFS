function [W,WResult] = InterationW_ortho_autoSPCAFS(X,gamma,m,u,p,s)
%X: data matrix(num*dim)
%gamma: regularization parameter
%m: projection dimension of W (dim*m)

num = size(X,1);
dim = size(X,2);
Id = eye(dim);
In = eye(num);
X_T = X';

INTER_W = 100;
[~,~,V] = svd(X);
W = V(:,1:m);

X = X - repmat(mean(X,1),num,1);
St= X'*X;


WResult=zeros(INTER_W,1);
% a_result=zeros(INTER_W,1);
% ww = zeros(INTER_W,7);
% id_box = zeros(INTER_W,dim);
% E = eye(dim);
% iE = E;
for i = 1:INTER_W
    %% update Q
    A = St*W;
    [UQ,~,VQ] = svds(A,m);
    Q = UQ*VQ';%dm
    %% update alpha before W can speed up convergency
    w_2psum = sum((sqrt(sum(W.^2,2)+eps)).^p);%column must be ortho,||row||2==1 if and only if m=d
    w_2sum = sum(sum(W.^2,2));
    alpha = 1/((s*w_2psum/ w_2sum).^(1/(1-u))+1);
%     a_result(i) = alpha;
    %% E
    tempE = alpha^u + s*0.5*p * (1-alpha)^u* (sqrt(sum(W.^2,2)+eps)).^(p-2);
    E = spdiags(tempE,0,dim,dim);
    iE = spdiags(1./(gamma*tempE + 0.01),0,dim,dim);

    %% update W 左除右除
    if num > dim
%         tic
        W = (St+gamma*E+0.001*Id)\(St*Q);
%         t1 = toc;
%         t2=0;
    elseif num <= dim
%         tic
%         iB = iE - iE*X_T/(In + X*iE*X_T)*X*iE;
%         W = iB * St * Q;
%         t1 = toc;

%         tic
        W = iE*X_T/(In + X*iE*X_T)*X*Q;
%         t2 = toc;%time is less
%         W2 = iB * St; is correct
    end
  
%% obj
    w1 = norm(X-X*W*Q','fro')^2; %  Tr(W'*St*W)
    w2 = alpha^u * w_2sum+s*(1-alpha)^u * w_2psum;% gamma*||W||_21

    w3 = sum(sum(W.^2,2).^(p/2));
    w4 = sum(sum(W.^2,2));
    ww(i,:) = [w2,w3,w4,alpha,alpha^u,(1-alpha),(1-alpha)^u];

    sqW = (W.^2);
    sumW = sum(sqW,2);
%     [~,id] = sort(sumW,'descend');
    id_box(i,:) = sumW;

    WResult(i) = w1+gamma*w2;

    if i > 1 && abs(WResult(i-1)-WResult(i)) < 10^-5
        break;
    end;

end;
ww_table = array2table(ww,...
    'VariableNames',{'sumof(L2+Lp)','Lp','L2','alpha','alpha^u','1-alpha','(1-alpha)^u'});
end