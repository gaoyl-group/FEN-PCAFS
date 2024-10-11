function [id,obj,W,sumW] = auto_ortho_SPCAFS(X,gamma,m,u,p,s)
% Input
% X: num*dim data matrix
% gamma: regularization parameter
% m: projection dimension of W (dim*m)

%Output
%id: sorted features by ||w_i||_2

% Ref: Sparse PCA via L2,p-Norm Regularization for Unsupervised Feature Selection, 2021. 
% Authors: Zhengxin Li, Feiping Nie, Jintang Bian, Danyang Wu, and Xuelong Li.

dim = size(X,2);
num = size(X,1);

% W
[W,obj] = InterationW_ortho_autoSPCAFS(X,gamma,m,u,p,s);
% [W,obj] = InterationW_autoSPCAFS(X,gamma,m,u,p,s);

sqW = (W.^2);
sumW = sum(sqW,2);
[~,id] = sort(sumW,'descend');

% X_2 = X(id(1:selectedfea),:);%here is no need to 
% X2=mapminmax(X_2,0,1);%dn


end



