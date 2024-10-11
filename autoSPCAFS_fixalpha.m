function [X2,id,obj,W] = autoSPCAFS_fixalpha(X,selectedfea,gamma,m,u,p)
% Input
% X: dim*num data matrix
% gamma: regularization parameter
% m: projection dimension of W (dim*m)

%Output
%id: sorted features by ||w_i||_2

% Ref: Sparse PCA via L2,p-Norm Regularization for Unsupervised Feature Selection, 2021. 
% Authors: Zhengxin Li, Feiping Nie, Jintang Bian, Danyang Wu, and Xuelong Li.

num = size(X,2);
dim = size(X,1);

% W
[W,obj] = InterationW_autoSPCAFS_fixalpha(X,gamma,m,u,p);

sqW = (W.^2);
sumW = sum(sqW,2);
[~,id] = sort(sumW,'descend');

X_2 = X(id(1:selectedfea),:);
X2=mapminmax(X_2,0,1);%dn


end




