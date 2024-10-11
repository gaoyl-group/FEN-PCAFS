addpath(genpath('E:\Users\xzh\data_test'))
clc;clear
load 00150Iris.mat

% fea_num = 1:size(X,2);
[data_num,~] = size(X);
choose_norm = 1 ;% Normalization methods, 0: no normalization, 1: z-score, 2: max-min 3 centralize 4 normaized 5.fs
% init=4; % Initialization methods, 1: random, 2: K-means, 3: fuzzt c-means, 4: K-means clustering, accelerated by matlab matrix operations.
% repeat_num=10; % Repeat the experiment repeat_num times
% [X1,normalstr] = normlization(X, choose_norm);%nonormlization buxing
class = length(unique(Y));

% fea = X1';
% H = eye(data_num)-(1/data_num)*ones(data_num);
% St = fea*H*fea';
gamma = 1e2;
m = class-1;
mu = 0.01;
p = 0.5;
s = 1;
[id,obj,W,sumW] = auto_ortho_SPCAFS(X,gamma,m,mu,p,s);
% [id,obj,W,sumW] = autoSPCAFS(X',St,lambda,m,mu,p,s);

[a,b] = eig(St1);
[c,d] = sort(diag(b));

choose_norm = 0 ;% Normalization methods, 0: no normalization, 1: z-score, 2: max-min 3 centralize 4 normaized 5.fs
% init=4; % Initialization methods, 1: random, 2: K-means, 3: fuzzt c-means, 4: K-means clustering, accelerated by matlab matrix operations.
% repeat_num=10; % Repeat the experiment repeat_num times
[X0,normalstr] = normlization(X, choose_norm);%nonormlization buxing


fea = X0';
H = eye(data_num)-(1/data_num)*ones(data_num);
St0 = fea*H*fea';

choose_norm = 2 ;% Normalization methods, 0: no normalization, 1: z-score, 2: max-min 3 centralize 4 normaized 5.fs
% init=4; % Initialization methods, 1: random, 2: K-means, 3: fuzzt c-means, 4: K-means clustering, accelerated by matlab matrix operations.
% repeat_num=10; % Repeat the experiment repeat_num times
[X2,normalstr] = normlization(X, choose_norm);%nonormlization buxing

fea = X2';
H = eye(data_num)-(1/data_num)*ones(data_num);
St2 = fea*H*fea';

% [res_multidim,res_std,rumtime_average,idx_box]=autoSPCAFS_singlefun231109(X,Y,fea_num);
