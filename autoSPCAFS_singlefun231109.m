function [res_multidim,res_std,rumtime_average,idx_box]=autoSPCAFS_singlefun231109(X,Y,fea_num,nKm)
% function [rumtime_average,idx_box]=autoSPCAFS_singlefun231109(X,Y)
%X:nd
% nKm=20;
% [~,fean]=size(X);
class_num = length(unique(Y));%求类别，unique数组唯一值
m=class_num-1;%

p_num = [0.1 0.5 1];

lambda_candi = 10.^(-6:1:6);

mu_candi = 0:0.1:1;

s_candi = 0.1:0.1:1;

m_candi = m;

% m_candi = [fean];
% %
% lambda_candi=1e2;
% mu_candi = 0.5;
% s_candi =1;
% p_num = 1;


paramCell = autoSPCAFS_buildpara(p_num,lambda_candi,mu_candi,s_candi,m_candi);
% 
% 
idx_box = cell(length(paramCell), 1);
run_time=zeros(length(paramCell),1);

result_acc=zeros(length(fea_num),length(p_num),length(lambda_candi),length(mu_candi),length(s_candi),length(m_candi));
result_ARI=zeros(length(fea_num),length(p_num),length(lambda_candi),length(mu_candi),length(s_candi),length(m_candi));
result_NMI1=zeros(length(fea_num),length(p_num),length(lambda_candi),length(mu_candi),length(s_candi),length(m_candi));
result_NMI2=zeros(length(fea_num),length(p_num),length(lambda_candi),length(mu_candi),length(s_candi),length(m_candi));
result_purity=zeros(length(fea_num),length(p_num),length(lambda_candi),length(mu_candi),length(s_candi),length(m_candi));
result_prec=zeros(length(fea_num),length(p_num),length(lambda_candi),length(mu_candi),length(s_candi),length(m_candi));
result_recall=zeros(length(fea_num),length(p_num),length(lambda_candi),length(mu_candi),length(s_candi),length(m_candi));
result_RI=zeros(length(fea_num),length(p_num),length(lambda_candi),length(mu_candi),length(s_candi),length(m_candi));
result_MI=zeros(length(fea_num),length(p_num),length(lambda_candi),length(mu_candi),length(s_candi),length(m_candi));
result_HI=zeros(length(fea_num),length(p_num),length(lambda_candi),length(mu_candi),length(s_candi),length(m_candi));
result_f1=zeros(length(fea_num),length(p_num),length(lambda_candi),length(mu_candi),length(s_candi),length(m_candi));


result_accstd=zeros(length(fea_num),length(p_num),length(lambda_candi),length(mu_candi),length(s_candi),length(m_candi));
result_ARIstd=zeros(length(fea_num),length(p_num),length(lambda_candi),length(mu_candi),length(s_candi),length(m_candi));
result_NMI1std=zeros(length(fea_num),length(p_num),length(lambda_candi),length(mu_candi),length(s_candi),length(m_candi));
result_NMI2std=zeros(length(fea_num),length(p_num),length(lambda_candi),length(mu_candi),length(s_candi),length(m_candi));
result_puritystd=zeros(length(fea_num),length(p_num),length(lambda_candi),length(mu_candi),length(s_candi),length(m_candi));
result_precstd=zeros(length(fea_num),length(p_num),length(lambda_candi),length(mu_candi),length(s_candi),length(m_candi));
result_recallstd=zeros(length(fea_num),length(p_num),length(lambda_candi),length(mu_candi),length(s_candi),length(m_candi));
result_RIstd=zeros(length(fea_num),length(p_num),length(lambda_candi),length(mu_candi),length(s_candi),length(m_candi));
result_MIstd=zeros(length(fea_num),length(p_num),length(lambda_candi),length(mu_candi),length(s_candi),length(m_candi));
result_HIstd=zeros(length(fea_num),length(p_num),length(lambda_candi),length(mu_candi),length(s_candi),length(m_candi));
result_f1std=zeros(length(fea_num),length(p_num),length(lambda_candi),length(mu_candi),length(s_candi),length(m_candi));

parfor i1 = 1:length(paramCell)
    fprintf(['autoSPCAFS parameter search %d out of %d...\n'], i1, length(paramCell));
    t0=cputime;%gamma,m,u,p,s
    [id,~,~,~] = autoSPCAFS(X',paramCell{i1}.lambda,paramCell{i1}.m,paramCell{i1}.mu,paramCell{i1}.p,paramCell{i1}.s);
    idx_box{i1,1} = id;
    %         reduceddata{i1,1}=X2;%dn
    %         result1{i1,1} = compute_Clusteringfs(X2', Y, nKm);
    run_time(i1,1)=cputime-t0;
end

rumtime_average = mean(run_time);
clear run_time
result1 = cell(length(fea_num),length(paramCell));
% km_label = cell(length(fea_num),length(paramCell));
for q = 1:length(fea_num)
    f = fea_num(q);
    parfor u = 1:length(paramCell)
        fprintf(['autoSPCAFS cluster evaluation  u%d fea_num%d ...\n'],u,q);
        idx10 = idx_box{u,1};
        re1= eval_fs(X,Y,idx10(1:f),nKm);
        result1{q,u} = re1;
        %         km_label{q,u} = label;
    end
    resultA = result1(q,:)';
    B = reshape(resultA,length(m_candi),length(s_candi),length(mu_candi),length(lambda_candi),length(p_num));
    result2 = permute(B,[5 4 3 2 1]);%d k jsigam
    clear resultA
    clear B

    for  d=1:length(p_num)
        for k=1:length(lambda_candi)
            for j=1:length(mu_candi)
                for r = 1:length(s_candi)
                    for zz =1:length(m_candi)
                        result = result2{d,k,j,r,zz};
                        result_acc(q,d,k,j,r,zz)= result.mean_acc;
                        result_ARI(q,d,k,j,r,zz)= result.mean_ARI ;
                        result_NMI1(q,d,k,j,r,zz)= result.mean_nmi_max ;
                        result_NMI2(q,d,k,j,r,zz)=result.mean_nmi_sqrt;
                        result_purity(q,d,k,j,r,zz)= result.mean_purity;
                        result_prec(q,d,k,j,r,zz)= result.mean_prec ;
                        result_recall(q,d,k,j,r,zz)= result.mean_recall ;
                        result_RI(q,d,k,j,r,zz)=result.mean_RI;
                        result_MI(q,d,k,j,r,zz)= result.mean_MI;
                        result_HI(q,d,k,j,r,zz)= result.mean_HI ;
                        result_f1(q,d,k,j,r,zz)= result.mean_f1 ;

                        result_accstd(q,d,k,j,r,zz)= result.std_acc;
                        result_ARIstd(q,d,k,j,r,zz)= result.std_ARI ;
                        result_NMI1std(q,d,k,j,r,zz)= result.std_nmi_max ;
                        result_NMI2std(q,d,k,j,r,zz)=result.std_nmi_sqrt;
                        result_puritystd(q,d,k,j,r,zz)= result.std_purity;
                        result_precstd(q,d,k,j,r,zz)= result.std_prec ;
                        result_recallstd(q,d,k,j,r,zz)= result.std_recall ;
                        result_RIstd(q,d,k,j,r,zz)=result.std_RI;
                        result_MIstd(q,d,k,j,r,zz)= result.std_MI;
                        result_HIstd(q,d,k,j,r,zz)= result.std_HI ;
                        result_f1std(q,d,k,j,r,zz)= result.std_f1 ;
                    end
                end
            end
        end
    end
end



res_multidim = struct('result_acc', result_acc,'result_ARI',result_ARI,'result_NMI1',result_NMI1,'result_NMI2',result_NMI2, ...
    'result_purity',result_purity,'result_prec',result_prec, 'result_recall',result_recall,'result_RI',result_RI,'result_MI',result_MI, ...
    'result_HI',result_HI,'result_f1',result_f1);

res_std = struct('result_accstd', result_accstd,'result_ARIstd',result_ARIstd,'result_NMI1std',result_NMI1std,'result_NMI2std',result_NMI2std, ...
    'result_puritystd',result_puritystd, 'result_precstd',result_precstd,'result_recallstd',result_recallstd,'result_RIstd',result_RIstd, ...
    'result_MIstd',result_MIstd,'result_HIstd',result_HIstd,'result_f1std',result_f1std);

% [maxacc,maxaccidx] = max(result_acc(:));
%
% sizeacc = size(result_acc);
%
% [q1, d1, k1, j1, r1] = ind2sub(sizeacc, maxaccidx);
%
% sizeacccut = sizeacc(1,2:end);
% maxaccpos = [q1, d1, k1, j1, r1];
%
% if numel(sizeacccut) == 1
%     u1 = 1;
% else
%     u1 = sub2ind(sizeacccut,d1, k1, j1, r1);
% end
%
%  max_acc_featurelabel = idx_box{q1, u1};
%  max_acc_kmlabel = km_label{q1, u1};



end

