function [result2_max,result2_pic,rumtime_average,bestreducedata,result2,idx_box]=autoSPCAFS_singlefun_fixalpha(X,Y,fea_num,p)
%X:nd
 nKm=20;


lambda_candi = [1e-6 1e-4 1e-2 1 1e2 1e4 1e6];
%  lambda_candi=5;
mu_candi = [5 10 15 20 25 30 35 40];
 paramCell = autoSPCAFS_buildpara(fea_num,lambda_candi,mu_candi);
class_num = length(unique(Y));%求类别，unique数组唯一值
  m=class_num-1;%
 reduceddata=cell(length(paramCell), 1);
 idx_box = cell(length(paramCell), 1);
 result1=cell(length(paramCell), 1);
 run_time=zeros(length(paramCell),1);
 
 result_acc=zeros(length(fea_num),length(lambda_candi),length(mu_candi));
 result_ARI=zeros(length(fea_num),length(lambda_candi),length(mu_candi));
 result_NMI1=zeros(length(fea_num),length(lambda_candi),length(mu_candi));
 result_NMI2=zeros(length(fea_num),length(lambda_candi),length(mu_candi));
  result_purity=zeros(length(fea_num),length(lambda_candi),length(mu_candi));
 result_prec=zeros(length(fea_num),length(lambda_candi),length(mu_candi));
 result_recall=zeros(length(fea_num),length(lambda_candi),length(mu_candi));
 result_RI=zeros(length(fea_num),length(lambda_candi),length(mu_candi));
  result_MI=zeros(length(fea_num),length(lambda_candi),length(mu_candi));
 result_HI=zeros(length(fea_num),length(lambda_candi),length(mu_candi));
 result_f1=zeros(length(fea_num),length(lambda_candi),length(mu_candi));

 
 result_accstd=zeros(length(fea_num),length(lambda_candi),length(mu_candi));
 result_ARIstd=zeros(length(fea_num),length(lambda_candi),length(mu_candi));
 result_NMI1std=zeros(length(fea_num),length(lambda_candi),length(mu_candi));
 result_NMI2std=zeros(length(fea_num),length(lambda_candi),length(mu_candi));
  result_puritystd=zeros(length(fea_num),length(lambda_candi),length(mu_candi));
 result_precstd=zeros(length(fea_num),length(lambda_candi),length(mu_candi));
 result_recallstd=zeros(length(fea_num),length(lambda_candi),length(mu_candi));
 result_RIstd=zeros(length(fea_num),length(lambda_candi),length(mu_candi));
  result_MIstd=zeros(length(fea_num),length(lambda_candi),length(mu_candi));
 result_HIstd=zeros(length(fea_num),length(lambda_candi),length(mu_candi));
 result_f1std=zeros(length(fea_num),length(lambda_candi),length(mu_candi));
 
 for i1 = 1:length(paramCell)
        fprintf(['autoSPCAFS_fixalpha parameter search %d out of %d...\n'], i1, length(paramCell));
        t0=cputime;
        [X2,id,obj,W] = autoSPCAFS_fixalpha(X',paramCell{i1}.feanum,paramCell{i1}.lambda,m,paramCell{i1}.mu,p);
        idx_box{i1,1} = id;
        reduceddata{i1,1}=X2;%dn
        result1{i1,1} = compute_Clusteringfs(X2', Y, nKm);
        run_time(i1,1)=cputime-t0;
 end
    B = reshape(result1,length(mu_candi),length(lambda_candi),length(fea_num));
    result2 = permute(B,[3 2 1]);%d k jsigam
    clear result1
    clear B
    
    C = reshape(reduceddata,length(mu_candi),length(lambda_candi),length(fea_num));
    reduceddata1 = permute(C,[3 2 1]);%d k jsigam
    clear reduceddata
    clear C
    
    rumtime_average = mean(run_time);
    clear run_time
         for  d=1:length(fea_num)
                for j=1:length(mu_candi)
                     for k=1:length(lambda_candi)
                     fprintf(['autoSPCAFS parameter evaluation %d ..\n'], d);
                           result= cell2mat(result2(d,k,j));
                           result_acc(d,k,j)= result.mean_acc; 
                           result_ARI(d,k,j)= result.mean_ARI ;
                           result_NMI1(d,k,j)= result.mean_nmi_max ;
                           result_NMI2(d,k,j)=result.mean_nmi_sqrt;
                           result_purity(d,k,j)= result.mean_purity; 
                           result_prec(d,k,j)= result.mean_prec ;
                           result_recall(d,k,j)= result.mean_recall ;
                           result_RI(d,k,j)=result.mean_RI;
                           result_MI(d,k,j)= result.mean_MI; 
                           result_HI(d,k,j)= result.mean_HI ;
                           result_f1(d,k,j)= result.mean_f1 ;
                         
                           
                           result_accstd(d,k,j)= result.std_acc; 
                           result_ARIstd(d,k,j)= result.std_ARI ;
                           result_NMI1std(d,k,j)= result.std_nmi_max ;
                           result_NMI2std(d,k,j)=result.std_nmi_sqrt;
                           result_puritystd(d,k,j)= result.std_purity; 
                           result_precstd(d,k,j)= result.std_prec ;
                           result_recallstd(d,k,j)= result.std_recall ;
                           result_RIstd(d,k,j)=result.std_RI;
                           result_MIstd(d,k,j)= result.std_MI; 
                           result_HIstd(d,k,j)= result.std_HI ;
                           result_f1std(d,k,j)= result.std_f1 ;
                end
          end
     end
     clear result 

     res_multidim = struct('result_acc', result_acc,'result_ARI',result_ARI,'result_NMI1',result_NMI1,'result_NMI2',result_NMI2, ...
   'result_purity',result_purity,'result_prec',result_prec, 'result_recall',result_recall,'result_RI',result_RI,'result_MI',result_MI, ...
   'result_HI',result_HI,'result_f1',result_f1);

res_std = struct('result_accstd', result_accstd,'result_ARIstd',result_ARIstd,'result_NMI1std',result_NMI1std,'result_NMI2std',result_NMI2std, ...
   'result_puritystd',result_puritystd, 'result_precstd',result_precstd,'result_recallstd',result_recallstd,'result_RIstd',result_RIstd, ...
   'result_MIstd',result_MIstd,'result_HIstd',result_HIstd,'result_f1std',result_f1std);

 [result2_max,result2_pic,bestreducedata] = evalmax(fea_num,res_multidim,res_std,reduceddata1);

 
%     result2_matrix={result_acc,result_ARI,result_NMI1,result_NMI2};
%     [accmax,pos1]=max(result_acc(:));
%     [acc_dim,acck,accsigma]=ind2sub(size(result_acc),pos1);
%     maxaccstd=result_accstd(acc_dim,acck,accsigma);
%     bestreducedata = cell2mat(reduceddata1(acc_dim,acck,accsigma));
%     clear  result_accstd reduceddata1
% 
%     [ARImax,pos2]=max(result_ARI(:));
%     [ARI_dim,ARI_k,ARI_sigma]=ind2sub(size(result_ARI),pos2);
%     maxARIstd=result_ARIstd(ARI_dim,ARI_k,ARI_sigma);
%     clear  result_ARIstd 
%     
%     [NMI1max,pos3]=max(result_NMI1(:));
%     [NMI1_dim,NMI1_k,NMI1_sigma]=ind2sub(size(result_NMI1),pos3);
%     maxNMI1std=result_NMI1std(NMI1_dim,NMI1_k,NMI1_sigma);
%     clear  result_NMI1std 
%     
%     [NMI2max,pos4]=max(result_NMI2(:));
%     [NMI2_dim,NMI2_k,NMI2_sigma]=ind2sub(size(result_NMI2),pos4);
%     maxNMI2std=result_NMI2std(NMI2_dim,NMI2_k,NMI2_sigma);
%      clear  result_NMI2std 
% 
%     [puritymax,pos5]=max(result_purity(:));
%     [purity_dim,purity_k,purity_sigma]=ind2sub(size(result_purity),pos5);
%     maxpuritystd=result_puritystd(purity_dim,purity_k,purity_sigma);
%     clear result_purity result_puritystd
%     
%     [precmax,pos6]=max(result_prec(:));
%     [prec_dim,prec_k,prec_sigma]=ind2sub(size(result_prec),pos6);
%     maxprecstd=result_precstd(prec_dim,prec_k,prec_sigma);
%     clear result_purity result_precstd
%     
%     [recallmax,pos7]=max(result_recall(:));
%     [recall_dim,recall_k,recall_sigma]=ind2sub(size(result_recall),pos7);
%     maxrecallstd=result_recallstd(recall_dim,recall_k,recall_sigma);
%     clear result_recall result_recallstd
%     
%     [RImax,pos8]=max(result_RI(:));
%     [RI_dim,RI_k,RI_sigma]=ind2sub(size(result_RI),pos8);
%     maxRIstd=result_RIstd(RI_dim,RI_k,RI_sigma);
%     clear result_RI result_RIstd
%     
%     [MImax,pos9]=max(result_MI(:));
%     [MI_dim,MI_k,MI_sigma]=ind2sub(size(result_MI),pos9);
%     maxMIstd=result_MIstd(MI_dim,MI_k,MI_sigma);
%     clear result_MI result_MIstd
%     
%     [HImax,pos10]=max(result_HI(:));
%     [HI_dim,HI_k,HI_sigma]=ind2sub(size(result_HI),pos10);
%     maxHIstd=result_HIstd(HI_dim,HI_k,HI_sigma);
%     clear result_HI result_HIstd
%     
%     [f1max,pos11]=max(result_f1(:));
%     [f1_dim,f1_k,f1_sigma]=ind2sub(size(result_f1),pos11);
%     maxf1std=result_f1std(f1_dim,f1_k,f1_sigma);
%     clear result_f1 result_f1std
%     
%     
%     result2_max=[accmax maxaccstd acc_dim;
%                 ARImax maxARIstd ARI_dim; 
%                 NMI1max maxNMI1std NMI1_dim;
%                 NMI2max  maxNMI2std NMI2_dim;
%                 puritymax maxpuritystd purity_dim;
%                 precmax  maxprecstd prec_dim;
%                 recallmax maxrecallstd recall_dim;
%                 RImax  maxRIstd RI_dim;
%                 MImax maxMIstd MI_dim;
%                 HImax maxHIstd HI_dim;
%                 f1max maxf1std f1_dim;
%                 ];
%     
%      D2_acc = squeeze(max(result_acc,[],2));%按行（即维度）dkj,消去k,j页中，每d行取max值(消k)
%      acc2_max=max(D2_acc,[],2);%d*j 按行取最大值
% 
%      D2_ari = squeeze(max(result_ARI,[],2));%按行（即维度）
%      ari2_max=max(D2_ari,[],2);
% 
%      D2_nmim = squeeze(max(result_NMI1,[],2));%按行（即维度）
%      nmim2_max=max(D2_nmim,[],2);
% 
%      D2_nmis = squeeze(max(result_NMI2,[],2));%按行（即维度）
%      nmis2_max=max(D2_nmis,[],2);
%      clear result_acc result_ARI result_NMI1 result_NMI2
%      
% %      D2_purity = squeeze(max(result_purity,[],2));%按行（即维度）
% %      purity_max=max(D2_purity,[],2);
% %      
% %      D2_prec = squeeze(max(result_prec,[],2));%按行（即维度）
% %      nmis2_max=max(D2_prec,[],2);
% %      
% %      D2_recall = squeeze(max(result_recall,[],2));%按行（即维度）
% %      nmis2_max=max(D2_recall,[],2);
%      
%      result2_pic=[acc2_max,ari2_max,nmim2_max,nmis2_max];
%  
   
%  options = [];
% %  options.Metric = 'Euclidean';
%  options.NeighborMode = 'KNN';
%  options.WeightMode = 'HeatKernel';
% 
%  K=12;

%  sigmalist=[2 200 20000 2000000];

%   run_time=zeros(reducedim,K-1,length(sigmalist));
% %  result_matrix=cell(1,4);
%  for d=1:reducedim
%         new_lower_dimension=d;
% %        options.new_lower_dimension=d;
%      for k=2:K
%          options.k=k;
%         for j=1:length(sigmalist)
%                 options.t=sigmalist(j);
%                 sigma = sigmalist(j);
%                 
% %                  W = constructW(X,options);%n d
% %                  [ eigvec,eigval] = LPP_by_lqd(W,X',new_lower_dimension);%d n
% %                   mappedX1 = X * eigvec; 
% %                   
% %                 [mappedX2, mapping] = lpp(X, new_lower_dimension, k, sigma );
% %                 
% %                  [eigvector, eigvalue] = LPP_caideng(W, options, X);%内部已预处理
% %                   mappedX3 = X * eigvector; 
%                   
%                   disp(['LPP and reduced dimension is' num2str(d)]);
%                  t0=cputime;
% %                   [mappedX, mapping] = lpp_xzh(X, new_lower_dimension, k, sigma);%X:nd
%            run_time(d,k-1,j)=cputime-t0;
%            result = compute_Clustering(mappedX, Y, nKm);
%            result_acc(d,k-1,j)= result.mean_acc; 
%            result_ARI(d,k-1,j)= result.mean_ARI ;
%            result_NMI1(d,k-1,j)= result.mean_nmi_max ;
%            result_NMI2(d,k-1,j)=result.mean_nmi_sqrt;
%       end
%     end
%  end
%     result_matrix={result_acc,result_ARI,result_NMI1,result_NMI2};
%     [accmax,pos1]=max(result_acc(:));
%     [acc_dim,~,~]=ind2sub(size(result_acc),pos1);
%     [ARImax,pos2]=max(result_ARI(:));
%     [ARI_dim,~,~]=ind2sub(size(result_ARI),pos2);
%     [NMI1max,pos3]=max(result_NMI1(:));
%     [NMI1_dim,~,~]=ind2sub(size(result_NMI1),pos3);
%     [NMI2max,pos4]=max(result_NMI2(:));
%     [NMI2_dim,~,~]=ind2sub(size(result_NMI2),pos4);
%     
%     result1=[accmax acc_dim;ARImax ARI_dim;NMI1max NMI1_dim;NMI2max NMI2_dim];
end

