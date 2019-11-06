function[finalresult]=samwalker(chdata,chtest,chtrust)
eps=1e-16;
randn('seed',1.0);
rand('seed',1.0);
data=load(chdata); %training data
test=load(chtest); %test data
trust=load(chtrust); %trustnetwork
max1=max([data;test]);
max2=max(trust);
n=max(max1(:,1),max2(:,1));
n=max(n,max2(:,2));
m=max1(:,2);


lambda=1; % decay of u,v
alpha=0.1; % step-size of updating u,v
lambda1=0.3; % decay of fa
alpha1=0.05; % step-size of updating phi
c=0.9;
dep=5; % max depth
al=80; %alpha
bet=20; %beta
episa=1e-3; %epsilon
k=10; % dimension of u,v
ch=5; % test recommendation performance in terms of pre@ch, rec@ch


userid_train=data(:,1); 
itemid_train=data(:,2);
fb_train=data(:,3);
t1=trust(:,1); %trusterid
t2=trust(:,2); %trusteeid

nn=length(userid_train); 
HR=sparse(userid_train,itemid_train,ones(nn,1),n,m); %feedback matrix
HT=sparse(trust(:,1),trust(:,2),ones(length(trust),1),n,n); % social network matrix
HT=HT|HT';
[trsuti,trustj]=find(HT);
trust=[trsuti,trustj];
t1=trust(:,1);
t2=trust(:,2);
zhuant1=sparse(t1,1:length(t1),ones(length(t1),1),n,length(t1));
zhuant2=sparse(t2,1:length(t2),ones(length(t2),1),n,length(t2));
% transform social relations into bi-directional relations

nep=1000; %number of iterations
idcg=zeros(n,1);

     tt=length(test(:,1));
    H=sparse(test(:,1),test(:,2),ones(tt,1),n,m);
 
    num=sum(H,2);
    
   
    for i=1:n 
        [pp,an]=sort(H(i,:),2,'descend');
        for j=1:num(i)
            idcg(i)=idcg(i)+1/log2(j+1);
        end
    end
  

  tie=(rand(length(trust),1)-0.5); %tie strength (w_ik)
  tie0=((rand(n,1)-0.5)); % w_i0
  tiesim=exp(tie);
  stresum=myv2s(t1,tiesim,n)+exp(tie0);
  stre=tiesim./stresum(t1);
  fa=sparse(t1,t2,stre,n,n); %sparse matrix of phi
  fa0=exp(tie0)./stresum; 

 u=randn(n,k)*0.1;
   v=randn(m,k)*0.1;
    negconst=5;
   


    cu=0;
    mid=100; %N_si
    rt=randperm(m);
    rtbun=mod(mid-mod(m,mid),mid);
    rtbu=randi(m,1,rtbun);
    rt=[rt,rtbu];
    tolm=length(rt)/mid;


   
    mnum=sum(HR,1);
    proa=repmat(0.1,n,m);

    opadam=cell(2,4);
    nowga=cell(1,4);
    for i=1:2
        opadam{i,1}=zeros(size(u));
        opadam{i,2}=zeros(size(v));
         opadam{i,3}=zeros(size(tie));
         opadam{i,4}=zeros(size(tie0));
     end
    fl=zeros(n,mid,dep+1);
    flpre=ones(n,n)/n*HR;
    mfl=zeros(dep+1,mid);
    dfl=zeros(n,mid,dep+1);  
tic
for ep=1:1000

    fprintf('iteration %d/1000\n',ep);

  %% update u,v
    ran=rand(n*dep*al,1);
    % tic
    [Ax,Ay,Az]=find(fa);
    A=[Ax,Ay,Az];
    samq=mysamwalknew(A,[userid_train,itemid_train],n,al,bet,dep,c,ran,fa0); % sample informative training instrances
    % toc
    x=samq(:,1);
    y=samq(:,2);
    nsam=length(x);
    z=HR(sub2ind(size(HR),x,y));
    samgu=sparse(x,1:nsam,ones(nsam,1),n,nsam);
    samgv=sparse(y,1:nsam,ones(nsam,1),m,nsam);
    nz=2*z-1;
    zhongr=sum(u(x,:).*v(y,:),2);
    zhong=ga(-nz.*zhongr);
    du=samgu*bsxfun(@times,zhong.*nz,v(y,:));
    dv=samgv*bsxfun(@times,zhong.*nz,u(x,:));
    
    du=du-lambda*u;
    dv=dv-lambda*v;

    prga={du,dv};
    % adam
    p=length(prga);
    beta1=0.9;
    beta2=0.999;
    eps=1e-8;

    for i=1:p
      opadam{1,i}=opadam{1,i}*beta1+(1-beta1)*prga{1,i};
      opadam{2,i}=opadam{2,i}*beta2+(1-beta2)*(prga{1,i}.^2);
      nowm=opadam{1,i}/(1-beta1.^ep);
      nowv=opadam{2,i}/(1-beta2.^ep);
      nowga{i}=nowm./(sqrt(nowv+eps));
    end

    u=u+alpha*nowga{1};
    v=v+alpha*nowga{2};
    


  %% update fa

  if(mod(ep-1,tolm)==0)
    rt=randperm(m);
    rtbun=mod(mid-mod(m,mid),mid);
    rtbu=randi(m,1,rtbun);
    rt=[rt,rtbu];
  end

  idzu=mod(ep-1,tolm);
  id=rt(idzu*mid+1:(idzu+1)*mid); %% sample items to update fa

 

  HRid=HR(:,id);
  D=ga(u*v(id,:)');
  fl(:,:,1)=flpre(:,id);

  mfl(1,:)=mean(fl(:,:,1),1);
  for i=1:dep
    fl(:,:,i+1)=c*(fa*fl(:,:,i)+fa0*mfl(i,:))+(1-c)*HRid;
    mfl(i+1,:)=mean(fl(:,:,i+1),1);
  end


  cuid=fl(:,:,dep+1)+eps;
  

  dfl(:,1:mid,dep+1)=HRid.*log((D+eps)/episa)+(1-HRid).*log((1-D+eps)/(1-episa))+(log(proa(:,id)./cuid)-log((1-proa(:,id))./(1-cuid)));
  for i=dep:-1:1
    dfl(:,:,i)=c*(fa'*dfl(:,:,i+1)+fa0'/n*dfl(:,:,i+1));
  end
  dtrust=zeros(length(trust),1);
  dfa0=zeros(n,1);
  for i=1:dep
    dtrust=dtrust+c*sum(fl(t2,:,i).*dfl(t1,:,i+1),2);
    dfa0=dfa0+c*dfl(:,:,i+1)*mfl(i,:)';
  end
  


  zhongtrust=dtrust.*stre;
  zhongfa0=dfa0.*fa0;
  sdtrust=myv2s(t1,zhongtrust,n)+zhongfa0;

  datr=stre.*(dtrust-sdtrust(t1));

  dafa0=fa0.*(dfa0-sdtrust);
  datr=datr-lambda1*tie;
  dafa0=dafa0-lambda1*tie0;

   prga={datr,dafa0};
    % adam
    p=length(prga);

    for i=3:2+p
      opadam{1,i}=opadam{1,i}*beta1+(1-beta1)*prga{1,i-2};
      opadam{2,i}=opadam{2,i}*beta2+(1-beta2)*(prga{1,i-2}.^2);
      nowm=opadam{1,i}/(1-beta1.^ep);
      nowv=opadam{2,i}/(1-beta2.^ep);
      nowga{i}=nowm./(sqrt(nowv+eps));
    end

    tie=tie+alpha1*nowga{3};
    tie0=tie0+alpha1*nowga{4};


    tiesim=exp(tie);
    stresum=myv2s(t1,tiesim,n)+exp(tie0);
    stre=tiesim./stresum(t1);
    fa=sparse(t1,t2,stre,n,n);
    fa0=exp(tie0)./stresum;

        
    
     if(ep>=1&&mod(ep,200)==0)
      % test Precision, Recall, Ndcg and MRR
      % make predictions based on the probailbity that the item will be consumed by the user
     cu=zeros(n,n);
    nowfa=eye(n);
    for i=1:dep
      cu=cu+(nowfa)*(1-c);
      nowfa=nowfa*c*(fa)+repmat(nowfa*c*fa0/n,1,n);
    end
    cu=cu+nowfa*ones(n,n)/n;
    cu=cu*HR;
     R=u*v';
   R=cu.*ga(R);
    R(sub2ind(size(R),userid_train,itemid_train))=-inf;
   D=R;
    [~,an]=sort(D,2,'descend');


   nt=length(test(:,1));
     num=sum(H,2);
    precision=zeros(n,1);
    recall=zeros(n,1);
    np=zeros(n,1);
     ndcg=zeros(n,1);
     ndc=zeros(n,1);
     mrr=zeros(n,1);
    cao=full(sum(num>0));  
    pq=zeros(1,m);
    for i=1:n
        id1=find(H(i,:)~=0);
        id2=an(i,1:ch);
        pr=intersect(id1,id2);
        if(num(i)~=0)
           precision(i)= length(pr)/ch;
           recall(i)=length(pr)/num(i);
        np(i)=length(pr)/min(num(i),ch);
        pq(an(i,1:m))=1:m;
        pg=pq(id1);
        ndcg(i)=(sum(H(i,id1)./(log2(pg+1))))/idcg(i);   
        mrr(i)=sum(H(i,id1)./pg);
        end  
    end
     pre=sum(precision)/cao;
     reca=sum(recall)/cao;
     nd=sum(ndcg)/cao;
     mr=sum(mrr)/cao;
       fprintf('iteration %d: precision@5=%f recall@5=%f NDCG=%f MRR=%f\n',ep,pre,reca,nd,mr); 
      finalresult=[pre,reca,nd,mr];
  end
end
end
