import torch 
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
from torch.utils.data import DataLoader, Dataset
class WMF(torch.nn.Module):
    def __init__(self, config):
        super(WMF, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_user.weight.data*=0.1
        self.embedding_item.weight.data*=0.1
        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()
    def allpre(self):
        rating=self.logistic(self.embedding_user.weight.mm(self.embedding_item.weight.t()))
        return rating
    def getem(self):
        return self.embedding_user.weight.detach(),self.embedding_item.weight.detach()
    def itempre(self,samitem):
        with torch.no_grad():
            rating=self.logistic(self.embedding_user.weight.mm(self.embedding_item(samitem).t()))
            return rating
    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        logits = torch.sum(element_product,dim=1)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass
    
class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, item_tensor):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.item_tensor = item_tensor

    def __getitem__(self, index):
        return self.item_tensor[index]

    def __len__(self):
        return self.item_tensor.size(0)


class Gen(torch.nn.Module):
    def __init__(self, config,tposuser,tpositem):
        super(Gen, self).__init__()
        self.n = config['num_users']
        self.m=config['num_items']
        self.r1=torch.tensor(tposuser)
        self.r2=torch.tensor(tpositem)
        self.ct=torch.stack((self.r1,self.r2),dim=0)
        self.cf=torch.stack((self.r2,self.r1),dim=0)
        self.w1=torch.nn.Parameter(torch.Tensor(np.random.random(len(self.r1))-0.5))
        self.w2=torch.nn.Parameter(torch.Tensor(np.random.random(len(self.r1))-0.5))
        self.la1=torch.nn.Parameter(torch.Tensor(np.random.random((self.n,config['gen_dim']))-0.5))
        self.la2=torch.nn.Parameter(torch.Tensor(np.random.random((self.n,config['gen_dim']))-0.5))
        self.eps=1e-16
    def gaolfa(self):
        with torch.no_grad():
            expw1=torch.exp(self.w1)
            expw2=torch.exp(self.w2)
            expla1=torch.exp(self.la1)
            expla2=torch.exp(self.la2)
            sumw1=expla1.sum(dim=1)+self.eps
            sumw1.scatter_add_(0,self.r1,expw1)
            lfw1=expw1.div(sumw1[self.r1])
            lfla1=expla1.div(sumw1.reshape(-1,1))
            if(cd==0):
                sumw2=torch.zeros((self.m),device='cpu')+self.eps
            else:
                sumw2=torch.zeros((self.m),device='cuda')+self.eps
            sumw2.scatter_add_(0,self.r2,expw2)
            lfw2=expw2.div(sumw2[self.r2])
            lfla2=expla2/(expla2.sum(dim=0,keepdim=True)+self.eps)
            return lfw1,lfw2,lfla1,lfla2
        
    def forward(self,flp,hnow):
        expw1=torch.exp(self.w1)
        expw2=torch.exp(self.w2)
        expla1=torch.exp(self.la1)
        expla2=torch.exp(self.la2)
        sumw1=expla1.sum(dim=1)+self.eps
        sumw1.scatter_add_(0,self.r1,expw1)
        lfw1=expw1.div(sumw1[self.r1])
        lfla1=expla1.div(sumw1.reshape(-1,1))
        if(cd==0):
            sumw2=torch.zeros((self.m),device='cpu')+self.eps
        else:
            sumw2=torch.zeros((self.m),device='cuda')+self.eps
        sumw2.scatter_add_(0,self.r2,expw2)
        lfw2=expw2.div(sumw2[self.r2])
        lfla2=expla2/(expla2.sum(dim=0,keepdim=True)+self.eps)
        
        
        fa0=torch.zeros((self.n),device='cuda')  
        fa0=(fa0.scatter_add_(0,self.r1,lfw1*lfw2)).reshape(-1,1)
        slfw1=torch.cuda.sparse.FloatTensor(self.ct,lfw1,torch.Size([self.n,self.m]))     
        slfw2=torch.cuda.sparse.FloatTensor(self.cf,lfw2,torch.Size([self.m,self.n]))
        
        for i in range(dep):
            mflp=torch.mean(flp,dim=0,keepdim=True)
            flp=c*(torch.sparse.mm(slfw1,torch.sparse.mm(slfw2,flp))-fa0*flp+fa0.mm(mflp)+ \
            lfla1.mm(lfla2.t().mm(flp)))+(1-c)*hnow;
        flp+=eps
        return lfw1.detach(),lfw2.detach(),lfla1.detach(),lfla2.detach(),flp

    def init_weight(self):
        pass


import argparse

# arguments setting
def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainingdata', type=str, default='trainingdata.txt', help='The path of training data.')
    parser.add_argument('--testdata', type=str, default='testdata.txt', help='The path of test data')
    parser.add_argument('--baselr', type=float, default=0.1, help='learning rate of base RS model')
    parser.add_argument('--basedecay', type=float, default=1, help='decay of base RS model')
    parser.add_argument('--samlr', type=float, default=0.05, help='learning rate of sampling model')
    parser.add_argument('--samdecay', type=float, default=1, help='decay of sampling model')
    parser.add_argument('--topK', type=int, default=5, help='Top-k recommendation')
    parser.add_argument('--iteration', type=int, default=1000, help='number of iterations')
    parser.add_argument('--emb', type=int, default=10, help='length of embeddings')
    parser.add_argument('--numcom', type=int, default=10, help='number of latent community')
    parser.add_argument('--alpha', type=int, default=100, help='alpha')
    parser.add_argument('--beta', type=int, default=20, help='beta')
    parser.add_argument('--c', type=float, default=0.9, help='c')
    parser.add_argument('--samdeep', type=int, default=5, help='deepth of random walk')
    return parser.parse_args()


if __name__ == "__main__": 
    args = parse_args()
    print('Running SamWalker++:')
    # data process
    import cppimport.import_hook
    torch.cuda.set_device(3)
    from scipy.sparse import csr_matrix
    config = {   'num_users': 0,
                  'num_items': 0,
                  'latent_dim': 10,
                  'gen_dim':10,
                 } 

    np.random.seed(0)
    cd=1
    eps=1e-8


    data=pd.read_table(args.trainingdata,header=None)
    test=pd.read_table(args.testdata,header=None)

    n=max(data[:][0].append(test[:][0])) # number of user
    m=max(data[:][1].append(test[:][1])) # number of item
    ch=args.topK

    data=data-1; #change user ID from 1->n to 0->(n-1) 
    test=test-1;
    posuser=np.array(data[:][0])
    positem=np.array(data[:][1])
    config['num_users']=n;
    config['num_items']=m;
    config['latent_dim']=args.emb;
    config['gen_dim']=args.numcom;
    testuser=np.array(test[:][0])
    testitem=np.array(test[:][1])
    hh=np.ones(len(posuser))
    hr=csr_matrix((hh, (posuser, positem)),shape=(n, m))
    fullhr=hr.toarray().astype(np.float64)
    tfullhr=torch.FloatTensor(fullhr)
    flpre=np.tile(np.array((hr.sum(axis=0)/n).reshape(-1)).astype(np.float64),(n,1))
    batchsize=100
    proa=0.1*torch.ones((n,batchsize))
    if(cd==1):
        proa=proa.cuda()


    dataset = UserItemRatingDataset(item_tensor=torch.LongTensor(range(m)))
    dl=DataLoader(dataset, batch_size=batchsize, shuffle=True,drop_last=True)

    import biwalker # C++ source code for conducting sampling
    def sampleall(lfw1,lfw2,lfla1,lfla2,chuan,can):
        ran=np.random.random((n*dep*al))
        zhong=biwalker.negf(lfw1,lfw2,lfla1,lfla2,chuan,can,ran)
        samu=zhong[:,0]
        sami=zhong[:,1]
        rating=np.asarray(hr[samu,sami]).reshape(-1).astype(np.float64)
        return samu,sami,rating


    import ex # C++ source code for fast evaluation
    para=[0.01]
    fil = open("outputofsamwalkerplus.txt", "w") 
    for ii in range(1):
        for jj in range(1):
            torch.set_default_tensor_type('torch.FloatTensor')
            import time
            weight1=args.samdecay
            learning_rate1=args.samlr
            episa=1e-3
            dep=args.samdeep
            al=args.alpha
            bet=args.beta
            c=args.c
            canshu=np.array([n,al,bet,dep,c,m]);
            model = WMF(config)
            opt=torch.optim.Adam(model.parameters(),lr=args.baselr,weight_decay=args.basedecay)
            tposuser=torch.tensor(posuser)
            tpositem=torch.tensor(positem)
            chuan=np.hstack((posuser.reshape(-1,1),positem.reshape(-1,1)))
            if(cd==1):
                tposuser=tposuser.cuda()
                tpositem=tpositem.cuda()
                model.cuda()
            gen=Gen(config,tposuser,tpositem)
            if(cd==1):
                gen.cuda()

            tlfw1,tlfw2,tlfla1,tlfla2=gen.gaolfa()
            bce=torch.nn.BCELoss()
            optgen=torch.optim.Adam(gen.parameters(),lr=learning_rate1,weight_decay=weight1)
            ite=0
            s1= time.clock()
            for ep in range(1000):
                if(ite>=1000):
                    break
                for batch_id, samitem in enumerate(dl):
                    ite=ite+1
                    print('Iterations:',ite,'/1000',end='\r',flush=True)
                    nst=np.array(samitem)
                    lfw1=tlfw1.cpu().numpy()
                    lfw2=tlfw2.cpu().numpy()
                    lfla1=tlfla1.cpu().numpy()
                    lfla2=tlfla2.cpu().numpy()

                    user, item, rating = sampleall(lfw1,lfw2,lfla1,lfla2,chuan,canshu)
                    tuser=torch.LongTensor(user)
                    titem=torch.LongTensor(item)
                    trating=torch.FloatTensor(rating)
                    if(cd==1):
                        tuser=tuser.cuda()
                        titem=titem.cuda()
                        trating=trating.cuda()
                    opt.zero_grad()
                    pre=model(tuser,titem)
                    loss=bce(pre,trating)*len(pre)
                    loss.backward()
                    opt.step()

                    flp=torch.FloatTensor(flpre[:,nst])
                    hrnow=torch.FloatTensor(fullhr[:,nst])
                    if(cd==1):
                        flp=flp.cuda()
                        hrnow=hrnow.cuda()
                        proa=proa.cuda()
                        samitem=samitem.cuda()
                    optgen.zero_grad()
                    [tlfw1,tlfw2,tlfla1,tlfa2,cuid]=gen(flp,hrnow)
                    pregen=model.itempre(samitem)

                    losgen=-torch.sum(cuid*(hrnow*torch.log((pregen+eps)/episa)+(1-hrnow)*torch.log((1-pregen+eps)/(1-episa)))+
                                     cuid*torch.log(proa/(1-proa))- cuid*torch.log(cuid+eps)-(1-cuid)*torch.log(1-cuid+eps))
                    losgen.backward()
                    optgen.step()
        

                    if(ite%200==0):
                        [teu,tev]=model.getem()
                        eu=teu.cpu().numpy()
                        ev=tev.cpu().numpy()
                        prerating=1/(1+np.exp(-eu.dot(ev.T)))
                        cu=np.zeros((n,n));
                        nowfa=np.identity(n);

                        slfw1=csr_matrix((lfw1,(posuser,positem)),shape=[n,m]).toarray()
                        slfw2=csr_matrix((lfw2,(positem,posuser)),shape=[m,n]).toarray()
                        zm=slfw1.dot(slfw2)
                        fa0=np.diag(zm)
                        zm=zm-np.diag(fa0)
                        zm=zm+lfla1.dot(lfla2.T)+fa0.reshape(-1,1)/n
                        for i in range(dep):
                            cu=cu+(nowfa)*(1-c);
                            nowfa=c*nowfa.dot(zm);
                        cu=cu+nowfa.dot(np.ones((n,n))/n);
                        cu=cu.dot(fullhr)

                        prefinal=cu*prerating
                        prefinal[posuser.reshape(-1),positem.reshape(-1)]=-(1<<50)
                        id=np.argsort(prefinal,axis=1,kind='quicksort',order=None)
                        id=id[:,::-1]
                        id1=id[:,:ch]
                       # print(id1) 
                        ans=ex.gaotest(testuser,testitem,id1,id)
                        print('Number of iterations:',ite,'Performance:','Precions=',ans[0],'Recall=',ans[1],'NDCG=',ans[2],'MRR=',ans[4])
                        print(' ')
                     #   print(ep,ite,':',ans,file=fil)
                        if(ite>=1000):
                            break

    fil.close()

