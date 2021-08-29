// cppimport
#include<iostream>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<map>
#include<algorithm>
#include<vector>
#include<cmath>
using namespace std;
namespace py = pybind11;
#define maxans 50000000
#define maxnm 500000
using namespace std;
    vector<int>q1[maxnm];
    vector<double>w1[maxnm];
    vector<int>co[maxnm];
    vector<int>q2[maxnm];
    vector<double>w2[maxnm];
    vector<double>la1[maxnm];
    vector<double>la2[maxnm];
    vector<double>zu[maxnm];
    double eps=1e-4;
    int n,dep,al,beta,m,kk;
    double c;
struct node
{
    int x;
    int y;
}v;
vector<node>vq;
int ans; 
double *ran;
double *gl;
int ranmax;
int rannow;


int bisearch(vector<double> &x,double y)
{
    int p=lower_bound(x.begin(),x.end(),y-eps)-x.begin();
    if(p<0||p>=x.size())
    {
        for(int i=0;i<x.size();i++)
        {
            cout<<x[i]<<'\t'<<endl;
        }
        cout<<'g'<<endl;
        cout<<p<<' '<<y<<' '<<x[x.size()-1]<<endl;
        throw std::runtime_error("error p"); 
    }
    return p;
}

int find(int x)
{
    for(int k=1;k<=dep;k++)
    {
        double g=*(ran+rannow);
     //   mexPrintf("gg%f %d %d\n",g,rannow,ranmax);
        rannow=(rannow+1)%ranmax;
    //    mexPrintf("gg%d %d %f\n",x,k,g);
        if(g>c)
        {
            return x;
        }
        else
        {
            g=*(ran+rannow);
            rannow=(rannow+1)%ranmax;
            int zhong=bisearch(zu[x],g);
          //  cout<<x<<' '<<zhong<<endl;
            if(zhong<q1[x].size())
            {
                int p=q1[x][zhong];
                g=*(ran+rannow);
                rannow=(rannow+1)%ranmax;
                int id=bisearch(w2[p],g);
                int y=q2[p][id];
                if(y==x)
                {
                    g=*(ran+rannow);
                    rannow=(rannow+1)%ranmax;
                    x=int(g*n-eps);
                }
                else
                {
                    x=y;
                }
            }
            else
            {
                int p=zhong-q1[x].size();
                g=*(ran+rannow);
                rannow=(rannow+1)%ranmax;
                int y=bisearch(la2[p],g);
                x=y;
            }
        }
    }
    double g=*(ran+rannow);
    rannow=(rannow+1)%ranmax;
    return int(g*n-eps);
}
void judge(vector<double> &x)
{
    for(int j=1;j<x.size();j++)
    {
        x[j]+=x[j-1];
    }
    if(x.size()>0&&x[x.size()-1]<1-eps)
    {
        printf("%.10f\n",x[x.size()-1]);
         throw std::runtime_error("error sum"); 
    }
}
void sample(void)
{
    ans=0;
    rannow=0;
    vq.clear();
    for(int i=0;i<n;i++)
    {
     //   cout<<i<<endl;
        judge(zu[i]);
    }
    for(int i=0;i<m;i++)
    {
     //   cout<<i<<endl;
        judge(w2[i]);
    }
    for(int i=0;i<kk;i++)
    {
      //  cout<<i<<endl;
        judge(la2[i]);
    }


 //   mexPrintf("gg\n");

     for(int j=1;j<=al;j++)
    {
        for(int i=0;i<n;i++)
        {
            
         //   cout<<i<<endl;
            int y=find(i);
         //   cout<<y<<'g'<<endl;
          //   mexPrintf("%d %d %d\n",j,i,y);
            int pt=co[y].size();
            if(pt==0)
                continue;
            double g=*(ran+rannow);
            rannow=(rannow+1)%ranmax;
            int num=(pt+int(g*beta-eps))/beta;
            for(int k=1;k<=num;k++)
            {
                double g=*(ran+rannow);
                rannow=(rannow+1)%ranmax;
               int yy=co[y][int(g*pt-eps)];
               v.x=i;
               v.y=yy;
               ans++;
               vq.push_back(v);
                if(ans>maxans)
                    throw std::runtime_error("error too many output"); 
            }

        }
    }
}
py::array_t<long long> negf (py::array_t<double>& inputw1, py::array_t<double>& inputw2, py::array_t<double>& inputla1, py::array_t<double>& inputla2, py::array_t<long long>& input2,py::array_t<double>& input3, py::array_t<double>& input4) {
    py::buffer_info bufw1 = inputw1.request();
    py::buffer_info bufw2 = inputw2.request();
    py::buffer_info bufla1 = inputla1.request();
    py::buffer_info bufla2 = inputla2.request();

    py::buffer_info buf2 = input2.request();
    py::buffer_info buf3 = input3.request();
    py::buffer_info buf4 = input4.request();
    
    // test
    int nn1=bufw1.shape[0];
    int nn2=buf2.shape[0];
    int nnw2=bufw2.shape[0];
    int nnla1=bufla1.shape[0];
    int nnla2=bufla2.shape[0];
    kk=bufla1.shape[1];
    int kk1=bufla2.shape[1];
    
    ranmax=buf4.shape[0];
    if(nn1!=nnw2||buf2.shape[1]!=2||buf3.shape[0]!=6||kk!=kk1)
    {
       throw std::runtime_error("error input");
    }
    

  //  cout<<nn1<<'\t'<<nn2<<endl;

    auto weight1 = inputw1.unchecked<1>();
    auto weight2 = inputw2.unchecked<1>();
    auto lat1 = inputla1.unchecked<2>();
    auto lat2 = inputla2.unchecked<2>();
     auto re = input2.unchecked<2>();
    auto canshu = input3.unchecked<1>();
    
    n=canshu(0);
    al=canshu(1);
    beta=canshu(2);
    dep=canshu(3);
    c=canshu(4);
    m=canshu(5);
    ran=(double *)buf4.ptr;
    if(nnla1!=n||nnla2!=n)
        throw std::runtime_error("error latent");
    
    for(int i=0;i<n;i++)
    {
        q1[i].clear();
        w1[i].clear();
        co[i].clear();
        la1[i].clear();
        zu[i].clear();
    }
    for(int i=0;i<m;i++)
    {
        q2[i].clear();
        w2[i].clear();
    }
    for(int i=0;i<kk;i++)
    {
        la2[i].clear();
    }

    
   for(int i=0;i<nn1;i++)
     {
        int x=re(i,0);
        int y=re(i,1);
        double z=weight1(i);
        if(x>=n||y>=m||x<0||y<0)
              throw std::runtime_error("error n"); 
        q1[x].push_back(y);
        w1[x].push_back(z);
        zu[x].push_back(z);

        double zf=weight2(i);
        q2[y].push_back(x);
        w2[y].push_back(zf);

         // mexPrintf("%d %d\n",x,y);
     }
    
     for(int i=0;i<nn2;i++)
     {
        int x=re(i,0);
        int y=re(i,1);
         if(x>=n||x<0)
              throw std::runtime_error("error n"); 
         co[x].push_back(y);
         // mexPrintf("%d %d\n",x,y);
     }
 //    cout<<nn1<<'\t'<<nn2<<endl;

     for(int i=0;i<n;i++)
     {
        for(int j=0;j<kk;j++)
        {
            double z=lat1(i,j);
            la1[i].push_back(z);
            zu[i].push_back(z);
        }
     }
     for(int i=0;i<kk;i++)
     {
        for(int j=0;j<n;j++)
        {
            double z=lat2(j,i);
            la2[i].push_back(z);
        }
     }

     sample();
     auto result = py::array_t<long long>(2*ans);
      py::buffer_info buf5 = result.request();   
     result.resize({ans,2});
     auto p5=result.mutable_unchecked<2>();
    for(int i=0;i<ans;i++)
    {
        v=vq[i];
       p5(i,0)=v.x;
       p5(i,1)=v.y;
    }
   return result;
    
    
    
    
}




PYBIND11_MODULE(biwalker, m) {

    m.doc() = "Simple demo using numpy!";
    m.def("negf", &negf);
}

/*
<%
setup_pybind11(cfg)
%>
*/













