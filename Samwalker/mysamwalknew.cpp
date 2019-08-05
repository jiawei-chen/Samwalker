#include "mex.h"
#include<vector>
using namespace std;
    vector<int>q[100005];
    vector<double>w[100005];
    vector<int>co[100005];
    double eps=1e-12;
    int n,dep,al,beta;
    double c;
#define maxans 5000000
struct node
{
    int x;
    int y;
}v[maxans+3];
int ans; 
double *ran;
double *gl;
int ranmax;
int rannow;
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
            int p=-1;
            for(int l=0;l<w[x].size();l++)
            {
                if(w[x][l]>=g-eps)
                {
                    p=l;
                    break;
                }
            }
            if(p==-1)
            {
                g=*(ran+rannow);
                rannow=(rannow+1)%ranmax;
                x=int(g*n-eps)+1;
            }
            else
                x=q[x][p];
        }
    }
    double g=*(ran+rannow);
    rannow=(rannow+1)%ranmax;
    return int(g*n-eps)+1;
}
void sample(void)
{
    ans=0;
    rannow=0;
    for(int i=1;i<=n;i++)
    {
        for(int j=1;j<w[i].size();j++)
        {
            w[i][j]+=w[i][j-1];
        }
    }
 //   mexPrintf("gg\n");

     for(int j=1;j<=al;j++)
    {
        for(int i=1;i<=n;i++)
        {
            int y=find(i);
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
                v[ans].x=i;
                v[ans++].y=yy;
                if(ans>maxans)
                   mexErrMsgTxt("error too many output");  
            }

        }
    }
}
void mexFunction(int nlhs,mxArray *plhs[], int nrhs,const mxArray *prhs[])
{
     int  nn=mxGetM(prhs[0]); //»ñµÃÐÐÊý
     int nnk=mxGetN(prhs[0]);
     if(nrhs!=9)
        mexErrMsgTxt("error number of input"); 
     if(nnk!=3)
         mexErrMsgTxt("error matrix 1"); 
     int mm=mxGetM(prhs[1]);
     int mmk=mxGetN(prhs[1]);
       if(mmk!=2)
         mexErrMsgTxt("error matrix 2");
    

     n=*(mxGetPr(prhs[2]));
     al=*(mxGetPr(prhs[3]));
     beta=*(mxGetPr(prhs[4]));
     dep=*(mxGetPr(prhs[5]));
     c=*(mxGetPr(prhs[6]));

     ranmax=mxGetM(prhs[7]);

      // mexPrintf("%d %d %d %d %f %d\n",n,al,beta,dep,c,ranmax);

     double *p=mxGetPr(prhs[0]);
     double *pc=mxGetPr(prhs[1]);
     double *gl=mxGetPr(prhs[8]);
      int ngl=mxGetM(prhs[8]);

     if(ngl!=n)
        mexErrMsgTxt("error fa0");

     ran=mxGetPr(prhs[7]);
     for(int i=1;i<=n;i++)
     {
        q[i].clear();
        w[i].clear();
        co[i].clear();
     }
     for(int i=0;i<nn;i++)
     {
        int x=*(p+i)+eps;
        int y=*(p+nn+i)+eps;
        double z=*(p+2*nn+i);
        if(x>n||y>n||x<=0||y<=0)
             mexErrMsgTxt("error n"); 
        q[x].push_back(y);
        w[x].push_back(z);
         // mexPrintf("%d %d\n",x,y);
     }
     for(int i=0;i<mm;i++)
     {
        int x=*(pc+i)+eps;
        int y=*(pc+mm+i)+eps;
         if(x>n||x<=0)
             mexErrMsgTxt("error n"); 
         co[x].push_back(y);
         // mexPrintf("%d %d\n",x,y);
     }
     sample();
     plhs[0] = mxCreateDoubleMatrix(ans, 2, mxREAL);
     double *out=mxGetPr(plhs[0]);
     for(int i=0;i<ans;i++)
     {
        *(out+i)=v[i].x;
        *(out+i+ans)=v[i].y;     
     }
}








