#include "mex.h"
#include<vector>
using namespace std;
void mexFunction(int nlhs,mxArray *plhs[], int nrhs,const mxArray *prhs[])
{
     int  nn=mxGetM(prhs[0]); 
     //获得行数
     int n=*(mxGetPr(prhs[2]));
     int m=mxGetN(prhs[1]); 
     double *p=mxGetPr(prhs[0]);
    double *pa=mxGetPr(prhs[1]);

     plhs[0]=mxCreateDoubleMatrix(n,m,mxREAL);
      double *a = mxGetPr(plhs[0]);
      for(int i=0;i<n;i++)
      {
        for(int j=0;j<m;j++)
        {
            a[i+j*n]=0;
        }
      }
      for(int i=0;i<nn;i++)
      {
        int x=*(p+i)-1;
        for(int j=0;j<m;j++)
        {
            a[x+j*n]+=*(pa+i+j*nn);
        }
      }
}








