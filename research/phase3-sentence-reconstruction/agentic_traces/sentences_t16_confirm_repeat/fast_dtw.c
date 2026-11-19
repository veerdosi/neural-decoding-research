#include <stdlib.h>
void subdtw(double *S,int n,int m,double penalty,double *D,signed char *P){
 for(int i=0;i<=n;i++)for(int j=0;j<=m;j++)D[i*(m+1)+j]=(i==0)?0:-1e9;
 for(int i=1;i<=n;i++)for(int j=1;j<=m;j++){
  double v=D[(i-1)*(m+1)+j-1]+S[(i-1)*m+j-1];int p=0;
  if(i>=2){double z=D[(i-2)*(m+1)+j-1]+S[(i-2)*m+j-1]+S[(i-1)*m+j-1]-penalty;if(z>v){v=z;p=1;}}
  if(j>=2){double z=D[(i-1)*(m+1)+j-2]+.5*(S[(i-1)*m+j-2]+S[(i-1)*m+j-1])-penalty;if(z>v){v=z;p=2;}}
  D[i*(m+1)+j]=v;P[i*(m+1)+j]=p;
 }
}
void interval_scores(double *S,int n,int m,int minlen,int maxlen,double penalty,double *out){
 double *D=malloc((maxlen+1)*(m+1)*sizeof(double));
 for(int s=1;s<n-minlen;s++){
  int nn=n-1-s;if(nn>maxlen)nn=maxlen;
  for(int i=0;i<=nn;i++)for(int j=0;j<=m;j++)D[i*(m+1)+j]=-1e9;
  D[0]=0;
  for(int i=1;i<=nn;i++)for(int j=1;j<=m;j++){
   double v=D[(i-1)*(m+1)+j-1]+S[(s+i-1)*m+j-1];
   if(i>=2){double z=D[(i-2)*(m+1)+j-1]+S[(s+i-2)*m+j-1]+S[(s+i-1)*m+j-1]-penalty;if(z>v)v=z;}
   if(j>=2){double z=D[(i-1)*(m+1)+j-2]+.5*(S[(s+i-1)*m+j-2]+S[(s+i-1)*m+j-1])-penalty;if(z>v)v=z;}
   D[i*(m+1)+j]=v;
  }
  for(int i=minlen;i<=nn;i++)out[s*(n+1)+s+i]=D[i*(m+1)+m]/i;
 }
 free(D);
}
