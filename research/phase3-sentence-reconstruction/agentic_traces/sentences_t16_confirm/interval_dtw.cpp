extern "C" void all_dtw(const double* S, int n, int m, double* out) {
 const double neg=-1e9; const double penalty=.035;double D[20000];
 if((n+1)*(m+1)>20000)return;
 for(int s=1;s<n-4;s++){
  for(int k=0;k<(n+1)*(m+1);k++)D[k]=neg;D[s*(m+1)]=0;
  int imax=(n-1<s+40?n-1:s+40);
  for(int i=s+1;i<=imax;i++){
   for(int j=1;j<=m;j++){
    double v=D[(i-1)*(m+1)+j-1]+S[(i-1)*m+j-1];
    if(i>=s+2){double v2=D[(i-2)*(m+1)+j-1]+S[(i-2)*m+j-1]+S[(i-1)*m+j-1]-penalty;if(v2>v)v=v2;}
    if(j>=2){double v3=D[(i-1)*(m+1)+j-2]+.5*(S[(i-1)*m+j-2]+S[(i-1)*m+j-1])-penalty;if(v3>v)v=v3;}
    D[i*(m+1)+j]=v;
   }
   if(i-s>=4)out[s*(n+1)+i]=D[i*(m+1)+m]/(i-s);
  }
 }
}
extern "C" void subseq(const double* S,int n,int m,double* out) {
 double D[20000];int P[20000];if((n+1)*(m+1)>20000)return;
 for(int k=0;k<(n+1)*(m+1);k++)D[k]=-1e9;
 for(int j=0;j<=m;j++)D[j]=0;
 for(int i=1;i<=n;i++)for(int j=1;j<=m;j++){
  double v=D[(i-1)*(m+1)+j-1]+S[(i-1)*m+j-1];int p=0;
  if(i>=2){double v2=D[(i-2)*(m+1)+j-1]+S[(i-2)*m+j-1]+S[(i-1)*m+j-1]-.035;if(v2>v){v=v2;p=1;}}
  if(j>=2){double v3=D[(i-1)*(m+1)+j-2]+.5*(S[(i-1)*m+j-2]+S[(i-1)*m+j-1])-.035;if(v3>v){v=v3;p=2;}}
  D[i*(m+1)+j]=v;P[i*(m+1)+j]=p;
 }
 int end=1;for(int j=2;j<=m;j++)if(D[n*(m+1)+j]>D[n*(m+1)+end])end=j;
 out[0]=D[n*(m+1)+end]/n;out[2]=end;
 int i=n,j=end;while(i>0){int p=P[i*(m+1)+j];if(p==0){i--;j--;}else if(p==1){i-=2;j--;}else{i--;j-=2;}}
 out[1]=j;
}
