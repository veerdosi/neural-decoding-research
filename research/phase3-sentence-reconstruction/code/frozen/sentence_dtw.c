/* Exact Sol-authored local_dtw.py score/path recurrence, accelerated in C99.
   No external libraries, model parameters, or target labels. */
static double subsequence_core(const double* s, int n, int m,
                      int* out_path, int* out_len, int* out_start, int* out_end, int prefix) {
  double d[(n+1)*(m+1)];int p[(n+1)*(m+1)];
  for(int k=0;k<(n+1)*(m+1);++k){d[k]=-1e9;p[k]=0;}
  for(int j=0;j<=m;++j)if(!prefix || j==0)d[j]=0;
  for(int i=1;i<=n;++i)for(int j=1;j<=m;++j){
    double a=d[(i-1)*(m+1)+j-1]+s[(i-1)*m+j-1];
    double b=i>=2?d[(i-2)*(m+1)+j-1]+s[(i-2)*m+j-1]+s[(i-1)*m+j-1]-.035:-1e9;
    double c=j>=2?d[(i-1)*(m+1)+j-2]+.5*(s[(i-1)*m+j-2]+s[(i-1)*m+j-1])-.035:-1e9;
    int k=0;double best=a;if(b>best){best=b;k=1;}if(c>best){best=c;k=2;}
    d[i*(m+1)+j]=best;p[i*(m+1)+j]=k;
  }
  int j=0;for(int t=1;t<=m;++t)if(d[n*(m+1)+t]>d[n*(m+1)+j])j=t;
  double score=d[n*(m+1)+j]/n;*out_end=j;int i=n;
  int path[4*(n+m)],length=0;
  while(i>0){
    int k=p[i*(m+1)+j];path[length++]=i-1;path[length++]=j-1;
    if(k==0){--i;--j;}
    else if(k==1){path[length++]=i-2;path[length++]=j-1;i-=2;--j;}
    else{path[length++]=i-1;path[length++]=j-2;--i;j-=2;}
  }
  *out_start=j;*out_len=length/2;
  for(int k=0;k<*out_len;++k){out_path[k*2]=path[(*out_len-k-1)*2];out_path[k*2+1]=path[(*out_len-k-1)*2+1];}
  return score;
}
double subsequence_dtw(const double* s,int n,int m,int* p,int* l,int* a,int* b){return subsequence_core(s,n,m,p,l,a,b,0);}
double prefix_dtw(const double* s,int n,int m,int* p,int* l,int* a,int* b){return subsequence_core(s,n,m,p,l,a,b,1);}
double global_dtw(const double* s,int n,int m,int* out_path,int* out_len,int* out_start,int* out_end){
  double d[(n+1)*(m+1)];int p[n*m];
  for(int k=0;k<(n+1)*(m+1);++k)d[k]=1e100;d[0]=0;
  for(int i=1;i<=n;++i)for(int j=1;j<=m;++j){
    double a=d[(i-1)*(m+1)+j-1],b=d[(i-1)*(m+1)+j]+.12,c=d[i*(m+1)+j-1]+.12;
    int k=0;double best=a;if(b<best){best=b;k=1;}if(c<best){best=c;k=2;}
    d[i*(m+1)+j]=1-s[(i-1)*m+j-1]+best;p[(i-1)*m+j-1]=k;
  }
  int i=n-1,j=m-1,path[4*(n+m)],length=0;
  while(i>=0 && j>=0){path[length++]=i;path[length++]=j;int k=p[i*m+j];if(k==0){--i;--j;}else if(k==1)--i;else --j;}
  *out_len=length/2;*out_start=0;*out_end=m;
  for(int k=0;k<*out_len;++k){out_path[k*2]=path[(*out_len-k-1)*2];out_path[k*2+1]=path[(*out_len-k-1)*2+1];}
  return d[n*(m+1)+m]/(*out_len);
}
