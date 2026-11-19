#include <stdlib.h>
void subdtw(double *S,int n,int m,double penalty,double *result,int *path,int anchored){
 double *D=malloc((n+1)*(m+1)*sizeof(double));int *P=calloc((n+1)*(m+1),sizeof(int));
 for(int k=0;k<(n+1)*(m+1);k++)D[k]=-1e9;
 if(anchored) D[0]=0;else for(int j=0;j<=m;j++)D[j]=0;
 for(int i=1;i<=n;i++)for(int j=1;j<=m;j++){
 double best=D[(i-1)*(m+1)+j-1]+S[(i-1)*m+j-1];int p=0;
 if(i>=2){double v=D[(i-2)*(m+1)+j-1]+S[(i-2)*m+j-1]+S[(i-1)*m+j-1]-penalty;if(v>best){best=v;p=1;}}
 if(j>=2){double v=D[(i-1)*(m+1)+j-2]+.5*(S[(i-1)*m+j-2]+S[(i-1)*m+j-1])-penalty;if(v>best){best=v;p=2;}}
 D[i*(m+1)+j]=best;P[i*(m+1)+j]=p;
 }
 int j=1;for(int k=2;k<=m;k++)if(D[n*(m+1)+k]>D[n*(m+1)+j])j=k; if(anchored==2)j=m;
 result[0]=D[n*(m+1)+j]/n;result[2]=j;int i=n,l=0;
 while(i>0&&j>0){int p=P[i*(m+1)+j];path[2*l]=i-1;path[2*l+1]=j-1;l++;
 if(p==0){i--;j--;}else if(p==1){path[2*l]=i-2;path[2*l+1]=j-1;l++;i-=2;j--;}else{path[2*l]=i-1;path[2*l+1]=j-2;l++;i--;j-=2;}}
 result[1]=j;result[3]=l;free(D);free(P);
}
