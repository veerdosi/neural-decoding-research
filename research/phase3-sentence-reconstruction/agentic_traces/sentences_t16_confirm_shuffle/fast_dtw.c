#include <stdlib.h>
#include <math.h>
float match(float *s,int n,int m,int anchored,int *start,int *end,int *path){
 float *d=malloc((n+1)*(m+1)*sizeof(float));int *p=calloc((n+1)*(m+1),sizeof(int));
 for(int k=0;k<(n+1)*(m+1);k++)d[k]=-1e9;
 d[0]=0;if(!anchored)for(int j=1;j<=m;j++)d[j]=0;
 for(int i=1;i<=n;i++)for(int j=1;j<=m;j++){
  float a=d[(i-1)*(m+1)+j-1]+s[(i-1)*m+j-1]; int v=0;
  if(i>=2){float b=d[(i-2)*(m+1)+j-1]+s[(i-2)*m+j-1]+s[(i-1)*m+j-1]-.035;if(b>a){a=b;v=1;}}
  if(j>=2){float b=d[(i-1)*(m+1)+j-2]+.5*(s[(i-1)*m+j-2]+s[(i-1)*m+j-1])-.035;if(b>a){a=b;v=2;}}
  d[i*(m+1)+j]=a;p[i*(m+1)+j]=v;
 }
 int j=1;for(int k=2;k<=m;k++)if(d[n*(m+1)+k]>d[n*(m+1)+j])j=k; if(anchored==2)j=m;
 float score=d[n*(m+1)+j]/n;*end=j;int i=n;
 while(i>0 && j>0){int v=p[i*(m+1)+j];path[i-1]=j-1;if(v==0){i--;j--;}else if(v==1){path[i-2]=j-1;i-=2;j--;}else{i--;j-=2;}}
 *start=j;free(d);free(p);return score;
}
