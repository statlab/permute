stoch.ord<-function(y,x,alt=c(-1,1),B=1000,seed=101){

K=length(unique(x))
g=unique(sort(x))
n=table(x)

if(K==1){return(1)}

T<-array(0,dim=c((B+1),K-1))

for(j in 1:(K-1)){

 ID<-g[1:j]
 ID.not<-g[-c(1:j)]

#cat("ID:",ID,"\t ID.not:",ID.not,"\n")

s= (sum((y[x%in%ID]-mean(y[x%in%ID]))^2)+sum((y[x%in%ID.not]-mean(y[x%in%ID.not]))^2))/(sum(n)-2)

if(alt==-1){T[1,j]<-(mean(y[x%in%ID])-mean(y[x%in%ID.not]))/sqrt(s)}
if(alt==1){T[1,j]<-(mean(y[x%in%ID.not])-mean(y[x%in%ID]))/sqrt(s)}

}

set.seed(seed)

for(bb in 2:(B+1)){

y.perm<-sample(y)

for(j in 1:(K-1)){

 ID<-g[1:j]
 ID.not<-g[-c(1:j)]

#cat("ID:",ID,"\t ID.not:",ID.not,"\n")


s= (sum((y.perm[x%in%ID]-mean(y.perm[x%in%ID]))^2)+sum((y.perm[x%in%ID.not]-mean(y.perm[x%in%ID.not]))^2))/(sum(n)-2)

if(alt==-1){T[bb,j]<-(mean(y.perm[x%in%ID])-mean(y.perm[x%in%ID.not]))/sqrt(s)}
if(alt==1){T[bb,j]<-(mean(y.perm[x%in%ID.not])-mean(y.perm[x%in%ID]))/sqrt(s)}

}

}#fine bb


P=t2p(T)

T1<-apply(P,1,function(x){-2*log(prod(x))})

P1=t2p(T1)

p.val=P1[1]
return(P1)
}