stoch.ord2<-function(y,x,z=NULL,alt=c(-1,1),B=1000,cat=0,rep=FALSE,seed=101){

K=length(unique(x))
g=unique(sort(x))
n=table(x)


if(K==1){return(1)}

T<-array(0,dim=c((B+1),K-1))




for(j in 1:(K-1)){

 ID<-g[1:j]
 ID.not<-g[-c(1:j)]

#cat("ID:",ID,"\t ID.not:",ID.not,"\n")

if(cat==0){
s= (sum((y[x%in%ID]-mean(y[x%in%ID]))^2)+sum((y[x%in%ID.not]-mean(y[x%in%ID.not]))^2))/(sum(n)-2)
if(alt==-1){T[1,j]<-(mean(y[x%in%ID])-mean(y[x%in%ID.not]))/sqrt(s)}
if(alt==1){T[1,j]<-(mean(y[x%in%ID.not])-mean(y[x%in%ID]))/sqrt(s)}
}


if(cat==1){	### two pseudo-samples



label = as.integer(names(table(y)[table(y)>0]))		## categories
l = length(label)

N = array(0,dim=c(l,2)) ; rownames(N)=label

y1 = y[x%in%ID]
y2 = y[x%in%ID.not]

for(i in 1:l){
N[i,] = c(sum(y1%in%label[i]),sum(y2%in%label[i]))
}



N=apply(N,2,cumsum)

if(alt==1){T[1,j]<-sum(N[,1]/(apply(N,1,sum)*(sum(N)-apply(N,1,sum)))^.5)}
if(alt==-1){T[1,j]<-sum(N[,2]/(apply(N,1,sum)*(sum(N)-apply(N,1,sum)))^.5)}

}## end cat


}## end j


set.seed(seed)

for(bb in 2:(B+1)){


if(rep==FALSE){y.perm<-sample(y)}

if(rep==TRUE){
y.perm<-y
n = length(unique(z))
n.star = sample(n)
for(i in 1:n){
y.perm[z==i] = y[z==n.star[i]]
}
}


for(j in 1:(K-1)){

 ID<-g[1:j]
 ID.not<-g[-c(1:j)]

#cat("ID:",ID,"\t ID.not:",ID.not,"\n")

if(cat==0){
s= (sum((y.perm[x%in%ID]-mean(y.perm[x%in%ID]))^2)+sum((y.perm[x%in%ID.not]-mean(y.perm[x%in%ID.not]))^2))/(sum(n)-2)
if(alt==-1){T[bb,j]<-(mean(y.perm[x%in%ID])-mean(y.perm[x%in%ID.not]))/sqrt(s)}
if(alt==1){T[bb,j]<-(mean(y.perm[x%in%ID.not])-mean(y.perm[x%in%ID]))/sqrt(s)}
}

if(cat==1){


label = as.integer(names(table(y)[table(y)>0]))		## categories
l = length(label)

N = array(0,dim=c(l,2)) ; rownames(N)=label

y1 = y.perm[x%in%ID]
y2 = y.perm[x%in%ID.not]

for(i in 1:l){
N[i,] = c(sum(y1%in%label[i]),sum(y2%in%label[i]))
}

###	N=apply(N,2,function(x){cumsum(x)/sum(x)})	### relative frequencies

N=apply(N,2,cumsum)

if(alt==1){T[bb,j]<-sum(N[,1]/(apply(N,1,sum)*(sum(N)-apply(N,1,sum)))^.5)}
if(alt==-1){T[bb,j]<-sum(N[,2]/(apply(N,1,sum)*(sum(N)-apply(N,1,sum)))^.5)}

}## end cat



}## end j

}## end bb


P=t2p(T)

T1<-apply(P,1,function(x){-2*log(prod(x))})

P1=t2p(T1)

p.val=P1[1]
return(P1)
}