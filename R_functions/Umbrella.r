umbrella<-function(x,y,B=1000,repeated=FALSE,trend=FALSE,alt=NULL){


## x: array of data
## y: array of labels

setwd("F:/R/NPC")
source("combine.r")
source("T_to_P.r")

K=length(unique(y))
label<-unique(y)

N<-length(x)


if(repeated==FALSE){
U<-array(runif(N*B),dim=c(N,B))
U<-apply(U,2,rank)
}



if(repeated==TRUE){
n=N/K
U<-array(runif(n*B*K),dim=c(B,n,K))
U<-aperm(apply(U,c(1,2),rank),c(2,1,3))
}



T<-array(0,dim=c((B+1),K,2))



 for(k.hat in 1:K){
 

 g<-c(1:k.hat)
 if(k.hat==1){g=c(1,1)}
 for(j in 1:(length(g)-1)){
 ID<-g[c(1:j)]
 ID.not<-g[(-c(1:j))]


T[1,k.hat,1]<-T[1,k.hat,1]+mean(x[y%in%ID.not])-mean(x[y%in%ID])


 }

 g<-c(k.hat:K)
 if(k.hat==K){g=c(K,K)}
 for(j in 1:(length(g)-1)){
 ID<-g[c(1:j)]
 ID.not<-g[(-c(1:j))]




  T[1,k.hat,2]<-T[1,k.hat,2]+mean(x[y%in%ID])-mean(x[y%in%ID.not])
 }




for(bb in 2:(B+1)){


if(repeated==FALSE){
x.perm<-x[U[,(bb-1)]]
y.perm=y
}

if(repeated==TRUE){

x.perm<-matrix(x,nrow=n,byrow=FALSE)					##c'era TRUE

for(i in 1:n){
x.perm[i,]<-x.perm[i,U[(bb-1),,i]]
}

x.perm<-as.vector(x.perm)
y.perm<-rep(seq(1,K),each=n)

}



 g<-c(1:k.hat)
 if(k.hat==1){g=c(1,1)}
 for(j in 1:(length(g)-1)){
 ID<-g[c(1:j)]
 ID.not<-g[(-c(1:j))]
  

  T[bb,k.hat,1]<-T[bb,k.hat,1]+mean(x.perm[y.perm%in%ID.not])-mean(x.perm[y.perm%in%ID])


 }

 g<-c(k.hat:K)
 if(k.hat==K){g=c(K,K)}
 for(j in 1:(length(g)-1)){
 ID<-g[c(1:j)]
 ID.not<-g[(-c(1:j))]
  


  T[bb,k.hat,2]<-T[bb,k.hat,2]+mean(x.perm[y.perm%in%ID])-mean(x.perm[y.perm%in%ID.not])



 }

 }
 }         



P<-T.to.P(T)

T1<-apply(P,c(1,2),function(x){-2*log(prod(x))})
P1<-T.to.P(T1)			

p.part<-P1[1,]




if(trend==FALSE){

T2<-combine(P1,which=2,fun="Tippett")	
P.glob<-sum(T2<=T2[1])/B
max<-label[p.part==min(p.part)]

cat("\n Unknow peak \n") 
}


if(trend==TRUE){

#T2<-combine(P1[,c(1,K)],which=2,fun="Tippett")	
#P.glob<-sum(T2<=T2[1])/B

if(alt=="less"){cat("\n Know peak equal to",K,"\n");P.glob=P1[1,K];max<-K}
if(alt=="greater"){cat("\n Know peak equal to",1,"\n");P.glob=P1[1,1];max=1}
}



ris<-list(Global.p.value=P.glob,Partial.p.values=p.part,Max=max)		##,T=T,U=U
return(ris)
#return(T2)
}




