IC<-function(x1,x2,conf.lev=0.95,B=1000,max.delta=20,length.delta=100){

n1<-length(x1)
n2<-length(x2)
n<-n1+n2


alpha<-(1-conf.lev)/2

delta<-seq(1/length.delta,max.delta,length.out=length.delta)

p=1
k=1

while(p>alpha & k<length(delta)){

y1<-x1-mean(x1)+mean(x2)+delta[k]
y2<-x2

y<-c(y1,y2)

######Parte che fa il test a due campioni########

T<-array(0,dim=c((B+1),1))

T[1]<-mean(y1)-mean(y2)

for(bb in 2:(B+1)){
y.perm<-sample(y)
T[bb]<-mean(y.perm[1:n1])-mean(y.perm[-c(1:n1)])
}


p<-mean(T[-1]>=T[1])

##################################################
cat(p,k,"\n")
k=k+1

}

upper<-mean(x1)-mean(x2)+delta[k]
lower<-mean(x1)-mean(x2)-delta[k]


return(c(lower,upper))

}

IC<-function(x1,x2,conf.lev=0.95,B=1000,max.delta=20,length.delta=100){

n1<-length(x1)
n2<-length(x2)
n<-n1+n2


alpha<-(1-conf.lev)/2

delta<-seq(1/length.delta,max.delta,length.out=length.delta)

p=1
k=1

while(p>alpha & k<length(delta)){

y1<-x1-mean(x1)+mean(x2)+delta[k]
y2<-x2

y<-c(y1,y2)

######Parte che fa il test a due campioni########

T<-array(0,dim=c((B+1),1))

T[1]<-mean(y1)-mean(y2)

for(bb in 2:(B+1)){
y.perm<-sample(y)
T[bb]<-mean(y.perm[1:n1])-mean(y.perm[-c(1:n1)])
}


p<-mean(T[-1]>=T[1])

##################################################
cat("iter:",k,"\t p.val:",p,"\n")
k=k+1

}

upper<-mean(x1)-mean(x2)+delta[k]
lower<-mean(x1)-mean(x2)-delta[k]


return(c(lower,upper))

}

## in Job satisfaction example:
#set.seed(101)
#IC(X[Y==1],X[Y==2],max.delta=10,length.delta=100)
#[1]  8.805303 25.778030
