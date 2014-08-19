FWE.minP<-function(P){


p.oss<-P[1,]
p.ord<-sort(p.oss,decreasing=FALSE)
o<-order(p.oss,decreasing=FALSE)

B=dim(P)[1]-1
p=dim(P)[2]

p.ris<-array(0,dim=c(p,1))


P.ord<-P[,o]


T=apply(P.ord,1,min)
p.ris[1] = mean(T[-1]<=T[1])

if(p>2){
for(j in 2:(p-1)){

T=apply(P.ord[,j:p],1,min)
p.ris[j] = max(mean(T[-1]<=T[1]),p.ris[(j-1)])
}
}
p.ris[p] = max(p.ord[p],p.ris[p-1])

p.ris[o]=p.ris

rownames(p.ris)=colnames(P)

return(p.ris)
}
