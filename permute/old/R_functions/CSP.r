CSP <- function(y, x, C = 1000, exact = FALSE) {
    
    
    
    
    ########## PRE-PROCESSING#########
    
    lA <- unique(x[, 1])
    lB <- unique(x[, 2])
    
    A <- length(lA)
    B <- length(lB)
    
    n <- length(y)/(A * B)
    
    Y <- array(0, dim = c(A, B, n))
    
    
    for (i in 1:A) {
        for (j in 1:B) {
            
            Y[i, j, ] <- y[x[, 1] == i & x[, 2] == j]
        }
    }
    
    
    ############ OBTAINING CSP PERMUTATIONS####################
    
    
    if (exact == TRUE) {
        require(combinat)
        U <- combn(2 * n, n)
        ind.perm <- apply(U, 2, function(x) {
            c(c(1:(2 * n))[x], c(1:(2 * n))[-x])
        })
        ind.perm <- cbind(c(1:(2 * n)), ind.perm)
        C <- dim(ind.perm)[2]
    }
    
    if (exact == FALSE) {
        ind.perm <- apply(matrix(runif(2 * n * C), ncol = C), 2, rank)
        ind.perm <- cbind(c(1:(2 * n)), ind.perm)
        C <- dim(ind.perm)[2]
    }
    
    T.A <- array(0, dim = c(C, 1))
    T.B <- array(0, dim = c(C, 1))
    T.AB.a <- array(0, dim = c(C, 1))
    T.AB.b <- array(0, dim = c(C, 1))
    T.AB <- array(0, dim = c(C, 1))
    
    
    ########## COMPUTING THE OBSERVED STATISTICS########################
    
    
    # Factor A
    
    # for(i in 1:(A-1)){ for(s in (i+1):A){
    # T.A[1]<-T.A[1]+(sum(Y[i,,])-sum(Y[s,,]))^2 } }
    
    # Factor B
    
    # for(j in 1:(B-1)){ for(h in (j+1):B){
    # T.B[1]<-T.B[1]+(sum(Y[,j,])-sum(Y[,h,]))^2 } }
    
    # Interaction (a)
    
    # for(i in 1:(A-1)){ for(s in (i+1):A){
    
    # for(j in 1:(B-1)){ for(h in (j+1):B){
    
    # T.AB.a[1]<-T.AB.a[1] + (sum(Y[i,j, ])-sum(Y[s,j, ])-sum(Y[i,h, ])+sum(Y[s,h,
    # ]))^2
    
    # } } } }
    
    # Interaction (b)
    
    # T.AB.b[1]<-T.AB.a[1]
    
    
    
    ###### OBTAINING THE SYNCHRONIZED PERMUTATION DISTRIBUTION#######
    
    
    for (cc in 1:C) {
        # print(cc)
        
        Y.perm <- Y
        
        
        ### Column pemrmutations
        
        
        
        for (i in 1:(A - 1)) {
            for (s in (i + 1):A) {
                
                for (j in 1:B) {
                  
                  pool <- c(Y[i, j, ], Y[s, j, ])
                  
                  
                  pool <- pool[ind.perm[, cc]]
                  
                  Y.perm[i, j, ] <- pool[c(1:n)]
                  Y.perm[s, j, ] <- pool[-c(1:n)]
                }
                
                
                # Factor A
                T.A[cc] <- T.A[cc] + (sum(Y.perm[i, , ]) - sum(Y.perm[s, , ]))^2
                
                
                for (j in 1:(B - 1)) {
                  for (h in (j + 1):B) {
                    
                    
                    # Interaction (a)
                    T.AB.a[cc] <- T.AB.a[cc] + (sum(Y.perm[i, j, ]) - sum(Y.perm[s, 
                      j, ]) - sum(Y.perm[i, h, ]) + sum(Y.perm[s, h, ]))^2
                    
                    
                  }
                }
            }
        }
        
        
        ### Row permutations
        
        
        
        for (j in 1:(B - 1)) {
            for (h in (j + 1):B) {
                
                for (i in 1:A) {
                  
                  pool <- c(Y[i, j, ], Y[i, h, ])
                  
                  
                  pool <- pool[ind.perm[, cc]]
                  
                  
                  Y.perm[i, j, ] <- pool[c(1:n)]
                  Y.perm[i, h, ] <- pool[-c(1:n)]
                }
                
                
                # Factor B
                T.B[cc] <- T.B[cc] + (sum(Y.perm[, j, ]) - sum(Y.perm[, h, ]))^2
                
                
                for (i in 1:(A - 1)) {
                  for (s in (i + 1):A) {
                    
                    
                    # Interaction (b)
                    T.AB.b[cc] <- T.AB.b[cc] + (sum(Y.perm[i, j, ]) - sum(Y.perm[s, 
                      j, ]) - sum(Y.perm[i, h, ]) + sum(Y.perm[s, h, ]))^2
                    
                  }
                }
            }
        }
        
        
    }  #end cc
    
    C = C - 1
    
    T.A <- round(T.A, digits = 8)
    T.B <- round(T.B, digits = 8)
    T.AB.a <- round(T.AB.a, digits = 8)
    T.AB.b <- round(T.AB.b, digits = 8)
    
    T.AB <- apply(cbind(T.AB.a, T.AB.b), 1, sum)
    
    
    
    digits <- ifelse(exact == TRUE, 6, log(C, 10))
    
    pa <- round(sum(T.A[-1] >= T.A[1])/C, digits)
    pb <- round(sum(T.B[-1] >= T.B[1])/C, digits)
    pab <- round(sum(T.AB[-1] >= T.AB[1])/C, digits)
    
    
    pab.a <- round(sum(T.AB.a[-1] >= T.AB.a[1])/C, digits)
    pab.b <- round(sum(T.AB.b[-1] >= T.AB.b[1])/C, digits)
    ###### RESULTS
    
    
    min.sig = rep(2/choose(2 * n, n), 2)
    
    return(list(pa = pa, pb = pb, pab = pab, pab.a = pab.a, pab.b = pab.b, TA = T.A[1], 
        TB = T.B[1], TAB.a = T.AB.a[1], TAB.b = T.AB.b[1], type = "Constrained", 
        C = C, min.sig = min.sig, exact = exact))
} 
