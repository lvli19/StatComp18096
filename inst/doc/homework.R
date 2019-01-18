## ----eval=FALSE----------------------------------------------------------
#  x<-rnorm(10)
#  y<-rnorm(10)
#  rbind(x,y)

## ----eval=FALSE----------------------------------------------------------
#  plot(x,y)

## ----eval=FALSE----------------------------------------------------------
#  # replace some plotting parameters
#  plot(x,y,xlab="Ten random values",ylab="Ten other values",xlim=c(-2,2),ylim=c(-2,2),pch=22,col="red",bg="yellow",bty="l",tcl=0.4,main="How to customize a plot with R",las=1,cex=1.5)

## ----eval=FALSE----------------------------------------------------------
#  opar <- par()
#  par(bg="lightyellow", col.axis="blue", mar=c(4, 4, 2.5, 0.25))
#  plot(x, y, xlab="Ten random values", ylab="Ten other values",
#  xlim=c(-2, 2), ylim=c(-2, 2), pch=22, col="red", bg="yellow",
#  bty="l", tcl=-.25, las=1, cex=1.5)
#  title("How to customize a plot with R (bis)", font.main=3, adj=1)

## ----eval=FALSE----------------------------------------------------------
#  opar <- par()
#  # change the backgroud and margin
#  par(bg="lightgray", mar=c(2.5, 1.5, 2.5, 0.25))
#  # plot a blank figure
#  plot(x, y, type="n", xlab="", ylab="", xlim=c(-2, 2), ylim=c(-2, 2), xaxt="n", yaxt="n")
#  # change the color of the drawing area
#  rect(-3, -3, 3, 3, col="cornsilk")
#  points(x, y, pch=10, col="red", cex=2)
#  axis(side=1, c(-2, 0, 2), tcl=-0.2, labels=FALSE)
#  axis(side=2, -1:1, tcl=-0.2, labels=FALSE)
#  title("How to customize a plot with R (ter)", font.main=4, adj=1, cex.main=1)
#  mtext("Ten random values", side=1, line=1, at=1, cex=0.9, font=3)
#  mtext("Ten other values", line=0.5, at=-1.8, cex=0.9, font=3)
#  mtext(c(-2, 0, 2), side=1, las=1, at=c(-2, 0, 2), line=0.3,
#  col="blue", cex=0.9)
#  mtext(-1:1, side=2, las=1, at=-1:1, line=0.2, col="blue", cex=0.9)

## ----eval=FALSE----------------------------------------------------------
#  library("lattice")
#  n <- seq(5, 45, 5)
#  x <- rnorm(sum(n))
#  y <- factor(rep(n, n), labels=paste("n =", n))
#  densityplot(~ x | y,
#              panel = function(x, ...) {
#                panel.densityplot(x, col="DarkOliveGreen", ...)
#                panel.mathdensity(dmath=dnorm,
#                                  args=list(mean=mean(x), sd=sd(x)),
#                                  col="darkblue")
#                })
#  

## ----eval=FALSE----------------------------------------------------------
#  data(quakes)
#  mini <- min(quakes$depth)
#  maxi <- max(quakes$depth)
#  int <- ceiling((maxi - mini)/9)
#  inf <- seq(mini, maxi, int)
#  quakes$depth.cat <- factor(floor(((quakes$depth - mini) / int)),
#  labels=paste(inf, inf + int, sep="-"))  # add the labels with paste
#  xyplot(lat ~ long | depth.cat, data = quakes) # split according to depth.cat
#  

## ----eval=FALSE----------------------------------------------------------
#  data(iris)
#  xyplot(
#  Petal.Length ~ Petal.Width, data = iris, groups=Species,
#  panel = panel.superpose,
#  type = c("p", "smooth"), span=.75,
#  auto.key = list(x = 0.15, y = 0.85) # legend,relevant location (0,1)
#  )

## ----eval=FALSE----------------------------------------------------------
#  splom(
#  ~iris[1:4], groups = Species, data = iris, xlab = "",
#  panel = panel.superpose,
#  auto.key = list(columns = 3)
#  )

## ----eval=FALSE----------------------------------------------------------
#      M <- 10000  #number of replicates
#      k <- 10     #number of strata
#      r <- M / k  #replicates per stratum
#      N <- 50     #number of times to repeat the estimation
#      T2 <- numeric(k)
#      estimates <- matrix(0, N, 2)
#  
#      g <- function(x) {
#          exp(-x - log(1+x^2)) * (x > 0) * (x < 1)
#          }
#  
#      for (i in 1:N) {
#          estimates[i, 1] <- mean(g(runif(M)))
#          for (j in 1:k)
#              T2[j] <- mean(g(runif(M/k, (j-1)/k, j/k)))
#          estimates[i, 2] <- mean(T2)
#      }
#  #The result of this simulation produces the following estimates.
#      apply(estimates, 2, mean)
#      apply(estimates, 2, var)

## ----eval=FALSE----------------------------------------------------------
#  
#  ### Plot importance functions in Figures 5.1(a) and 5.1.(b)
#  
#      #par(ask = TRUE) #uncomment to pause between graphs
#  
#      x <- seq(0, 1, .01)
#      w <- 2
#      f1 <- exp(-x)
#      f2 <- (1 / pi) / (1 + x^2)
#      f3 <- exp(-x) / (1 - exp(-1))
#      f4 <- 4 / ((1 + x^2) * pi)
#      g <- exp(-x) / (1 + x^2)
#  
#      #for color change lty to col
#  
#      #figure (a)
#      plot(x, g, type = "l", main = "", ylab = "",
#           ylim = c(0,2), lwd = w)
#      lines(x, g/g, lty = 2, lwd = w)
#      lines(x, f1, lty = 3, lwd = w)
#      lines(x, f2, lty = 4, lwd = w)
#      lines(x, f3, lty = 5, lwd = w)
#      lines(x, f4, lty = 6, lwd = w)
#      legend("topright", legend = c("g", 0:4),
#             lty = 1:6, lwd = w, inset = 0.02)
#  
#      #figure (b)
#      plot(x, g, type = "l", main = "", ylab = "",
#          ylim = c(0,3.2), lwd = w, lty = 2)
#      lines(x, g/f1, lty = 3, lwd = w)
#      lines(x, g/f2, lty = 4, lwd = w)
#      lines(x, g/f3, lty = 5, lwd = w)
#      lines(x, g/f4, lty = 6, lwd = w)
#      legend("topright", legend = c(0:4),
#             lty = 2:6, lwd = w, inset = 0.02)
#  

## ----eval=FALSE----------------------------------------------------------
#   n <- 20
#      alpha <- .05
#      x <- rnorm(n, mean=0, sd=2)
#      UCL <- (n-1) * var(x) / qchisq(alpha, df=n-1)
#      print(UCL)

## ----echo = FALSE,eval=FALSE---------------------------------------------
#  x<- 0:4
#  prob <- c(0.1,0.2,0.2,0.2,0.3)
#  x.prob <- rbind(x,prob)
#  x.prob

## ----eval=FALSE----------------------------------------------------------
#  x<- 0:4
#  prob <- c(0.1,0.2,0.2,0.2,0.3)
#      cp <- cumsum(prob); m <- 1000; r <- numeric(m)
#      r <- x[findInterval(runif(m),cp)+1]
#      ct <- as.vector(table(r)); ctfre<-ct/sum(ct);  ctdif<-ct/sum(ct)/prob
#  # generate a random sample of size 1000
#  n <- 1000
#  a <- sample(x,size = n,prob = prob ,replace = TRUE)
#  # compute the frequency
#  a <- as.vector(table(a))
#  a.fre <- a/n/prob
#  diffe1 <- a.fre
#  diffe2 <- ctdif
#  data <- data.frame(x,ct,a,ctfre,a.fre,prob,diffe1,diffe2)
#  colnames(data) <- c("x","ct-num","sample-num","ct-freq","samp-freq","prob","dif1","dif2")
#  data

## ----eval=FALSE----------------------------------------------------------
#  # write the function
#  rBeta <- function(n,a,b){
#    k <- 0; j <- 0
#    y <- numeric(n)
#    while (k < n) {
#      u <- runif(1)
#      j <- j + 1
#      x <- runif(1) #random variate from g
#      c <- prod(2:(a+b-1))/prod(2:(a-1))/prod(2:(b-1))
#      r <- x^(a-1)*(1-x)^(b-1)
#      if (r > u) {
#        #we accept x
#        k <- k + 1
#        y[k] <- x
#      }
#      }
#    y
#    }
#  
#  # Generate a random sample of size 1000 from the Beta(3,2) distribution.
#  a <- 3; b <- 2
#  c <- 12
#  t <- rBeta(1000,3,2)       #sample data
#  t
#  hist(t, prob = TRUE, main = bquote(f(x)==12*x^2*(1-x))) #density histogram of sample)
#  
#  y <- seq(0, 1, .01)
#  lines(y,c*y^(a-1)*(1-y)^(b-1)) # density curve

## ----eval=FALSE----------------------------------------------------------
#      #generate a Exponential-Gamma mixture
#      n <- 1000
#      r <- 4
#      beta <- 2
#      lambda <- rgamma(n, r, beta) #lambda is random
#  
#      #now supply the sample of lambda's as the exponential parameter
#      x <- rexp(n, lambda)        #the mixture
#      x

## ----eval=FALSE----------------------------------------------------------
#  MCbeta <- function(a,b,n,x){
#    cdf <- numeric(length(x))
#      for (i in 1 :length(x) ){
#        u <- runif(n,0,x[i])
#        g <- x[i] * factorial(a+b-1)/factorial(a-1)/factorial(b-1)* u ^(a-1) *(1-u) ^(b-1)
#        cdf[i] <- mean(g)   #estimated value
#      }
#      Phi <- pbeta(x,a,b)   #beta(a,b)
#      ratio <- cdf/Phi
#      print(round(rbind(x, cdf, Phi, ratio), 3))
#  }
#  
#  
#  x <- seq(.1,0.9, length=9)
#  MCbeta(3,3,1000,x)

## ----eval=FALSE----------------------------------------------------------
#  Antith.sampling <- function(sigma, n, antithetic = TRUE){
#    u <- runif(n/2)
#    if (antithetic){
#      v <- 1 - u
#      a <- sqrt(-2*sigma^2*log(1-u))
#      b <- sqrt(-2*sigma^2*log(1-v))
#      variance <- (var(a)+var(b)+2*cov(a,b))/4 # c(a,b) is the whole sample, n is the sample size)
#    }
#    else{
#      v <- runif(n/2)
#      u <- c(u, v)
#      a <- sqrt(-2*sigma^2*log(1-u)) # a is the whole sample, n is the sample size
#      variance <-var(a)
#    }
#  variance
#  }
#  
#  var1 <- Antith.sampling(2,1000,antithetic = FALSE) # variance of independent random variables
#  var2 <- Antith.sampling(2,1000,antithetic = TRUE) # var of antithentic variables
#  ratio <- 100 * (var1-var2)/var1   #the percent of reduction
#  print(c(var1,var2,ratio))
#  

## ----eval=FALSE----------------------------------------------------------
#  x <- seq(1,10,0.02)
#  y <- x^2/sqrt(2*pi)* exp((-x^2/2))
#  y1 <- exp(-x)
#  y2 <- 1 / (x^2)
#  
#  gs <- c(expression(g(x)==e^{-x^2/2}*x^2/sqrt(2*pi)),expression(f[1](x)==1/(x^2)),expression(f[2](x)==x*e^{(1-x^2)/4}/sqrt(2*pi)))
#  par(mfrow=c(1,2))
#      #figure (a)
#      plot(x, y, type = "l", ylab = "", ylim = c(0,0.5),main='density function')
#      lines(x, y1, lty = 2,col="red")
#      lines(x, y2, lty = 3,col="blue")
#      legend("topright", legend = 0:2,
#             lty = 1:3,inset = 0.02,col=c("black","red","blue"))
#  
#      #figure (b)
#     plot(x, y/y1,  type = "l",ylim = c(0,2),ylab = "",lty = 2, col="red",main = 'ratios')
#      lines(x, y/y2, lty = 3,col="blue")
#      legend("topright", legend = 1:2,
#             lty = 2:3, inset = 0.02,col=c("red","blue"))
#  

## ----eval=FALSE----------------------------------------------------------
#  g <- function(x) {x^2*exp(-x^2/2)/sqrt(2*pi)*(x>1)}
#  m <- 10000
#  theta.hat <- se <- numeric(2)
#  
#  x <- rexp(m, 1)   #using f1
#  fg <- g(x) / exp(-x)
#  theta.hat[1] <- mean(fg)
#  se[1] <- sd(fg)
#  
#  u <- runif(m)
#  x <- 1/(1-u) #using f2
#  fg <- g(x)*x^2
#  theta.hat[2] <- mean(fg)
#  se[2] <- sd(fg)
#  rbind(theta.hat,se)

## ----eval=FALSE----------------------------------------------------------
#  g <- function(x) {x^2*exp(-x^2/2)/sqrt(2*pi)*(x>1)}
#  m <- 1e4
#  u <- runif(m)
#  x <- 1/(1-u) #using f2
#  fg <- g(x)*x^2
#  theta.hat <- mean(fg)
#  print(theta.hat)
#  theta <- integrate(g,1,Inf)
#  theta

## ----eval=FALSE----------------------------------------------------------
#  
#  m <- 1000
#  n <- 1000
#  Gini <- function(m,n){
#    G_hat1<-G_hat2<-G_hat3 <- numeric(m)
#  
#    for (j in 1:m) {
#      x1 <- rlnorm(n)
#      x2 <- runif(n)
#      x3 <- rbinom(n,size=1,prob=0.1)
#      u1 <- mean(x1);u2 <- mean(x2);u3 <- mean(x3)
#      x1sort <- sort(x1);x2sort <- sort(x2);x3sort <- sort(x3)
#  
#      g1<-g2<-g3 <- numeric(m)
#      for (i in 1:m){
#        g1[i] <- (2*i-m-1)*x1sort[i]
#        g2[i] <- (2*i-m-1)*x2sort[i]
#        g3[i] <- (2*i-m-1)*x3sort[i]
#      }
#      G_hat1[j] <- sum(g1)/m^2/u1
#      G_hat2[j] <- sum(g2)/m^2/u2
#      G_hat3[j] <- sum(g3)/m^2/u3
#    }
#  
#    print(c(mean(G_hat1),median(G_hat1)))
#    print(quantile(G_hat1,seq(0,1,0.1)))
#    hist(G_hat1,prob=TRUE,main = "X from lognorm distribution")
#  
#    print(c(mean(G_hat2),median(G_hat2)))
#    print(quantile(G_hat2,seq(0,1,0.1)))
#    hist(G_hat2,prob=TRUE,main = "X from uniform distribution")
#  
#    print(c(mean(G_hat3),median(G_hat3)))
#    print(quantile(G_hat3,seq(0,1,0.1)))
#    hist(G_hat3,prob=TRUE,main = "X from Bionomial(0.1) distribution")
#  }
#  
#  Gini(m,n)
#  

## ----eval=FALSE----------------------------------------------------------
#  GCI <- function(m,n,a,b,alpha){
#    G <- numeric(m)
#    for (i in 1:m) {
#      # function to calculate Gini with Normal distribution.
#      Gi <- function(n,a,b){
#        y <- numeric(n)
#        y <- sort(rnorm(n)) # x=exp(y) ~ln(0,1)
#        mu <- exp(a+b^2/2) # mu of X
#        Gini <- 1/n^2/mu * sum((2*c(1:n)-n-1)*exp(y))
#        return(Gini)
#        }
#      G[i]<-Gi(n,a,b)
#      }
#    LCL <- quantile(G,alpha/2)
#    UCL <- quantile(G,1-alpha/2)
#    CI <- c(LCL,UCL)
#    count <-sum(G <= UCL)-sum(G < LCL)
#    coverage <- count/n #coverage rate
#    return(c(CI,coverage))
#    }
#  GCI(1000,1000,0,1,0.1)

## ----eval=FALSE----------------------------------------------------------
#  library(MASS)
#  powers <- function(N,alpha,mu,sigma){
#    p1<-p2<-p3<-numeric(N)
#    for(i in 1:N){
#    bvn<- mvrnorm(N, mu = mu, Sigma = sigma ) # mvrnorm function, independent
#    x<-bvn[,1];y<-bvn[,2]
#    p1[i]<-cor.test(x,y,method="spearman")$p.value
#    p2[i]<-cor.test(x,y,method="kendall")$p.value
#    p3[i]<-cor.test(x,y,method="pearson")$p.value
#    }
#  power<-c(mean(p1<=alpha),mean(p2<=alpha),mean(p3<=alpha))
#  return(power)
#  }
#  set.seed(123)
#  N <- 500 # Number of random samples
#  alpha<-0.05
#  # Target parameters for univariate normal distributions
#  rho <- 0
#  mu1 <- 0; s1 <- 1
#  mu2 <- 1; s2 <- 4
#  # Parameters for bivariate normal distribution
#  mu <- c(mu1,mu2) # Mean
#  sigma <- matrix(c(s1^2, s1*s2*rho, s1*s2*rho, s2^2),2,2) # Covariance matrix
#  powers(N,alpha,mu,sigma)

## ----eval=FALSE----------------------------------------------------------
#  library(MASS)
#  powers <- function(N,alpha,mu,sigma){
#    p1<-p2<-p3<-numeric(N)
#    for(i in 1:N){
#    bvn<- mvrnorm(N, mu = mu, Sigma = sigma ) # mvrnorm function, independent
#    x<-bvn[,1];y<-bvn[,2]
#    p1[i]<-cor.test(x,y,method="spearman")$p.value
#    p2[i]<-cor.test(x,y,method="kendall")$p.value
#    p3[i]<-cor.test(x,y,method="pearson")$p.value
#    }
#  power<-c(mean(p1<=alpha),mean(p2<=alpha),mean(p3<=alpha))
#  print(round(power),3)
#  }
#  N <- 500 # Number of random samples
#  alpha<-0.05
#  # Target parameters for univariate normal distributions
#  rho <- .6
#  mu1 <- 0; s1 <- 1
#  mu2 <- 1; s2 <- 4
#  # Parameters for bivariate normal distribution
#  mu <- c(mu1,mu2) # Mean
#  sigma <- matrix(c(s1^2, s1*s2*rho, s1*s2*rho, s2^2),2,2) # Covariance matrix
#  powers(N,alpha,mu,sigma)

## ----eval=FALSE----------------------------------------------------------
#  
#  #The answer of first question:
#  m <- 1000
#  n <- 20
#  g <- numeric(m)
#  medians  <- means <- numeric(3)
#  y <- gini1 <- gini2 <- gini3 <- numeric(m)
#  
#  for (i in 1:m) {
#    x <- sort(rlnorm(n))
#    xmean <- mean(x)
#    for (j in 1:n) {
#    y[j] <- (2*j-n-1)*x[j]
#  }
#    gini1[i] <- 1/n^2/xmean*sum(y[1:n])
#  }
#  
#  for (i in 1:m) {
#    x <- sort(runif(n))
#    xmean <- mean(x)
#    for (j in 1:n) {
#    y[j] <- (2*j-n-1)*x[j]
#  }
#    gini2[i] <- 1/n^2/xmean*sum(y[1:n])
#  }
#  
#  for (i in 1:m) {
#    x <- sort(rbinom(n,c(0,1),c(0.1,0.9)))
#    xmean <- mean(x)
#    for (j in 1:n) {
#    y[j] <- (2*j-n-1)*x[j]
#  }
#    gini3[i] <- 1/n^2/xmean*sum(y[1:n])
#  }
#  
#  par(mfrow=c(3,1))
#  par(pin=c(2,1))
#  hist(gini1,prob = TRUE)
#  lines(density(gini1),col = "red",lwd = 2)
#  hist(gini2,prob = TRUE)
#  lines(density(gini2),col = "blue",lwd = 2)
#  hist(gini3,prob = TRUE)
#  lines(density(gini3),col = "green",lwd = 2)
#  
#  medians[1] <- median(gini1)
#  medians[2] <- median(gini2)
#  medians[3] <- median(gini3)
#  medians
#  
#  quantiles1 <- as.vector(quantile(gini1,seq(0.1,0.9,0.1)))
#  quantiles1
#  
#  quantiles2 <- as.vector(quantile(gini2,seq(0.1,0.9,0.1)))
#  quantiles2
#  
#  quantiles3 <- as.vector(quantile(gini3,seq(0.1,0.9,0.1)))
#  quantiles3
#  
#  means[1] <- mean(gini1)
#  means[2] <- mean(gini2)
#  means[3] <- mean(gini3)
#  means
#  
#  
#  #The answer of second question:
#  m <- 1000
#  n <- 20
#  gini <- numeric(m)
#  
#  #Get A series of gini ratios genarating from a lognormal distribution
#  for (i in 1:m) {
#    x <- sort(rlnorm(n))
#    xmean <- mean(x)
#    for (j in 1:n) {
#    y[j] <- (2*j-n-1)*x[j]
#  }
#    gini[i] <- 1/n^2/xmean*sum(y[1:n])
#  }
#  
#  #get the lower confidence interval
#  LCI<- mean(gini)-sd(gini)*qt(0.975,m-1)
#  #get the upper confidence interval
#  UCI <- mean(gini)+sd(gini)*qt(0.975,m-1)
#  #get the confidence interval
#  CI <- c(LCI,UCI)
#  print(CI)
#  #calculate the converage rte
#  covrate<-sum(I(gini>CI[1]&gini<CI[2]))/m
#  print(covrate)
#  
#  
#  #The answer of third question:
#  #We need load the MASS package
#  library(MASS)
#  mean <- c(0, 0)
#  sigma <- matrix( c(25,5,
#                      5, 25),nrow=2, ncol=2)
#  m <- 1000
#  
#  #Calculate the power using pearson correlation test by setting the parameter method as pearson
#  pearvalues <- replicate(m, expr = {
#      mydata1 <- mvrnorm(50, mean, sigma)
#      x <- mydata1[,1]
#      y <- mydata1[,2]
#      peartest <- cor.test(x,y,alternative = "two.sided", method = "pearson")
#      peartest$p.value
#  } )
#  power1 <- mean(pearvalues <= 0.05)
#  power1
#  
#  #Calculate the power using spearman correlation test by setting the parameter method as spearman
#  spearvalues <- replicate(m, expr = {
#      mydata2 <- mvrnorm(50, mean, sigma)
#      x <- mydata2[,1]
#      y <- mydata2[,2]
#      speartest <- cor.test(x,y,alternative = "two.sided", method = "spearman")
#      speartest$p.value
#  } )
#  power2 <- mean(spearvalues <= 0.05)
#  power2
#  
#  #Calculate the power using kendall correlation test by setting the parameter method as kendall
#  kenvalues <- replicate(m, expr = {
#      mydata3 <- mvrnorm(50, mean, sigma)
#      x <- mydata3[,1]
#      y <- mydata3[,2]
#      kentest <- cor.test(x,y,alternative = "two.sided", method = "kendall")
#      kentest$p.value
#  } )
#  power3 <- mean(kenvalues <= 0.05)
#  power3
#  
#  R <- 1000
#  m <- 1000
#  
#  #Calculate the power using pearson correlation test by setting the parameter method as pearson
#  pearvalues <- replicate(m, expr = {
#      u <- runif(R/2,0,10)
#      v <- sin(u)
#      peartest <- cor.test(u,v,alternative = "two.sided", method = "pearson")
#      peartest$p.value
#  } )
#  power1 <- mean(pearvalues <= 0.05)
#  power1
#  
#  #Calculate the power using spearman correlation test by setting the parameter method as spearman
#  spearvalues <- replicate(m, expr = {
#      u <- runif(R/2,0,10)
#      v <- sin(u)
#      speartest <- cor.test(u,v,alternative = "two.sided", method = "spearman")
#      speartest$p.value
#  } )
#  power2 <- mean(spearvalues <= 0.05)
#  power2
#  
#  #Calculate the power using kendall correlation test by setting the parameter method as kendall
#  kenvalues <- replicate(m, expr = {
#      u <- runif(R/2,0,10)
#      v <- sin(u)
#      kentest <- cor.test(u,v,alternative = "two.sided", method = "kendall")
#      kentest$p.value
#  } )
#  power3 <- mean(kenvalues <= 0.05)
#  power3
#  

## ---- echo=FALSE,eval=FALSE----------------------------------------------
#  set.seed(1)
#  library(bootstrap)    #for the law data
#  a<-matrix(c(round(law$LSAT,digits = 0),law$GPA),nrow=2,byrow = TRUE )
#  dimnames(a)<-list(c("LSAT","GPA"),1:15)
#  knitr::kable(a)

## ----eval=FALSE----------------------------------------------------------
#  library(bootstrap)
#  x <- law$LSAT; y<-law$GPA
#  cor <- cor(x,y)
#  n <- length(x)
#  cor_jack <- numeric(n)  #storage of the resamples
#  
#  for (i in 1:n)
#    cor_jack[i] <- cor(x[-i],y[-i])
#  
#  bias.jack <- (n-1)*(mean(cor_jack)-cor)
#  
#  se.jack <- sqrt((n-1)/n*sum((cor_jack-mean(cor_jack)))^2)
#  print(list(cor= cor ,est=cor-bias.jack, bias = bias.jack,se = se.jack, cv = bias.jack/se.jack))

## ----eval=FALSE----------------------------------------------------------
#  #Bootstrap
#  library(boot)
#  data(aircondit,package = "boot")
#  air <- aircondit$hours
#  theta.hat <- mean(air)
#  #set up the bootstrap
#  B <- 2000            #number of replicates
#  n <- length(air)      #sample size
#  theta.b <- numeric(B)     #storage for replicates
#  
#  #bootstrap estimate of standard error of R
#  for (b in 1:B) {
#    #randomly select the indices
#    i <- sample(1:n, size = n, replace = TRUE)
#    dat <- air[i]       #i is a vector of indices
#    theta.b[b] <- mean(dat)
#  }
#  
#  bias.theta <- mean(theta.b - theta.hat)
#  se <- sd(theta.b)
#  
#  print(list(bias.b = bias.theta,se.b = se))
#  
#  theta.boot <- function(dat,ind) {
#    #function to compute the statistic
#    mean(dat[ind])
#  }
#  boot.obj <- boot(air, statistic = theta.boot, R = 2000)
#  print(boot.obj)

## ----eval=FALSE----------------------------------------------------------
#  print(boot.ci(boot.obj, type=c("basic","norm","perc","bca")))

## ----eval=FALSE----------------------------------------------------------
#  #Jackknife
#  #compute the jackknife replicates, leave-one-out estimates
#  library(bootstrap)
#  data(scor,package = "bootstrap")
#  n <- length(scor)
#  theta.jack <- numeric(n)
#  dat <- cbind(scor$mec, scor$vec, scor$alg, scor$ana, scor$sta)
#  sigma.hat <- cov(dat)
#  theta.hat <- eigen(sigma.hat)$values[1]/sum(eigen(sigma.hat)$values)
#  for (i in 1:n){
#    sigma.jack <- cov(dat[-i,])
#    theta.jack[i] <- eigen(sigma.jack)$values[1]/sum(eigen(sigma.jack)$values)
#  }
#  
#  #jackknife estimate of bias
#  bias.jack <- (n - 1) * (mean(theta.jack) - theta.hat)
#  
#  #Jackknife estimate of standard error
#  se.j <- sqrt((n-1) * mean((theta.jack - mean(theta.jack))^2))
#  
#  print(list(bias.jack = bias.jack,se.jack = se.j))

## ----eval=FALSE----------------------------------------------------------
#      ptm <- proc.time()
#  
#      library(DAAG); attach(ironslag)
#      n <- length(magnetic)   #in DAAG ironslag
#      e1 <- e2 <- e3 <- e4 <- matrix(0,n,n)
#  #    yhat1 <- yhat2 <- yhat3 <- yhat4 <- matrix(0,n,n)
#   #   logyhat3 <- logyhat4 <- matrix(0,n,n)
#  
#  
#      # for n-fold cross validation
#      # fit models on leave-two-out samples
#      for (i in 1:n) {
#        for (j in 1:n){
#          if (j != i){
#            y <- magnetic[c(-i,-j)]
#            x <- chemical[c(-i,-j)]
#  
#            J1 <- lm(y ~ x)
#            yhat1 <- J1$coef[1] + J1$coef[2] * chemical[j]
#            e1[i,j] <- magnetic[j] - yhat1
#  
#            J2 <- lm(y ~ x + I(x^2))
#            yhat2 <- J2$coef[1] + J2$coef[2] * chemical[j] +
#              J2$coef[3] * chemical[j]^2
#            e2[i,j] <- magnetic[j] - yhat2
#  
#            J3 <- lm(log(y) ~ x)
#            logyhat3 <- J3$coef[1] + J3$coef[2] * chemical[j]
#            yhat3 <- exp(logyhat3)
#            e3[i,j] <- magnetic[j] - yhat3
#  
#            J4 <- lm(log(y) ~ log(x))
#            logyhat4 <- J4$coef[1] + J4$coef[2] * log(chemical[j])
#            yhat4 <- exp(logyhat4)
#            e4[i,j] <- magnetic[j]- yhat4
#          }
#        }
#      }
#  
#  
#  ltocv <- c(sum(e1^2), sum(e2^2), sum(e3^2),sum(e4^2))/(n*(n-1))
#  ltocv.ptm <- proc.time() - ptm # same function with system.time(exp)
#  print(list("timeconsuming_of_ltocv"=ltocv.ptm[1:3]))
#  ltocv

## ---- echo=FALSE, eval=FALSE---------------------------------------------
#  ### Example 7.18 (Model selection: Cross validation)
#  
#      # Example 7.17, cont.
#      ptm <- proc.time()
#      n <- length(magnetic)   #in DAAG ironslag
#      e1 <- e2 <- e3 <- e4 <- numeric(n)
#  
#      # for n-fold cross validation
#      # fit models on leave-one-out samples
#      for (k in 1:n) {
#          y <- magnetic[-k]
#          x <- chemical[-k]
#  
#          J1 <- lm(y ~ x)
#          yhat1 <- J1$coef[1] + J1$coef[2] * chemical[k]
#          e1[k] <- magnetic[k] - yhat1
#  
#          J2 <- lm(y ~ x + I(x^2))
#          yhat2 <- J2$coef[1] + J2$coef[2] * chemical[k] +
#                  J2$coef[3] * chemical[k]^2
#          e2[k] <- magnetic[k] - yhat2
#  
#          J3 <- lm(log(y) ~ x)
#          logyhat3 <- J3$coef[1] + J3$coef[2] * chemical[k]
#          yhat3 <- exp(logyhat3)
#          e3[k] <- magnetic[k] - yhat3
#  
#          J4 <- lm(log(y) ~ log(x))
#          logyhat4 <- J4$coef[1] + J4$coef[2] * log(chemical[k])
#          yhat4 <- exp(logyhat4)
#          e4[k] <- magnetic[k] - yhat4
#      }
#  
#  
#      loocv <- c(mean(e1^2), mean(e2^2), mean(e3^2), mean(e4^2))
#      loocv.ptm <- proc.time() - ptm
#      print(list("timeconsuming_of_loocv"=loocv.ptm[1:3]))

## ----eval=FALSE----------------------------------------------------------
#  a <- data.frame(rbind(loocv,ltocv))
#  row.names(a) <- c("loocv","ltocv")
#  names(a) <- c("L1","L2","L3","L4")
#  knitr::kable(a)

## ------------------------------------------------------------------------
set.seed(1)
attach(chickwts)
x <- sort(as.vector(weight[feed == "soybean"]))
y <- sort(as.vector(weight[feed == "linseed"]))
detach(chickwts)

n <- length(x) #sample1 size
m <- length(y) #sample2 size
z <- c(x, y)          #pooled sample
N <- n + m
h <- numeric(N)
T <- m*n/(m+n)^2
R <- 999              #number of replicates
K <- 1:N
reps <- numeric(R)   #storage for replicates
v.n <- v1.n <- numeric(n); v.m <- v1.m <- numeric(m)
for (i in 1:n) v.n[i] <- ( x[i] - i )**2
for (j in 1:m) v.m[j] <- ( y[j] - j )**2

reps0 <- ( (n * sum(v.n) + m * sum(v.m)) / (m * n * N) ) -  (4 * m * n - 1) / (6 * N) 
#replicates
for (i in 1:R) {
  #generate indices k for the first sample
  k <- sample(K, size = n, replace = FALSE)
  x1 <- z[k]
  y1 <- z[-k]      #complement of x1
  z1 <- c(x1,y1)
  for (i in 1:n) { v1.n[i] <- ( x1[i] - i )**2 }
  for (j in 1:m) { v1.m[j] <- ( y1[j] - j )**2 }
  reps[k] <- ( (n * sum(v1.n) + m * sum(v1.m)) / (m * n * N) ) - (4 * m * n - 1) / (6 * N)
}
p <- mean( c(reps0, reps) >= reps0 )
p

## ----eval=FALSE----------------------------------------------------------
#  library(RANN)
#  library(energy)
#  library(Ball)
#  library(boot)
#  m <- 50; k<-3; p<-2
#  n1 <- n2 <- 15;N = c(n1,n2); R<-999
#  
#  Tn <- function(z, ix, sizes,k) {
#    n1 <- sizes[1]
#    n2 <- sizes[2]
#    n <- n1 + n2
#    z <- z[ix, ]
#    o <- rep(0, NROW(z))
#    z <- as.data.frame(cbind(z, o))
#    NN <- nn2(z, k=k+1)  #uses package RANN
#    block1 <- NN$nn.idx[1:n1,-1]
#    block2 <- NN$nn.idx[-(1:n1),-1]
#    i1 <- sum(block1 < n1 + 0.5)
#    i2 <- sum(block2 > n1 + 0.5)
#    return((i1 + i2) / (k * n))
#  }
#  
#  eqdist.nn <- function(z,sizes,k){
#    boot.obj <- boot(data=z,statistic=Tn,R=R,
#                     sim = "permutation", sizes = sizes,k=k)
#    ts <- c(boot.obj$t0,boot.obj$t)
#    p.value <- mean(ts>=ts[1])
#    list(statistic=ts[1],p.value=p.value)
#  }
#  p.values <- matrix(NA,m,3)
#  
#  #Unequal variances and equal expectations
#  for(i in 1:m){
#    x <- matrix(rnorm(n1*2),ncol=2);
#    y <- matrix(rnorm(n1*2,0,2),ncol=2);
#    z <- rbind(x,y)
#    p.values[i,1] <- eqdist.nn(z,N,k)$p.value  #NN
#    p.values[i,2] <- eqdist.etest(z,sizes=N,R=R)$p.value  #energy method
#    p.values[i,3] <- bd.test(x=x,y=y,R=R,seed=i*12345)$p.value  #ball method
#  }
#  alpha <- 0.05                        #confidence level
#  pow <- colMeans(p.values<alpha)
#  pow

## ----eval=FALSE----------------------------------------------------------
#  #Unequal variances and unequal expectations
#  for(i in 1:m){
#    x <- matrix(rnorm(n1*2),ncol=2);
#    y <- cbind(rnorm(n2,0,2),rnorm(n2,1,2));
#    z <- rbind(x,y)
#    p.values[i,1] <- eqdist.nn(z,N,k)$p.value  #NN
#    p.values[i,2] <- eqdist.etest(z,sizes=N,R=R)$p.value  #energy method
#    p.values[i,3] <- bd.test(x=x,y=y,R=R,seed=i*12345)$p.value  #ball method
#  }
#  alpha <- 0.05                        #confidence level
#  pow <- colMeans(p.values<alpha)
#  pow

## ----eval=FALSE----------------------------------------------------------
#  #Non-normal distributions: t distribution with 1 df (heavy-tailed distribution), bimodel distribution (mixture of two normal distributions)
#  for(i in 1:m){
#    x <- matrix(rt(n1*2,1), ncol = 2)
#    y <- cbind(rnorm(n2),rnorm(n2,1,2))
#    z <- rbind(x,y)
#    p.values[i,1] <- eqdist.nn(z,N,k)$p.value  #NN
#    p.values[i,2] <- eqdist.etest(z,sizes=N,R=R)$p.value  #energy method
#    p.values[i,3] <- bd.test(x=x,y=y,R=R,seed=i*12345)$p.value  #ball method
#  }
#  alpha <- 0.05                        #confidence level
#  pow <- colMeans(p.values<alpha)
#  pow

## ----eval=FALSE----------------------------------------------------------
#  #Unbalanced samples (say, 1 case versus 10 controls)
#  n1 <- 10; n2 <- 100; N = c(n1,n2)
#  for(i in 1:m){
#    x <- matrix(rt(n1*2,1),ncol=2);
#    y <- matrix(rnorm(n2*2),ncol=2);
#    z <- rbind(x,y)
#    p.values[i,1] <- eqdist.nn(z,N,3)$p.value  #NN
#    p.values[i,2] <- eqdist.etest(z,sizes=N,R=R)$p.value  #energy method
#    p.values[i,3] <- bd.test(x=x,y=y,R=R,seed=i*12345)$p.value  #ball method
#  }
#  alpha <- 0.05                        #confidence level
#  pow <- colMeans(p.values<alpha)
#  pow

## ---- eval=FALSE---------------------------------------------------------
#  # Cauchy distribution density function
#  f <- function(x, theta,eta) {
#    stopifnot(theta > 0)
#    return(1/(theta*pi*(1+((x-eta)/theta)^2)))
#    }
#  
#  m <- 10000
#  theta <- 1; eta <- 0
#  x <- numeric(m)
#  x[1] <- rnorm(1) #proposal densty
#  k <- 0
#  u <- runif(m)
#  
#  for (i in 2:m) {
#    xt <- x[i-1]
#    y <- rnorm(1, mean =  xt)
#    num <- f(y, theta, eta) * dnorm(xt, mean = y)
#    den <- f(xt, theta, eta) * dnorm(y, mean = xt)
#    if (u[i] <= num/den) x[i] <- y else {
#      x[i] <- xt
#      k <- k+1     #y is rejected
#    }
#  print(k)
#  }

## ----eval=FALSE----------------------------------------------------------
#  plot(1:m, x, type="l", main="", ylab="x")
#  
#  #discard the burnin sample
#  b <- 1001
#  y <- x[b:m]
#  a <- ppoints(100)
#  Qc <- qcauchy(a)  #quantiles of Standard Cauchy
#  Q <- quantile(x, a)
#  
#  qqplot(Qc, Q, main="", xlim=c(-2,2),ylim=c(-2,2), xlab="Standard Cauchy Quantiles", ylab="Sample Quantiles")
#  
#  hist(y, breaks="scott", main="", xlab="", freq=FALSE)
#  lines(Qc, f(Qc, 1, 0))

## ----eval=FALSE----------------------------------------------------------
#  gs <- c(125,18,20,34)  #group size
#  m <- 5000
#  w <- .25
#  b <- 1001
#  #the target density
#  prob <- function(y, gs) {
#    if (y < 0 | y >1)
#      return (0)
#    else
#      return((1/2+y/4)^gs[1] *((1-y)/4)^gs[2]*((1-y)/4)^gs[3]*(y/4)^gs[4])
#  }
#  
#  u <- runif(m)  #for accept/reject step
#  v <- runif(m, -w, w)  #proposal distribution
#  x[1] <- .25
#  for (i in 2:m) {
#    y <- x[i-1] + v[i]
#    if (u[i] <= prob(y, gs) / prob(x[i-1], gs))
#      x[i] <- y
#    else
#      x[i] <- x[i-1]
#  }
#  theta.hat <- mean(x[b:m])
#  theta.hat

## ----eval=FALSE----------------------------------------------------------
#  # compare
#  gs.hat <- sum(gs) * c((2+theta.hat)/4, (1-theta.hat)/4, (1-theta.hat)/4, theta.hat/4)
#  round(gs.hat)

## ----eval=FALSE----------------------------------------------------------
#  ### Gelman-Rubin method of monitoring convergence
#  
#  Gelman.Rubin <- function(psi) {
#    # psi[i,j] is the statistic psi(X[i,1:j])
#    # for chain in i-th row of X
#    psi <- as.matrix(psi)
#    n <- ncol(psi)
#    k <- nrow(psi)
#  
#    psi.means <- rowMeans(psi)     #row means
#    B <- n * var(psi.means)        #between variance est.
#    psi.w <- apply(psi, 1, "var")  #within variances
#    W <- mean(psi.w)               #within est.
#    v.hat <- W*(n-1)/n + (B/n)     #upper variance est.
#    r.hat <- v.hat / W             #G-R statistic
#    return(r.hat)
#  }
#  
#  gs <- c(125,18,20,34)  #group size
#  #m <- 5000
#  #w <- .25
#  b <- 1001
#  #the target density
#  prob <- function(y, gs) {
#    if (y < 0 | y >1)
#      return (0)
#    else
#      return((1/2+y/4)^gs[1] *((1-y)/4)^gs[2]*((1-y)/4)^gs[3]*(y/4)^gs[4])
#  }
#  
#  chain <- function(w, N) {
#    #generates a Metropolis chain for Normal(0,1)
#    #with Normal(X[t], sigma) proposal distribution
#    #and starting value X1
#    x <- rep(0, N)
#    u <- runif(N)  #for accept/reject step
#    v <- runif(N, -w, w)  #proposal distribution
#    x[1] <- w
#    for (i in 2:N) {
#      y <- x[i-1] + v[i]
#      if (u[i] <= prob(y, gs) / prob(x[i-1], gs))
#        x[i] <- y
#      else
#        x[i] <- x[i-1]
#    }
#    return(x)
#  }
#  
#  k <- 4          #number of chains to generate
#  n <- 15000      #length of chains
#  w <- c(0.05,0.25,0.5,0.7,0.9)
#  
#  #generate the chains
#  X <- matrix(0, nrow=k, ncol=n)
#  for (i in 1:k)
#    X[i, ] <- chain(w[i], n)
#  
#  #compute diagnostic statistics
#  psi <- t(apply(X, 1, cumsum))
#  for (i in 1:nrow(psi))
#    psi[i,] <- psi[i,] / (1:ncol(psi))
#  print(Gelman.Rubin(psi))
#  
#  #plot psi for the four chains
#  par(mfrow=c(2,2))
#  for (i in 1:k)
#    plot(psi[i, (b+1):n], type="l",
#         xlab=i, ylab=bquote(psi))
#  par(mfrow=c(1,1)) #restore default
#  
#  #plot the sequence of R-hat statistics
#  rhat <- rep(0, n)
#  for (j in (b+1):n)
#    rhat[j] <- Gelman.Rubin(psi[,1:j])
#  plot(rhat[(b+1):n], type="l", xlab="",ylim = range(1,1.4), ylab="R")
#  abline(h=1.2, lty=2)
#  abline(h=1.1,lty = 2)

## ----eval=FALSE----------------------------------------------------------
#  k <- c(4 : 25, 100, 500, 1000) #Declare the variables
#  n <- length(k)
#  a <- rep(0,n) #store the intersection points
#  eps <- .Machine$double.eps^0.25
#  for (i in 1:n) {
#    out <- uniroot(function(a){
#      pt(sqrt(a^2*(k[i] -1)/(k[i] - a ^ 2)),df = ( k[i] - 1)) - pt(sqrt(a ^ 2 * k[i] /(k[i] + 1 - a ^ 2)),df = k[i])},
#      lower = eps, upper = sqrt(k[i] - eps) #the endpoints
#    )
#    a[i] <- out$root
#  }
#  intsecp <- rbind(k,a)
#  #knitr::kable(intsecp)
#  print(round(intsecp,3))
#  plot(k[1:19] ,a[1:19], xlab = "k value(df)", ylab = "A(k)")

## ----eval=FALSE----------------------------------------------------------
#  set.seed(1)
#  # generate the distribution function of cauchy distribution
#  cdf <- function(x,eta,theta){
#    n <- length(x)
#    cdf <- numeric(n)
#    # compute the density function of cauchy distribution
#    pdf <- function(y) 1/(theta* pi *(1+((y-eta)/theta)^2))
#    for (i in 1 : n){
#      cdf[i] <- integrate(pdf, -Inf, x[i])$value
#    }
#    return(cdf)
#  }
#  # consider the special case
#  eta <- 2
#  theta <- 3
#  x <- seq(-15 + eta, 15 + eta, .01)
#  plot(x, cdf(x,eta, theta),type = "l",lwd = 3, xlab = "x            eta = 2, theta = 3", ylab = "cdf(x)")
#  lines(x,pcauchy(x,eta,theta),col = 2)
#  legend("topleft", c("cdf","pcauchy"), lwd = c(3,1),col = c(1,2))
#  #compute the difference
#  mean(cdf(x,eta, theta) - pcauchy(x))

## ----echo=FALSE,eval=FALSE-----------------------------------------------
#    dat <- rbind(Genotype=c('AA','BB','OO','AO','BO','AB','Sum'),
#    Frequency=c('p2','q2','r2','2pr','2qr','2pq',1),
#    Count=c('nAA','nBB','nOO','nAO','nBO','nAB','n'))
#    knitr::kable(dat,format="html",table.attr = "class=\"table table-bordered\"", align="c")

## ----eval=FALSE----------------------------------------------------------
#  formulas <- list(
#  mpg ~ disp,
#  mpg ~ I(1 / disp),
#  mpg ~ disp + wt,
#  mpg ~ I(1 / disp) + wt
#  )

## ----eval=FALSE----------------------------------------------------------
#  set.seed(1)
#  ## loop
#  out1 <- vector("list", length(formulas))
#  for (i in seq_along(formulas)) {
#    out1[[i]] <- lm(formulas[[i]],data = mtcars)
#  }
#  
#  ## lapply
#  (out1<-lapply(formulas,function(x) lm(x,mtcars)))

## ----eval=FALSE----------------------------------------------------------
#  bootstraps <- lapply(1:10, function(i) {
#  rows <- sample(1:nrow(mtcars), rep = TRUE)
#  mtcars[rows, ]
#  })

## ----eval=FALSE----------------------------------------------------------
#  ## loop
#  out2 <- vector("list", length(bootstraps))
#  for (i in seq_along(bootstraps)) {
#    out2[[i]] <- lm(mpg ~ disp,bootstraps[[i]])
#  }
#  ## lapply
#  (out2 <- lapply(bootstraps, function(x) lm(mpg ~ disp,x)))

## ----eval=FALSE----------------------------------------------------------
#  rsq <- function(mod) summary(mod)$r.squared

## ----eval=FALSE----------------------------------------------------------
#  set.seed(2)
#  # out is the linear model generated in ex3 and ex4.
#  out <- c(out1,out2)
#  (R_sqr <- sapply(out, rsq))
#  plot(1:14,R_sqr,ylim = c(0,1))

## ----eval=FALSE----------------------------------------------------------
#  trials <- replicate(
#  100,
#  t.test(rpois(10, 10), rpois(7, 10)),
#  simplify = FALSE
#  )

## ----eval=FALSE----------------------------------------------------------
#  ##sapply
#  sapply(trials, function(x) x$p.value)
#  
#  ##get rid of the anonymous function
#  sapply(trials, `[[`, 'p.value')

## ----eval=FALSE----------------------------------------------------------
#  library(parallel)
#  mcvMap <- function(f, FUN.VALUE , ...) {
#      out <- mcMap(f, ...)
#      vapply(out, identity, FUN.VALUE)
#  }

## ------------------------------------------------------------------------
# Testing for statistical independence
## 
chisq_test <- function(x){
  m <- nrow(x);  n <- ncol(x); N <- sum(x)
  E <- matrix(0,m,n)
  rowsums <- unlist(lapply(1:m, function(i) sum(x[i,]))) # which is used to computed pi.
  colsums <- unlist(lapply(1:n, function(j) sum(x[,j])))
  for (i in 1:m){
    for (j in 1:n) {
      E[i,j] <- rowsums[i]*colsums[j]/N
    }
  }
  df <- (m-1) * (n-1)
  chi_sqr <- sum((x-E)^2/E) #
  p_value <- dchisq(chi_sqr, df = df)
  (test <- list(chi_sqr = chi_sqr, df = df, p_value = p_value))
}

  # chisq.test() 
  ## From Agresti(2007) p.39
M <- as.table(rbind(c(762, 327, 468), c(484, 239, 477)))
dimnames(M) <- list(gender = c("F", "M"),
                    party = c("Democrat","Independent", "Republican"))
(M)
(Xsq <- chisq.test(M))  # Prints test summary

chisq_test(M)

print(microbenchmark::microbenchmark(
  chisq_test(M),
  chisq.test(M)
))

## ------------------------------------------------------------------------
table2 <- function(x, y) {
  x_val <- sort(unique(x)); y_val <- sort(unique(y)) # remove the duplicate elements
  m <- length(x_val); n <- length(y_val) # dimensions of the table 
  mat <- matrix(0L, m, n)
  for (i in seq_along(x)) {
    mat[which(x_val == x[[i]]), which(y_val == y[[i]])] <-
      mat[which(x_val == x[[i]]),  which(y_val == y[[i]])] + 1L
  }
  dimnames <- list(x_val, y_val)
  names(dimnames) <- as.character(as.list(match.call())[-1])  # R has names for dimnames... :/
  table <- array(mat, dim = dim(mat), dimnames = dimnames)
  class(table) <- "table"
  table
}
# Example:generate random samples.
data <- data.frame(group=as.factor(sample(1:4,100,TRUE)),
                     race=as.factor(sample(1:3,100,TRUE)))
# look at the table
tab1 <- with(data,table(group,race))
tab2 <- with(data,table2(group,race))
# Check whether the table2 is same with table function
identical(tab1,tab2)
# Use it to speed up your chi-square test for testing 
chisq_test(tab1)
microbenchmark::microbenchmark(chisq_test(tab1), chisq_test(tab2))

