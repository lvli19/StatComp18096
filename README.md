# StatComp18096

## Overview

__StatComp18096__ is a simple R package developed to compute the bias and the standard error of an estimator with resampling methods.Three functions are considered, namely, _boot_ (using bootstrap method) and _jack_ and _jackafterboot_. For each function, examples are given.

## Comparing _boot_, _jack_ and _jackafterboot_

The source R code for _boot_ is as follows:
```{r,eval=FALSE}
boot <- function(data,func=NULL, B){
  theta.hat <- func(data)
  #set up the bootstrap
  n <- length(data)      #sample size
  theta.b <- numeric(B)     #storage for replicates
  for (b in 1:B) {
    #randomly select the indices
    i <- sample(1:n, size = n, replace = TRUE)
    dat <- data[i]       #i is a vector of indices
    theta.b[b] <- func(dat)
  }
  #bootstrap estimate of standard error of R
  bias.theta <- mean(theta.b - theta.hat)
  se <- sd(theta.b)
  return(list(bias.b = bias.theta,se.b = se))
}
```

```{r,eval=FALSE}
jack <- function(data,func=NULL){
  theta.hat <- func(data)
  #set up the bootstrap
  #B is the number of replicates
  n <- length(data)      #sample size
  M <- numeric(n)
  for (i in 1:n) { #leave one out
    y <- data[-i]
    M[i] <- func(y)
  }
  Mbar <- mean(M)
  se.jack <- sqrt(((n - 1)/n) * sum((M - Mbar)^2))
  return(se.jack)
}
```

```{r,eval=FALSE}
jackafterboot <- function(data,func=NULL,B){
  n <- length(data)
  theta.b <- numeric(B)
  # set up storage for the sampled indices
  indices <- matrix(0, nrow = B, ncol = n)
  # jackknife-after-bootstrap step 1: run the bootstrap
  for (b in 1:B) {
    i <- sample(1:n, size = n, replace = TRUE)
    y <- data[i]
    theta.b[b] <- func(y)
    #save the indices for the jackknife
    indices[b, ] <- i
  }
  #jackknife-after-bootstrap to est. se(se)
  se.jack <- numeric(n)
  for (i in 1:n) {
    #in i-th replicate omit all samples with x[i]
    keep <- (1:B)[apply(indices, MARGIN = 1,
                        FUN = function(k) {!any(k == i)})]
    se.jack[i] <- sd(theta.b[keep])
  }
  se.boot <- sd(theta.b)
  se.jackafterboot <- sqrt((n-1) * mean((se.jack - mean(se.jack))^2))
  return(list(se.boot = se.boot, se.jackafterboot=se.jackafterboot))
}
```

In order to empirically compare _boot_ , _jack_ and _jackafterboot_, one generates 1,000 repicates of beta(2,3) * 20, and save it in data{StatComp}. To load the file, one simply calls _data(number-data)_. The R code for comparing the three functions is as follows.

```{r,eval=TRUE}
library(StatComp18096)
data = data(number_data)
attach(data)
boot <- boot(data = data, B = 200, func = median)$se.b
jack <- jack(data = data, func = median)
jackafterboot<- jackafterboot(data, B = 200, func = median)
#knitr::kable(boot,jack,jackafterboot)
```

