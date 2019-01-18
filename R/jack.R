#' @title Non-parametric Bootstrapping
#' @description Compute the jackknife estimate of standard error using R
#' @param data he data as a vector
#' @param func function to be bootstrapped
#' @return the standard error to be estimated
#' @examples
#' \dontrun{
#' data <- 20 * rbeta(1000,2,3)
#' jack(data = data, func = mean)
#' }
#' @export
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


