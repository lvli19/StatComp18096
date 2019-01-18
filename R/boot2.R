#' @title Non-parametric Bootstrapping
#' @description Estimate the standard error and the bias of an estimator using R
#' @param data he data as a vector, matrix or data frame
#' @param func function to be bootstrapped
#' @param B the number of replicates
#' @return a list of standard error and bias
#' @examples
#' \dontrun{
#' data <- 20 * rbeta(1000,2,3)
#' boot(data = data, func = mean, B = 2000)
#' }
#' @export
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
