#' @title Non-parametric Bootstrapping
#' @description Computes an estimate for each leave-one-out sample using R
#' @param data the data as a vector
#' @param func function to be bootstrapped
#' @param B the number of replicates
#' @return the standard error of both bootstrap method and jackknife-after-bootstrap method
#' @examples
#' \dontrun{
#' data(aircondit,package = "boot")
#' air <- aircondit$hours
#' jackafterboot(data = air, func=mean, B = 2000)
#' }
#' @export
jackafterboot <- function(data, func=NULL, B){
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