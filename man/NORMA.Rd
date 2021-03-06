% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/NORMA.R
\name{NORMA}
\alias{NORMA}
\title{NORMA Optimization}
\usage{
NORMA(x, y, f_0 = 0, beta_0 = 0, lambda = 0, rate = function(t) {     1
  }, kernel = linear_kernel, cost_der = ILF_cost_der,
  cost_name = "ILF_cost_der", gamma = 1, max_iterations = nrow(x),
  stopping_threshold = 0, trace = TRUE, no_beta = TRUE,
  fixed_epsilon = TRUE, ...)
}
\arguments{
\item{x}{\code{matrix} containing training points. Each row must be a point.}

\item{y}{\code{numeric} containing target for training points \eqn{x}.}

\item{f_0}{initial hypothesis.}

\item{beta_0}{initial value for offset \eqn{b}.}

\item{lambda}{NORMA optimization parameter \eqn{lambda}}

\item{rate}{learning rate for NORMA optimization. Must be a function with one argument.}

\item{kernel}{kernel function to use. Must be a function with three arguments such as \code{gaussian_kernel}.
See also \link{linear_kernel}}

\item{cost_der}{Loss function derivative to use. See also \link{ILF_cost_der}. Must be "ILF_cost_der" when
ILF derivative is used.}

\item{cost_name}{\code{character} indicating the symbolic name of \code{cost_der}.}

\item{gamma}{gaussian kernel parameter \eqn{\gamma}.}

\item{max_iterations}{maximum number of NORMA iterations computed.}

\item{stopping_threshold}{value indicating when to stop NORMA optimization. See also 'Details'.}

\item{trace}{\code{boolean} indicating if information messages should be printed (TRUE) or not (FALSE).}

\item{no_beta}{\code{boolean} indicating if an offset \eqn{b} is used (FALSE) or not (TRUE).}

\item{fixed_epsilon}{\code{boolean} indicating if \code{epsilon} should be updated (FALSE) or not (TRUE).}

\item{...}{additional arguments to be passed to the low level functions.}
}
\value{
Returns a \code{list} containing:

\describe{
 \item{alpha}{\code{matrix} representing \eqn{\alpha} parameters of NORMA optimization in each iteration, one per row.}
 \item{beta}{\code{numeric} representing \eqn{\beta} parameter of NORMA optimization in each iteration.}
 \item{n_iterations}{Number of NORMA iterations performed.}
}
}
\description{
Computes general noise SVR based on NORMA optimization.
}
\details{
Optimization will stop when the sum of the differences between all training predicted values of present
iteration versus values from previous iteration does not exceeds \code{stopping_threshold}.
}
\examples{
NORMA(x=matrix(rnorm(10),nrow=10,ncol=1,byrow=TRUE),y=rnorm(10),kernel=function(x,y,gamma=0){x\%*\%y},
cost_der=function(phi,sigma_cuad,mu){return((phi-mu)/sigma_cuad)},cost_name="example",
sigma_cuad=1,mu=0)
}
\author{
Jesus Prada, \email{jesus.prada@estudiante.uam.es}
}
\references{
Link to the scientific paper

Kivinen J., Smola A. J., Williamson R.C.: Online learning with kernels. In: IEEE
transactions on signal processing, vol. 52, pp. 2165-2176, IEEE (2004).

with theoretical background for NORMA optimization is provided below.

\url{http://realm.sics.se/papers/KivSmoWil04(1).pdf}
}

