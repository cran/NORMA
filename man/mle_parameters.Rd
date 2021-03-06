% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/NORMA.R
\name{mle_parameters}
\alias{beta_mle}
\alias{general_gaussian_mle}
\alias{general_laplace_mle}
\alias{mle_parameters}
\alias{moge_mle}
\alias{weibull_mle}
\alias{zero_gaussian_mle}
\alias{zero_laplace_mle}
\title{MLE Parameters}
\usage{
mle_parameters(phi, dist = "nm", ...)

zero_laplace_mle(phi)

general_laplace_mle(phi)

zero_gaussian_mle(phi)

general_gaussian_mle(phi)

beta_mle(phi, m1 = mean(phi, na.rm = T), m2 = mean(phi^2, na.rm = T),
  alpha_0 = (m1 * (m1 - m2))/(m2 - m1^2), beta_0 = (alpha_0 * (1 - m1)/m1))

weibull_mle(phi, k_0 = 1)

moge_mle(phi, lambda_0 = 1, alpha_0 = 1, theta_0 = 1)
}
\arguments{
\item{phi}{a vector with residual values used to estimate the parameters.}

\item{dist}{assumed distribution for the noise in the data. Possible values to take:
\itemize{
  \item{l: }{Zero-mean Laplace distribution.}
  \item{lm: }{General Laplace distribution.}
  \item{n: }{Zero-mean Gaussian distribution.}
  \item{nm: }{General Gaussian distribution.}
  \item{b: }{Beta distribution.}
  \item{w: }{Weibull distribution.}
  \item{moge: }{MOGE distribution.}
}}

\item{...}{additional arguments to be passed to the low level
functions (see below).}

\item{m1}{first moment of the residuals. Used to compute \code{alpha_0}.}

\item{m2}{second moment of the residuals. Used to compute \code{beta_0}.}

\item{alpha_0}{initial value for Newton-Raphson method for the parameter \eqn{\alpha}.}

\item{beta_0}{initial value for Newton-Raphson method for the parameter \eqn{\beta}.}

\item{k_0}{initial value for Newton-Raphson method for the parameter \eqn{\kappa}.}

\item{lambda_0}{initial value for Newton-Raphson method for the parameter \eqn{\lambda}.}

\item{theta_0}{initial value for Newton-Raphson method for the parameter \eqn{\theta}.

See also 'Details' and \link{multiroot}.}
}
\value{
\code{mle_parameters} returns a list with the estimated parameters. Depending on the distribution
these parameters will be one or more of the following ones:

\describe{
 \item{sigma}{scale parameter of the Laplace distribution.}
 \item{mu}{location or mean parameter of the Laplace or Gaussian distribution,
respectively.}
 \item{sigma_cuad}{variance parameter of the Gaussian distribution.}
 \item{alpha}{shape1 parameter of the Beta distribution or second parameter of the MOGE distribution.}
 \item{beta}{shape2 parameter of the Beta distribution.}
 \item{k}{shape parameter of the Weibull distribution.}
 \item{lambda}{lambda scale parameter of the Weibull distribution or first parameter of the MOGE distribution.}
 \item{theta}{third parameter of the MOGE distribution.}
}
}
\description{
\code{mle_parameters} computes the optimal parameters via MLE of
a given distribution.

\code{zero_laplace_mle} computes the optimal parameters via MLE
assuming a zero-mean Laplace as noise distribution.

\code{general_laplace_mle} computes the optimal parameters via MLE 
assuming a general Laplace as noise distribution.

\code{zero_gaussian_mle} computes the optimal parameters via MLE 
assuming a zero-mean Gaussian as noise distribution.

\code{general_gaussian_mle} computes the optimal parameters via MLE 
assuming a general Gaussian as noise distribution.

\code{beta_mle} computes the optimal parameters via MLE 
assuming a Beta as noise distribution.

\code{weibull_mle} computes the optimal parameters via MLE 
assuming a Weibull as noise distribution.

\code{moge_mle} computes the optimal parameters via MLE 
assuming a MOGE as noise distribution.
}
\details{
For the zero-\eqn{\mu} Laplace distribution the optimal MLE parameters are
\deqn{\sigma=mean(|\phi_i|)}, where \eqn{{\phi_i}} are the residuals passed as argument.

For the general Laplace distribution the optimal MLE parameters are
\deqn{\mu=median(\phi_i)}\deqn{\sigma=mean(|\phi_i - \mu|)}, where \eqn{{\phi_i}} are the residuals passed as argument.

For the zero-\eqn{\mu} Gaussian distribution the optimal MLE parameters are
\deqn{\sigma^2=mean(\phi_i^2)}, where \eqn{{\phi_i}} are the residuals passed as argument.

For the general Gaussian distribution the optimal MLE parameters are
\deqn{\mu=mean(\phi_i)}\deqn{\sigma^2=mean((\phi_i-\mu)^2)}, where \eqn{{\phi_i}} are the residuals passed as argument.

For the Beta distribution values of parameters \eqn{\alpha} and
\eqn{\beta} are estimated using Newton-Raphson method.

For the Weibull distribution value of parameter \eqn{\kappa} is estimated using Newton-Raphson method
and then estimated value of \eqn{\lambda} is computed using the following closed form that depends on \eqn{\kappa}:
\deqn{\lambda=mean(\phi_i^kappa)^(1/\kappa)}

For the MOGE distribution values of parameters \eqn{\lambda}, \eqn{\alpha} and
\eqn{\theta} are estimated using Newton-Raphson method.

See also 'References'.
}
\examples{
# Estimate optimal parameters using default distribution ("nm").
mle_parameters(rnorm(100))

# Estimate optimal parameters using "lm" distribution.
mle_parameters(rnorm(100),dist="lm")

# Equivalent to mle_parameters(rnorm(100),dist="l")
zero_laplace_mle(rnorm(100))

# Equivalent to mle_parameters(rnorm(100),dist="lm")
general_laplace_mle(rnorm(100))

# Equivalent to mle_parameters(rnorm(100),dist="n")
zero_gaussian_mle(rnorm(100))

# Equivalent to mle_parameters(rnorm(100),dist="nm")
general_gaussian_mle(rnorm(100))

# Equivalent to mle_parameters(rnorm(100),dist="b")
beta_mle(rnorm(100))

# Equivalent to mle_parameters(rnorm(100),dist="w")
weibull_mle(rnorm(100))

# Equivalent to mle_parameters(rnorm(100),dist="moge")
moge_mle(rnorm(100))
}
\author{
Jesus Prada, \email{jesus.prada@estudiante.uam.es}
}
\references{
Link to the scientific paper

Prada, Jesus, and Jose Ramon Dorronsoro. "SVRs and Uncertainty Estimates in Wind 
Energy Prediction." Advances in Computational Intelligence. Springer International 
Publishing, 2015. 564-577,

with theoretical background for this package is provided below.

\url{http://link.springer.com/chapter/10.1007/978-3-319-19222-2_47}
}

