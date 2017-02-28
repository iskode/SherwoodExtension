#include <cmath>

#include <stdexcept>
#include <limits>
  
static const double a[5] = {0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429};
static const double gama = 0.2316419;
static const double pi = 3.1415926535897932384626433832795;
static const double s = 1.0 / sqrt(2.0 * pi);
static const double A[4] = { 0.3253030, 0.4211071, 0.1334425, 0.006374323 };
static const double B[4] = { 0.1337764, 0.6243247, 1.3425378, 2.2626645 };


template <typename T> int sign(T val)
  {
    return (T(0) < val) - (val < T(0));
  }

  class Cdf1D
  {
    // Please see:
    // Options, Hull, J. C., "Futures, & Other Derivatives", 5th Edition, Ch 12, pp. 248,
    // Prentice Hall, New Jersey.

  public:
    static double N(double x)
    {
      double abs_x = std::abs(x);
      double k = 1.0 / (1.0 + abs_x * gama);

      double N_ = s * exp(-0.5 * x * x);

      double sum = ((((a[4] * k + a[3]) * k + a[2]) * k + a[1]) * k + a[0]) * k;

      double result = 1.0 - N_ * sum;

      if (x < 0.0)
        result = 1.0 - result;

      return result;
    }
	};

class Cdf2D
  {
  public:
    // Please see:
    // Options, Hull, J. C., "Futures, & Other Derivatives", 5th Edition, Appendix 12C, pp. 266,
    // Prentice Hall, New Jersey;

    static double f(double x, double y, double a_, double b_, double rho)
    {
      double result = a_ * (2 * x - a_) + b_ * (2 * y - b_) + 2 * rho * (x - a_) * (y - b_);
      return exp(result);
    }

    static double N(double x)
    {
      return Cdf1D::N(x);
    }

    static double M(double a, double b, double rho)
    {
      if (a>100.0)
        a = 100.0;
      if (a<-100.0)
        a = -100.0;
      if (b>100.0)
        b = 100.0;
      if (b<-100.0)
        b = -100.0;

      if (a <= 0.0 && b <= 0.0 && rho <= 0.0)
      {
        double a_ = a / sqrt(2.0 * (1.0 - rho * rho));
        double b_ = b / sqrt(2.0 * (1.0 - rho * rho));

        double sum = 0.0;
        for (int i = 0; i < 4; i++)
          for (int j = 0; j < 4; j++)
            sum += A[i] * A[j] * f(B[i], B[j], a_, b_, rho);
        sum = sum * sqrt(1.0 - rho * rho) / pi;
        return sum;
      }
      else if (a * b * rho <= 0.0)
      {
        if (a <= 0.0 && b >= 0.0 && rho >= 0.0)
          return N(a) - M(a, -b, -rho);
        else if (a >= 0.0 && b <= 0.0 && rho >= 0.0)
          return N(b) - M(-a, b, -rho);
        else if (a >= 0.0 && b >= 0.0 && rho <= 0.0)
          return N(a) + N(b) - 1.0 + M(-a, -b, rho);
      }
      else if (a * b * rho >= 0.0)
      {
        double denominator = sqrt(a * a - 2.0 * rho * a * b + b * b);
        double rho1 = ((rho * a - b) * sign(a)) / denominator;
        double rho2 = ((rho * b - a) * sign(b)) / denominator;
        double delta = (1.0 - sign(a) * sign(b)) / 4.0;
        return M(a, 0.0, rho1) + M(b, 0.0, rho2) - delta;
      }
      throw std::runtime_error("Invalid input for computation of bivariate normal CDF.");
    }
	};

