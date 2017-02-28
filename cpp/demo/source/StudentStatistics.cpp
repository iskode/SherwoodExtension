#include "StudentStatistics.h"

#include <iostream>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/digamma.hpp>

#include "DataPointCollection.h"
using namespace boost::math;

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{

  TStudentPdf2d::TStudentPdf2d(double nu, double mu_x, double mu_y, double Sigma_11, double Sigma_12, double Sigma_22)
  {
    nu_ = nu;
    
    if (nu_ <= 0.0)
    	throw std::runtime_error("degree of freedom must be >0.0.");
    
    mean_x_ = mu_x;
    mean_y_ = mu_y;

    Sigma_11_ = Sigma_11;
    Sigma_12_ = Sigma_12;
    Sigma_22_ = Sigma_22;

    det_Sigma_ = Sigma_11 * Sigma_22 - Sigma_12 * Sigma_12;

    if (det_Sigma_ < 0.0)
      throw std::runtime_error("covariance matrix must have determinant>0.0.");

    log_det_Sigma_ = log(det_Sigma_);

    inv_Sigma_11_ = Sigma_22 / det_Sigma_;
    inv_Sigma_22_ = Sigma_11 / det_Sigma_;
    inv_Sigma_12_ = -Sigma_12 / det_Sigma_;
  }

  double TStudentPdf2d::GetProbability(float x, float y) const
  {
    double x_ = x - mean_x_;
    double y_ = y - mean_y_;
    
    return tgamma(nu_/2.0 + 1) * pow(nu_ * 3.141593 * tgamma(nu_/2.0), -1.0) * pow(det_Sigma_, -0.5) * pow(
      1 + (x_ * (inv_Sigma_11_ * x_ + inv_Sigma_12_ * y_) - y_ * (inv_Sigma_12_ * x_ + inv_Sigma_22_ * y_))/nu_,
      -nu_/2.0 - 1);
  }

  double TStudentPdf2d::GetNegativeLogProbability(float x, float y) const
  {
    std::cout <<"Warning: It's the Gaussian formula not Student T" <<std::endl;
    double x_ = x - mean_x_;
    double y_ = y - mean_y_;

    double result = 0.5 * log_det_Sigma_ + 0.5 * (x_ * (inv_Sigma_11_ * x_ + inv_Sigma_12_ * y_) + y_ * (inv_Sigma_12_ * x_ + inv_Sigma_22_ * y_));

    return result;
  }

  double TStudentPdf2d::Entropy() const
  {
    double determinant = Sigma_11_ * Sigma_22_ - Sigma_12_ * Sigma_12_;

    if (determinant <= 0.0)
    {
      // If we used a sensible prior, this wouldn't happen. So the user can test
      // without a prior, we fail gracefully.
      return std::numeric_limits<double>::infinity();
    }
		double nu_d = nu_/2.0 + 1;
    return 0.5 * log(determinant) + log((nu_ * 3.141593 * tgamma(nu_d - 1))/tgamma(nu_d))
    				 											+ nu_d * (digamma(nu_d) - digamma(nu_d - 1));

  }

  TStudentAggregator2d::TStudentAggregator2d(double a, double b)
  {
    assert(a >= 0.0 && b >= 0.0); // Hyperparameters a and b must be greater than or equal to zero.

    sx_ = 0.0; sy_ = 0.0;
    sxx_ = 0.0; syy_ = 0.0;
    sxy_ = 0.0;
    sampleCount_ = 0;

    a_ = a;
    b_ = b;

    // The prior should guarantee non-degeneracy but the caller can
    // deactivate it (by setting hyperparameter a to 0.0). In this event
    // we have to tweak things slightly to ensure non-degenerate covariance matrices.
    if (a_ < 0.001)
      a_ = 0.001;
    if (b_ < 1)
      b_ = 1.0;
  }

  TStudentPdf2d TStudentAggregator2d::GetPdf() const
  {
    // Compute maximum likelihood mean and covariance matrix
    double mx = sx_ / sampleCount_;
    double my = sy_ / sampleCount_;
    double vxx = sxx_ / sampleCount_ - (sx_ * sx_) / (sampleCount_ * sampleCount_);
    double vyy = syy_ / sampleCount_ - (sy_ * sy_) / (sampleCount_ * sampleCount_);
    double vxy = sxy_ / sampleCount_ - (sx_ * sy_) / (sampleCount_ * sampleCount_);

    // Adapt using conjugate prior
    double alpha = sampleCount_/(sampleCount_ + a_);
    vxx = alpha * vxx + (1 - alpha) * b_;
    vyy = alpha * vyy + (1 - alpha) * b_;
    vxy = alpha * vxy;

    return GaussianPdf2d(mx, my, vxx, vxy, vyy);
  }

  // IStatisticsAggregator implementation
  void TStudentAggregator2d::Clear()
  {
    sx_ = 0.0; sy_ = 0.0;
    sxx_ = 0.0; syy_ = 0.0;
    sxy_ = 0.0;
    sampleCount_ = 0;
  }

  void TStudentAggregator2d::Aggregate(const IDataPointCollection& data, unsigned int index)
  {
    const DataPointCollection& concreteData = (const DataPointCollection&)(data);

    sx_ += concreteData.GetDataPoint((int)index)[0];
    sy_ += concreteData.GetDataPoint((int)index)[1];

    sxx_ += pow((double)(concreteData.GetDataPoint((int)index)[0]), 2.0);
    syy_ += pow((double)(concreteData.GetDataPoint((int)index)[1]), 2.0);

    sxy_ += concreteData.GetDataPoint((int)index)[0] * concreteData.GetDataPoint((int)index)[1];

    sampleCount_ += 1;
  }

  void TStudentAggregator2d::Aggregate(const TStudentAggregator2d& aggregator)
  {
    sx_ += aggregator.sx_;
    sy_ += aggregator.sy_;

    sxx_ += aggregator.sxx_;
    syy_ += aggregator.syy_;

    sxy_ += aggregator.sxy_;

    sampleCount_ += aggregator.sampleCount_;
  }

  TStudentAggregator2d TStudentAggregator2d::DeepClone() const
  {
    TStudentAggregator2d result(a_, b_); 

    result.sx_ = sx_;
    result.sy_ = sy_;

    result.sxx_ = sxx_;
    result.syy_ = syy_;

    result.sxy_ = sxy_;

    result.sampleCount_ = sampleCount_;

    result.a_ = a_;
    result.b_ = b_;

    return result;
  }

}}}
