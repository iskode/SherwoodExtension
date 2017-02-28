#include "GaussianNDAggregators.h"

#include <iostream>

#include "DataPointCollection.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{

  GaussianPdfNd::GaussianPdfNd(Eigen::VectorXd& mu, Eigen::MatrixXd& Cov)
  {
    assert(Cov.cols() == Cov.rows());
    assert(mu.size() == Cov.rows());
    Sigma_ = Cov;
    mean_ = mu; //Eigen::Map<Eigen::VectorXd> (mu.data(), mu.size());

    det_Sigma_ = Sigma_.determinant();
    if (det_Sigma_ < 0.0)
      throw std::runtime_error("Gaussian covariance matrix must have determinant>0.0.");

    log_det_Sigma_ = log(det_Sigma_);
    inv_Sigma_ = Sigma_.inverse();
  }

  double GaussianPdfNd::GetProbability(Eigen::VectorXd& x) const
  {
    assert(x.size() == mean_.size());
    Eigen::VectorXd centered = x - mean_;

    return pow(2.0 * 3.141593, -1.0) * pow(det_Sigma_, -0.5) * 
           exp(-0.5 * (centered.dot(inv_Sigma_ * centered)));
  }

  double GaussianPdfNd::GetNegativeLogProbability(Eigen::VectorXd& x) const
  {
    assert(x.size() == mean_.size());
    Eigen::VectorXd centered = x - mean_;

    return 0.5 * log_det_Sigma_ + 0.5 * centered.dot(inv_Sigma_ * centered);
  }

  double GaussianPdfNd::GetCdf(Eigen::VectorXd& upper) const
  {
     assert(upper.size() == mean_.size());
     Eigen::MatrixXd diag = cov.diagonal().array().pow(-0.5).matrix().asDiagonal();
     Eigen::MatrixXd corr = diag * cov * diag;
     //Copy under diag elst of corr in double[] as specified in mvt.f

  } 


  double GaussianPdfNd::Entropy() const
  {

    if (det_Sigma_ <= 0.0)
    {
      // If we used a sensible prior, this wouldn't happen. So the user can test
      // without a prior, we fail gracefully.
      return std::numeric_limits<double>::infinity();
    }

    return 0.5 * log(pow(2.0 * 3.141593 * 2.718282, mean_.size())  * det_Sigma_);
  }


  bool GaussianPdfNd::operator==( const GaussianPdfNd& g ) const
  {
    return ( mean_ == g.mean_ && Sigma_ == g.Sigma_ ) ;
  }


  GaussianAggregatorNd::GaussianAggregatorNd(double a, double b, int dim)
  {
    assert(a >= 0.0 && b >= 0.0); // Hyperparameters a and b must be greater than or equal to zero.
    assert(dim > 0);
    dim_ = dim;
    sum_ = Eigen::VectorXd::Zero(dim_);
    productSum_ = Eigen::MatrixXd::Zero(dim_, dim_); 
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

  GaussianPdfNd GaussianAggregatorNd::GetPdf() const
  {
    // Compute maximum likelihood mean and covariance matrix
    Eigen::VectorXd mean = sum_ / sampleCount_;
    Eigen::MatrixXd cov = (productSum_ / sampleCount_) - 
                        (sum_ * sum_.adjoint()) / (sampleCount_ * sampleCount_);
    
    // std::cout <<"mean = " << mean << std::endl;
    // std::cout <<"before prior ops, covariance = " << cov << std::endl;
    // Adapt using conjugate prior
    double alpha = sampleCount_/(sampleCount_ + a_);
    cov.noalias() = alpha*cov;
    cov.diagonal() += (Eigen::VectorXd::Constant(sum_.size(), (1 - alpha) * b_));
    // std::cout <<"after prior ops, covariance = " << cov << std::endl;

    return GaussianPdfNd(mean, cov);
  }

  // IStatisticsAggregator implementation
  void GaussianAggregatorNd::Clear()
  {
    sum_.setZero();
    productSum_.setZero();
    sampleCount_ = 0;
  }

  void GaussianAggregatorNd::Aggregate(const IDataPointCollection& data, unsigned int index)
  {
    const DataPointCollection& concreteData = (const DataPointCollection&)(data);

    if (concreteData.Dimensions() != dim_)
      throw std::runtime_error("Data and GaussianAggregatorNd Dimensions don't match.");

    //copy data
    Eigen::VectorXd vdata(dim_);
    for (int i = 0; i < dim_; i++)
      vdata(i) = concreteData.GetDataPoint((int)index)[i];

    sum_.noalias() += vdata;
    productSum_.noalias() += vdata * vdata.adjoint();
    
    sampleCount_ += 1;
  }

  void GaussianAggregatorNd::Aggregate(const GaussianAggregatorNd& aggregator)
  {
    assert(aggregator.sum_.size() == sum_.size());
    sum_ += aggregator.sum_;
    productSum_ += aggregator.productSum_;
    sampleCount_ += aggregator.sampleCount_;
  }

  GaussianAggregatorNd GaussianAggregatorNd::DeepClone() const
  {
    GaussianAggregatorNd result(a_, b_, dim_); 

    result.sum_ = sum_;
    result.productSum_ = productSum_;
    result.sampleCount_ = sampleCount_;

    result.a_ = a_;
    result.b_ = b_;

    return result;
  }

}}}
