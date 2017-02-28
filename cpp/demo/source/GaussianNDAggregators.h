#pragma once

// This file defines some IStatisticsAggregator implementations used by the
// example code in Classification.h, DensityEstimation.h, etc. Note we
// represent IStatisticsAggregator instances using simple structs so that all
// tree data can be stored contiguously in a linear array.

#include <math.h>

#include <limits>
#include <vector>
#include <Eigen/Dense>
#include "mvtnorm.h"

#include "Sherwood.h"

#include "DataPointCollection.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{

  class GaussianPdfNd
  {
  private:
    Eigen::VectorXd mean_;
    Eigen::MatrixXd Sigma_; // symmetric NxN covariance matrix
    Eigen::MatrixXd inv_Sigma_; // symmetric NxN inverse covariance matrix
    double det_Sigma_;
    double log_det_Sigma_;

  public:
    GaussianPdfNd() { }

    GaussianPdfNd(Eigen::VectorXd& mu, Eigen::MatrixXd& Cov);

    Eigen::VectorXd Mean() const
    {
      return mean_;
    }

    Eigen::MatrixXd Covariance() const
    {
      return Sigma_;
    }

    double GetProbability(Eigen::VectorXd& x) const;

    double GetNegativeLogProbability(Eigen::VectorXd& x) const;

    double GetCdf(Eigen::VectorXd& upper) const; // Cumulative Distribution Function

    double Entropy() const;
    bool operator==(const GaussianPdfNd& g ) const;
  };

   struct GaussianAggregatorNd
  {
  private:
    unsigned int sampleCount_;
    int dim_;
    Eigen::VectorXd sum_;
    Eigen::MatrixXd productSum_;    // sum of products, symmetric
    
    double a_, b_;      // hyperparameters of prior

  public:
    GaussianAggregatorNd()
    {
      GaussianAggregatorNd(0.001, 1.0, 2);//default N = 2;
      Clear();
    }

    GaussianAggregatorNd(double a, double b, int dim);

    GaussianPdfNd GetPdf() const;

    unsigned int SampleCount() const {  return sampleCount_; }

    // IStatisticsAggregator implementation
    void Clear();

    void Aggregate(const IDataPointCollection& data, unsigned int index);

    void Aggregate(const GaussianAggregatorNd& aggregator);

    GaussianAggregatorNd DeepClone() const;
  };

} } }
