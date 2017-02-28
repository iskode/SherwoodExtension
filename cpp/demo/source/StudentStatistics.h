#pragma once

// This file defines some IStatisticsAggregator implementations used by the
// example code in Classification.h, DensityEstimation.h, etc. Note we
// represent IStatisticsAggregator instances using simple structs so that all
// tree data can be stored contiguously in a linear array.

#include <math.h>

#include <limits>
#include <vector>

#include "Sherwood.h"

#include "DataPointCollection.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
  class TStudentPdf2d
  {
  private:
    double nu_; //degrees of freedom
    double mean_x_, mean_y_;
    double Sigma_11_, Sigma_12_, Sigma_22_; // symmetric 2x2 covariance matrix
    double inv_Sigma_11_, inv_Sigma_12_, inv_Sigma_22_; // symmetric 2x2 inverse covariance matrix
    double det_Sigma_;
    double log_det_Sigma_;

  public:
    TStudentPdf2d() { }

    TStudentPdf2d(double nu, double mu_x, double mu_y, double Sigma_11, double Sigma_12, double Sigma_22);

    double MeanX() const
    {
      return mean_x_;
    }

    double MeanY() const
    {
      return mean_y_;
    }

    double VarianceX() const
    {
      return Sigma_11_;
    }

    double VarianceY() const
    {
      return Sigma_22_;
    }

    double CovarianceXY() const
    {
      return Sigma_12_;
    }

    double DegreeFreedom() const
    {
      return nu_;
    }

    double GetProbability(float x, float y) const;

    double GetNegativeLogProbability(float x, float y) const;

    double Entropy() const;
  };

  struct TStudentAggregator2d
  {
  private:
    unsigned int sampleCount_;

    double sx_, sy_;    // sum
    double sxx_, syy_;  // sum squares
    double sxy_;        // sum products

    double a_, b_;      // hyperparameters of prior

  public:
    TStudentAggregator2d()
    {
      Clear();
    }

    TStudentAggregator2d(double a, double b);

    TStudentAggregator2d GetPdf() const;

    unsigned int SampleCount() const {  return sampleCount_; }

    // IStatisticsAggregator implementation
    void Clear();

    void Aggregate(const IDataPointCollection& data, unsigned int index);

    void Aggregate(const TStudentAggregator2d& aggregator);

    TStudentAggregator2d DeepClone() const;
  };

} } }
