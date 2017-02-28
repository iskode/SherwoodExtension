#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include "mvtnorm.h"


//Not compil
#include  "../GaussianNDAggregators.cpp"
#include "../DataPointCollection.cpp"

using namespace MicrosoftResearch::Cambridge::Sherwood;


void doAnyThing() {
    std::cout<<"It does anything"<<std::endl;
}

bool areEqual(double x, double y, bool print=false)
{
    if (print)
        std::cout<<"diff = "<< x-y <<std::endl;
    return (x-y) < 1.0e-12;
}

void adjustCovWithPrior(Eigen::MatrixXd* cov, double a, double b, int n_samples) {
    double alpha = n_samples/(n_samples + a);
    cov->noalias() = alpha * (*cov);
    cov->diagonal() += (Eigen::VectorXd::Constant(cov->rows(), (1 - alpha) * b));
}

TEST_CASE( "Gaussian N-Dimension Pdf" ) {
    Eigen::VectorXd mu(2);
    mu << 0, -1;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(2, 2);
    SECTION( "simple constructor call check" ) {
        /*GaussianPdfNd::*/ 
        GaussianPdfNd* g = new GaussianPdfNd(mu, cov);
        Eigen::VectorXd m(2); m << 0, -1;
        REQUIRE( g->Mean() == m );
        REQUIRE( g->Covariance() == Eigen::MatrixXd::Identity(2, 2) );
        REQUIRE( g->GetProbability(mu) == pow(2.0 * 3.141593, -1.0) );
        REQUIRE( g->GetNegativeLogProbability(mu) == 0.0 );
        REQUIRE( g->Entropy() == log(2.0 * 3.141593 * 2.718282) );



    }
    
}

TEST_CASE( "Gaussian N-Dimension Aggregator" ) {
    double a = 0.001, 
           b = 1;
    SECTION( "For N = 1" ) {
        /*GaussianPdfNd::*/ 
        std::pair<float, float> range(0, 10);
        int n_samples = 10;
        std::auto_ptr<DataPointCollection> data1d = DataPointCollection::Generate1dGrid(range, n_samples);
        GaussianAggregatorNd agg(a, b, 1); 
        

        REQUIRE( data1d->GetDataPoint(0)[0] == 0 );
        REQUIRE( agg.SampleCount() == 0 );
        //REQUIRE( g.getPdf() throws exception );
        agg.Aggregate(*data1d, 0);
        GaussianPdfNd pdf = agg.GetPdf();
        REQUIRE( agg.SampleCount() == 1 );
        REQUIRE( areEqual(pdf.Mean()(0), 0.0) );
        REQUIRE( areEqual(pdf.Covariance()(0), a/(1.0 + a)) );

        for (int i = 1; i < n_samples; i++) {
            agg.Aggregate(*data1d, i);
        }
        
        pdf = agg.GetPdf();
        //std::cout<<"Covariance = "<<pdf.Covariance() <<std::endl;
        REQUIRE( agg.SampleCount() == n_samples );
        REQUIRE( areEqual(pdf.Mean()(0), 4.5) );
        REQUIRE( areEqual(pdf.Covariance()(0), 8.25 + a/(1.0 + a)) );


    }

    SECTION( "For N = 2" ) {
        std::pair<float, float> range1(0, 15);
        std::pair<float, float> range2(10, 26);
        std::auto_ptr<DataPointCollection> data2d = DataPointCollection::Generate2dGrid(range1, 3, range2, 4);
        GaussianAggregatorNd agg(a, b, 2); 
        
        for (int i = 0; i < 11; i++){
            // std::cout <<data2d->GetDataPoint(i)[0]<< ", "<<data2d->GetDataPoint(i)[1] << std::endl; 
            agg.Aggregate(*data2d, i);
        }

        REQUIRE( agg.SampleCount() == 11 );
        // From python code: mean = array([  5.4545455 ,  16.54545403], dtype=float32)
        // Covariance.diagonal() = array([ 15.70247841,  18.24793625], dtype=float32)
        // Covariance Matrix = array([[ 15.70247934,  -2.97520661],
        //                            [ -2.97520661,  18.24793388]])

        GaussianPdfNd pdf = agg.GetPdf();
        //std::cout<<"Covariance = "<<pdf.Covariance() <<std::endl;
        REQUIRE( pdf.Covariance() == pdf.Covariance().adjoint() );
        Eigen::VectorXd expectedMean(2); expectedMean<<  4.5454545 ,  15.45454502;
        Eigen::MatrixXd expectedCov(2, 2); expectedCov<< 15.70247934,  -2.97520661,
                                                      -2.97520661,  18.24793388;
        adjustCovWithPrior(&expectedCov, a, b, agg.SampleCount());
        
    //     expectedCov.noalias() = *cov;
    // cov.diagonal() += (Eigen::VectorXd::Constant(sum_.size(), (1 - alpha) * b_));

        REQUIRE( (pdf.Mean() - expectedMean).array().abs().sum() <= 1e-5 );
        REQUIRE( (pdf.Covariance() - expectedCov).array().abs().sum() <= 1e-5 );
        


    }
}
