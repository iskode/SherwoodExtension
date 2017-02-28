#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <utility>
#include <Eigen/Dense>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <chrono>
using namespace boost::math;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Matrix_t; 
typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor> Vector_t; 

void testCovMatrix() {
  Eigen::MatrixXd sample(3,2);
  sample << 1, 2,
  	    5, 3, 
       	    7, 11;
    
  Eigen::MatrixXd centered = sample.rowwise() - sample.colwise().mean();
  Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(sample.rows() - 1);
  std::cout << "Here is the matrix m:\n" << sample << std::endl;
  std::cout << "Here is the centered matrix:\n" << centered << std::endl;
  std::cout << "Here is the covariance matrix:\n" << cov << std::endl;
}

void testCovMatrixUsingPtr() {
  auto sample = std::make_shared<Eigen::MatrixXd> (3,2);
  *sample <<1,  2,
  	    5,  3, 
       	    7, 11;
  double rows = double(sample->rows() - 1);
  std::cout << "Here is the matrix sample:\n" << *sample << std::endl;
  *sample = sample->rowwise() - sample->colwise().mean();
  std::cout << "Here is the sample matrix centered:\n" << *sample << std::endl;
  //std::cout << "Here is the transposed sample matrix centered:\n" << sample->adjoint() << std::endl;
  auto cov = std::make_shared<Eigen::MatrixXd>( (sample->adjoint() * (*sample)) / rows );
  
  std::cout << "Here is the sample covariance matrix:\n" << *cov << std::endl;
  std::cout << "Here is the  covariance determinant:\n" << cov->determinant() << std::endl;
  std::cout << "Here is the  log of the covariance determinant:\n" << logf(cov->determinant()) << std::endl;
  std::cout << "Here is the  logf of 0:\n" << logf(0.f) << std::endl;
}

int testRefparams(const int &a) {
  int c = a + 2;  
  return c;
}

void testIsnan(){
  float nan = logf(float(-1));
  std::cout << "Does log(-1) return nan :\n" << std::isnan(nan) << std::endl;
  float lowest = std::numeric_limits<float>::lowest();
  std::cout << "std::numeric_limits<float>::lowest() = \n" << lowest << std::endl;
}

float data[] = {2, 3, 5, 7, 4, 9 ,10};

void triangularMat(int size)
{
	Eigen::Map<Eigen::VectorXf> v(data, size);
	Eigen::MatrixXf cov = Eigen::MatrixXf(size, size).selfadjointView<Eigen::Lower>();
	cov += v*v.adjoint();
	std::cout << "cov = \n" << cov << std::endl;
	cov.setZero();
	v.setZero();
	std::cout << "after setZeros call, cov = \n" << cov << std::endl;
	std::cout << "after setZeros call, v = \n" << v << std::endl;
}



/**
     the data is:   2   4   8
  		    3   9  27
  		    5  25 125
 		    7  49   0
     Each row is a sample. But data are rearranged as an "inputs" vector below.
     We want to retrieve the initial data.
 
  **/
void convertVectorToMatrix() {
  int data_dim = 3, 
      n_samples = 4;
  std::vector<int> inputs = {2, 3, 5, 7, 4, 9 ,25, 49, 8, 27, 125, 0};
  

  auto matrix = std::make_shared<Matrix_t>
		(Eigen::Map<Matrix_t>(inputs.data(), n_samples, data_dim) );
  //Map<MatrixXd> mat(v.data());
  std::cout << "matrix:\n" << *matrix << std::endl;
  
  std::cout << "matrix transposed:\n" << matrix->adjoint() << std::endl;
  Eigen::VectorXd mean_;

}

void vecMatVecProd() {
  Vector_t vx(2) ;
  std::vector<double> x = {2, 3, 5, 7};
  vx(0) = 2.5; vx(1) = -1.0;
  Vector_t centered = (Eigen::Map<Vector_t> (&x[0], 2)) - vx;
  vx = Eigen::MatrixXd::Identity(2,2) * centered;
  x[1] = 3.33;
  //Matrix_t vecMatProd = centered.adjoint() * Matrix_t::Identity(2,2);
  std::cout << "vx = \n" <<centered.dot(Eigen::MatrixXd::Identity(2,2) * centered)<<std::endl;

}
/*
Eigen::VectorXi copyStdVecToEigenVec(const std::vector<int>& mu, int index){
	const int* ptrIdx = mu.data();
  return Eigen::Map<Eigen::VectorXi> (const(ptrIdx + index), mu.size());
  
}*/

void updateMat(Eigen::MatrixXd cov, Eigen::VectorXd& v, int& size) {
		//std::cout << "personal triangular cov =  \n" << cov << std::endl;
		auto t1 = std::chrono::high_resolution_clock::now();
		for(int i = 0; i < size; i++) 
			for(int j = 0; j <= i; j++) {
				cov(i, j) += v(i)*v(j);
			}
		auto t2 = std::chrono::high_resolution_clock::now();
		auto timeMultFull = std::chrono::duration_cast <std::chrono::microseconds>(t2 - t1).count();
		std::cout << "Time for updateMat: " << timeMultFull << " microseconds" << std::endl;
		//std::cout << "personal triangular cov =  " << cov << std::endl;
}
//Eigen initializes matrix to 0 automatically when doing:
// MartixXd myMatrix(n_rows, n_cols);
void testMatOpsPerf(){
	std::vector<double> inputs = {2, 3, 5, 7, 4, 9 ,10, 11, 23, 25, 52, 8, 6, 1, -2.5, -3.3, 49, 8, 27, 125, 0};
	Eigen::MatrixXd temp;
	Eigen::VectorXd v1;
	//inputs.size()
	for (int size = 3; size < 7 ; size++) {
		std::cout << "\n\nSize = " << size << std::endl;
		v1 = Eigen::Map<Eigen::VectorXd> (inputs.data(), size);
		Eigen::MatrixXd cov(size, size);
	
		auto t1 = std::chrono::high_resolution_clock::now();
	
		cov.selfadjointView<Eigen::Lower>();
		Eigen::MatrixXd temp = (v1 * v1.adjoint()).selfadjointView<Eigen::Lower>();
		cov.noalias() += temp;
	
		auto t2 = std::chrono::high_resolution_clock::now();
		auto timeMultFull = std::chrono::duration_cast <std::chrono::microseconds>(t2 - t1).count();
	
	 	std::cout << "full cov =  " << cov << std::endl;
	 	
	  Eigen::MatrixXd cov1(size, size);
		t1 = std::chrono::high_resolution_clock::now();
		
		cov1.selfadjointView<Eigen::Lower>();
		Eigen::MatrixXd temp1 = (v1 * v1.adjoint()).selfadjointView<Eigen::Lower>();
		cov1.noalias() += temp1;
	
		t2 = std::chrono::high_resolution_clock::now();
		auto timeMultTriang = std::chrono::duration_cast <std::chrono::microseconds>(t2 - t1).count();
		
		//Eigen::MatrixXd cov2(size, size);
		//updateMat(cov2, v1, size);
		std::cout << "Time full matrix: " << timeMultFull << " microseconds" << std::endl;
		std::cout << "triangular cov =  " << cov1 << std::endl;
		std::cout << "Time for triangular Matrix: " << timeMultTriang << " microseconds" << std::endl;
		//std::cout << "Time for personal triangular Matrix: " << timeMultManualTriang << " microseconds" << std::endl;
	}
	
}


int main()
{
  
	
  std::vector<double> inputs = {2, 3, 5, 7, 4, 9 ,25, 49, 8, 27, 125, 0};
  Eigen::Map<Eigen::MatrixXd> mat(inputs.data(), 1, 1); 
  mat.diagonal() -= Eigen::VectorXd::Constant(1, 0.5);
  Eigen::VectorXd v = Eigen::VectorXd::Constant(1, 0.5);
  std::cout << "mat = \n" <<mat<<std::endl;
  std::cout << "v*vT = \n" <<v*v.adjoint()<<std::endl;
	Eigen::MatrixXd m(2, 2);
	m << 4, 2, 3, 4;
	//m.diagonal() = Eigen::VectorXd::Constant(4, 4);
	std::cout << "m = \n" <<m<<std::endl;
  Eigen::MatrixXd diag = m.diagonal().array().pow(-0.5).matrix().asDiagonal();
	std::cout << "diag = \n" <<diag<<std::endl;
  
  /*
  Eigen::Map<Eigen::VectorXi> v(inputs, 5);
  std::cout << "v = \n" <<v<<std::endl;
  */
  
  
 	//int size = 3;
 	//triangularMat(size);
 	
 	//testMatOpsPerf();
}

