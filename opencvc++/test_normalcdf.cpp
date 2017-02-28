#include "normalcdf.cpp"
#include <iostream>


void testCdf2D(double a, double b, double rho, double expected) {
	double res = Cdf2D::M(a, b, rho);
	std::cout<<"\ncdf2d( " << a<<", "<< b <<", "<< rho << " ):"<<std::endl;
 	std::cout<<"result = " << res <<", expected = "<< expected <<"."<<std::endl;
}
int main() {
	double a = 0.0 ,
				 b = 0.0 ,
			 rho = 0.0 ,
	expected = 0.25;

	testCdf2D(a, b, rho, expected);
	a = 0.0 , b = 0.0 , rho = 0.0 , expected = 0.25;
	
//int Ω = 3;


 	std::cout<<"-inf - 2 = " << - std::numeric_limits<double>::infinity() - 2 <<"."<<std::endl;
	std::cout<<"+inf - 2 = " << + std::numeric_limits<double>::infinity() - 2 <<"."<<std::endl;
	std::cout<<"-inf + 2 = " << - std::numeric_limits<double>::infinity() + 2 <<"."<<std::endl;
	std::cout<<"+inf + 2 = " << + std::numeric_limits<double>::infinity() + 2 <<"."<<std::endl;

} 

