//#include <cmath>
#include "mvtnorm.h"
#include "GaussianNDAggregators.h"

//NB: for student case, use pmvnorm(...) from lib directly
//intialize parameter and call it.


namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
	class Util
	{
	public:
		static double[] correlationMatrix(Eigen::MatrixXd& cov) 
		{
			Eigen::MatrixXd diag = cov.diagonal().array().pow(-0.5).matrix().asDiagonal();
		}

		
		
	
		/* data */
	};


} } }
double normalCdf()