#include<iostream>
#include<memory>

template<typename input_dtype, typename feature_dtype, typename annotation_dtype>
  class IThresholdOptimizer {
	public:
	IThresholdOptimizer(){}
	virtual bool check_for_early_stop(const annotation_dtype * annotations) {
      return false;
    };

};

template<typename input_dtype, typename feature_dtype>
  class DensityThresholdOptimizer
    : public IThresholdOptimizer<input_dtype, feature_dtype, bool> {
	public:
	DensityThresholdOptimizer(){}

	virtual bool check_for_early_stop(const bool * annotations) {
      return false;
    };
};

int main() {
	IThresholdOptimizer<float, float, bool>* dt = new DensityThresholdOptimizer<float, float>();
	const bool* anot = new bool[3];
	dt->check_for_early_stop(anot);
	return 0;
}


