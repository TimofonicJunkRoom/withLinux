#include "thauxmath.h"

// original code from THTensorMath.c
void THTensor_(myadd)(THTensor *r_, THTensor *t, real value)
{
	THTensor_(resizeAs)(r_, t);
	if (THTensor_(isContiguous)(r_) &&
			THTensor_(isContiguous)(t) &&
			THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
		real *tp = THTensor_(data)(t);
		real *rp = THTensor_(data)(r_);
		long sz = THTensor_(nElement)(t);
		long i;
		#pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
		for (i = 0; i < sz; i++) {
			rp[i] = tp[i] + value;
		}
	} else {
		TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data + value;);
	}
	return;
}
