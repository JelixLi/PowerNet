#include "math/gpu_gemm.h"

#ifdef GPU

namespace Power {

	void gpu_gemm(Ptr<Float> A,Ptr<Float> B,Ptr<Float> C,Int m,Int n,Int k) {
	  Int qpuNums = numQPUs();

	  Int iNNC = 16;
	  Int ind = index();
	  Int inm = me()*k;

	  Ptr<Float> first_p = A+ind+inm;
	  Ptr<Float> first_q = B+ind;

	  Ptr<Float> p;
	  Ptr<Float> q;

	  Float x;
	  Float y;
	  Float sum;
	  Float ans;

	  For(Int r=me(),r<m,r=r+qpuNums) 
	    For(Int c=0,c<n,c++)
	         p = first_p + ((r-me())*k);
	         q = first_q + (c*k);
	         gather(p);
	         gather(q);
	         sum = 0;
	         For(Int s=0,s<k,s=s+iNNC)
	            gather(p+iNNC);
	            gather(q+iNNC);
	            receive(x);
	            receive(y);
	            sum = sum + x*y;
	            p=p+iNNC;
	            q=q+iNNC;
	         End
	         receive(x);
	         receive(y);
	         ans = sum;
	         For(Int c=0,c<15,c=c+1)
	            sum = rotate(sum,1);
	            ans = ans + sum;
	         End
	         store(ans,C + ind + ((r*n+c)<<4));
	    End 
	  End   
	}


	GemmKernelType Get_GemmKernel() {
		GemmKernelType GemmKernel = compile(gpu_gemm);
		return GemmKernel;
	}
	
	GManager<float> global_gpu_manager;

  	void Init_Kernel(GemmKernelType *Kernel) {
  		global_gpu_manager.set_GemmKernel(Kernel);
  	}

};

#endif 