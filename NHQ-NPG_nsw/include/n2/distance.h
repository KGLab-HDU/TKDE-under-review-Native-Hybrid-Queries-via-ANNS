
#pragma once

#if defined(__GNUC__)
  #define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
  #define PORTABLE_ALIGN32 __declspec(align(32))
#endif

namespace n2 {

class BaseDistance {
    public:
    BaseDistance() {}
    virtual ~BaseDistance() = 0;
    virtual float Evaluate(const float* __restrict pVect1, const float*  __restrict pVect2, size_t qty, float  *  __restrict TmpRes) const = 0;
};

class L2Distance : public BaseDistance {
   public:
   L2Distance() {}
   ~L2Distance() override {}
   float Evaluate(const float* __restrict pVect1, const float*  __restrict pVect2, size_t qty, float  *  __restrict TmpRes) const override;
};

class AngularDistance : public BaseDistance {
   public:
   AngularDistance() {}
   ~AngularDistance() override {}
   float Evaluate(const float* __restrict pVect1, const float*  __restrict pVect2, size_t qty, float  *  __restrict TmpRes) const override;
};

} // namespace n2
