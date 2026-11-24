// Minimal SiP reference bridge for complete-FFT comparisons.
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <fft_api.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string error;
void check(int status, const char* where) {
  if (status) throw std::runtime_error(std::string(where)+": "+std::to_string(status));
}
struct Plan {
  AsdSip::asdFftHandle handle = nullptr;
  void* workspace = nullptr;
  aclTensor* input = nullptr;
  aclTensor* output = nullptr;
  ~Plan() {
    if (input) aclDestroyTensor(input);
    if (output) aclDestroyTensor(output);
    if (workspace) aclrtFree(workspace);
    if (handle) AsdSip::asdFftDestroy(handle);
  }
};
}
extern "C" const char* sip_error() { return error.c_str(); }
extern "C" void* sip_create(int64_t n, int64_t batch, void* x, void* y) {
  Plan* p = new Plan;
  try {
    check(AsdSip::asdFftCreate(p->handle), "create");
    check(AsdSip::asdFftMakePlan1D(p->handle,n,AsdSip::ASCEND_FFT_C2C,
          AsdSip::ASCEND_FFT_FORWARD,batch,AsdSip::ASCEND_FFT_HORIZONTAL),"plan1d");
    size_t bytes=0;
    check(AsdSip::asdFftGetWorkspaceSize(p->handle,bytes),"workspace size");
    if (bytes) check(aclrtMalloc(&p->workspace,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"malloc");
    check(AsdSip::asdFftSetWorkspace(p->handle,p->workspace),"workspace");
    int64_t shape[]={batch,n}, strides[]={n,1};
    p->input=aclCreateTensor(shape,2,ACL_COMPLEX64,strides,0,ACL_FORMAT_ND,shape,2,x);
    p->output=aclCreateTensor(shape,2,ACL_COMPLEX64,strides,0,ACL_FORMAT_ND,shape,2,y);
    if (!p->input || !p->output) throw std::runtime_error("create tensor");
    return p;
  } catch (const std::exception& e) { error=e.what(); delete p; return nullptr; }
}
extern "C" int sip_run(void* opaque, void* stream) {
  try {
    auto* p=static_cast<Plan*>(opaque);
    check(AsdSip::asdFftSetStream(p->handle,static_cast<aclrtStream>(stream)),"stream");
    check(AsdSip::asdFftExecC2C(p->handle,p->input,p->output),"exec");
    return 0;
  } catch (const std::exception& e) { error=e.what(); return -1; }
}
extern "C" void sip_destroy(void* opaque) { delete static_cast<Plan*>(opaque); }
