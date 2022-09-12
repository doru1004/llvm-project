// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

int main() {
    float x_array[256];
    float *x = &x_array[0];

    // make x available on the GPU
    #pragma omp target data map(tofrom:x[0:256])
    {
        #pragma omp target data use_device_addr(x)
        {
            x[0] = 2;
        }
    }
    return x[0] == 2;
}

// CHECK-LABEL: @main()
// CHECK: [[X:%.+]] = alloca ptr, align 8
// CHECK: call void @__tgt_target_data_begin_mapper(
// CHECK: [[LOADED_X:%.+]] = load ptr, ptr [[X]], align 8
// CHECK: [[BASE_PTR_GEP:%.+]] = getelementptr inbounds [1 x ptr], ptr [[OFFLOAD_BASE_PTR:%.+]], i32 0, i32 0
// CHECK: store ptr [[LOADED_X]], ptr [[BASE_PTR_GEP]], align 8
// CHECK: [[OFFLOAD_PTR_GEP:%.+]] = getelementptr inbounds [1 x ptr], ptr [[OFFLOAD_PTR:%.+]], i32 0, i32 0
// CHECK: store ptr [[LOADED_X]], ptr [[OFFLOAD_PTR_GEP]], align 8
// CHECK: getelementptr inbounds [1 x ptr], ptr [[OFFLOAD_BASE_PTR]], i32 0, i32 0
// CHECK: getelementptr inbounds [1 x ptr], ptr [[OFFLOAD_PTR]], i32 0, i32 0
// CHECK: call void @__tgt_target_data_begin_mapper(
// CHECK: [[LOADED_DEVICE_X:%.+]] = load ptr, ptr [[BASE_PTR_GEP]], align 8
// CHECK: %arrayidx5 = getelementptr inbounds float, ptr %13, i64 0
// CHECK: store float 2.000000e+00, ptr %arrayidx5, align 4
// CHECK: getelementptr inbounds [1 x ptr], ptr [[OFFLOAD_BASE_PTR]], i32 0, i32 0
// CHECK: getelementptr inbounds [1 x ptr], ptr [[OFFLOAD_PTR]], i32 0, i32 0
// CHECK: call void @__tgt_target_data_end_mapper(
// CHECK: call void @__tgt_target_data_end_mapper(

#endif
