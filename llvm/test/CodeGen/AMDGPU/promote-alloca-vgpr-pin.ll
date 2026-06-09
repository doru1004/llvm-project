; promote-alloca pins a promoted alloca's vector defs only when the alloca is
; marked with !amdgpu.vgpr.pin (emitted by clang's amdgpu_pin_vgpr attribute):
; each promoted vector *store/load def* is wrapped in the void, side-effecting
; llvm.amdgcn.internal.vgpr.pin marker so the backend keeps it VGPR-resident. The
; promoted value itself is still what downstream code uses; the pin only
; references it. The frozen-poison init value is never pinned (it is dead once a
; real store defines the vector). An unmarked alloca is never pinned.
;
; promoteAllocaToVector runs in both the early "to-vector" pass and the late
; "promote-alloca" pass, so check both.
; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -passes=amdgpu-promote-alloca-to-vector < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -passes=amdgpu-promote-alloca < %s | FileCheck %s

; Both store defs of the marked alloca are pinned (2 total); the frozen-poison
; seed is not.
; CHECK-LABEL: @promote(
; CHECK:        %[[STACK:.*]] = freeze <4 x i32> poison
; CHECK-NOT:    call void @llvm.amdgcn.internal.vgpr.pin.v4i32(<4 x i32> %[[STACK]])
; CHECK:        %[[E0:.*]] = insertelement <4 x i32> %[[STACK]], i32 1, i32 0
; CHECK-NEXT:   call void @llvm.amdgcn.internal.vgpr.pin.v4i32(<4 x i32> %[[E0]])
; CHECK:        %[[E1:.*]] = insertelement <4 x i32> %[[E0]], i32 2, i32 1
; CHECK-NEXT:   call void @llvm.amdgcn.internal.vgpr.pin.v4i32(<4 x i32> %[[E1]])
; CHECK-NOT:    call void @llvm.amdgcn.internal.vgpr.pin.v4i32
define amdgpu_kernel void @promote(ptr addrspace(1) %out) {
entry:
  %stack = alloca [4 x i32], align 16, addrspace(5), !amdgpu.vgpr.pin !0
  %gep0 = getelementptr [4 x i32], ptr addrspace(5) %stack, i32 0, i32 0
  store i32 1, ptr addrspace(5) %gep0
  %gep1 = getelementptr [4 x i32], ptr addrspace(5) %stack, i32 0, i32 1
  store i32 2, ptr addrspace(5) %gep1
  %val = load i32, ptr addrspace(5) %gep0
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; An identical alloca without the metadata is never pinned.
; CHECK-LABEL: @promote_unmarked(
; CHECK-NOT:    @llvm.amdgcn.internal.vgpr.pin
define amdgpu_kernel void @promote_unmarked(ptr addrspace(1) %out) {
entry:
  %stack = alloca [4 x i32], align 16, addrspace(5)
  %gep0 = getelementptr [4 x i32], ptr addrspace(5) %stack, i32 0, i32 0
  store i32 1, ptr addrspace(5) %gep0
  %gep1 = getelementptr [4 x i32], ptr addrspace(5) %stack, i32 0, i32 1
  store i32 2, ptr addrspace(5) %gep1
  %val = load i32, ptr addrspace(5) %gep0
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; A store def inside a loop body feeds the loop phi. Pinning every store def (not
; just the entry-store def) keeps the value pinned across the back-edge: the
; entry-store def and both loop-body defs are pinned (3 total).
; CHECK-LABEL: @promote_loop(
; CHECK:        %[[LSTACK:.*]] = freeze <3 x i64> poison
; CHECK-NOT:    call void @llvm.amdgcn.internal.vgpr.pin.v3i64(<3 x i64> %[[LSTACK]])
; CHECK:        %[[LENTRY:.*]] = insertelement <3 x i64> %[[LSTACK]], i64 43, i32 0
; CHECK-NEXT:   call void @llvm.amdgcn.internal.vgpr.pin.v3i64(<3 x i64> %[[LENTRY]])
; CHECK:      loop:
; CHECK:        %[[LD0:.*]] = insertelement <3 x i64> %{{.*}}, i64 68, i32 0
; CHECK-NEXT:   call void @llvm.amdgcn.internal.vgpr.pin.v3i64(<3 x i64> %[[LD0]])
; CHECK:        %[[LD1:.*]] = insertelement <3 x i64> %[[LD0]], i64 32, i32 0
; CHECK-NEXT:   call void @llvm.amdgcn.internal.vgpr.pin.v3i64(<3 x i64> %[[LD1]])
; CHECK-NOT:    call void @llvm.amdgcn.internal.vgpr.pin.v3i64
define amdgpu_kernel void @promote_loop(i1 %cond) {
entry:
  %stack = alloca [3 x i64], align 4, addrspace(5), !amdgpu.vgpr.pin !0
  store i64 43, ptr addrspace(5) %stack
  br i1 %cond, label %loop, label %end

loop:
  %load.0 = load i64, ptr addrspace(5) %stack
  store i64 68, ptr addrspace(5) %stack
  %load.1 = load i64, ptr addrspace(5) %stack
  store i64 32, ptr addrspace(5) %stack
  %loop.cc = icmp ne i64 %load.0, %load.1
  br i1 %loop.cc, label %loop, label %end

end:
  %reload = load i64, ptr addrspace(5) %stack
  ret void
}

; A full-vector store of a plain argument overwrites the whole alloca, so the
; promoted value is the argument itself (not an instruction) and is left
; unpinned even though the alloca is marked; with the seed not pinned either, no
; pin is emitted at all.
; CHECK-LABEL: @promote_full_overwrite_arg(
; CHECK-NOT:    call void @llvm.amdgcn.internal.vgpr.pin.v4i32
; CHECK:        ret <4 x i32> %v
define <4 x i32> @promote_full_overwrite_arg(<4 x i32> %v) {
entry:
  %stack = alloca [4 x i32], align 16, addrspace(5), !amdgpu.vgpr.pin !0
  store <4 x i32> %v, ptr addrspace(5) %stack
  %reload = load <4 x i32>, ptr addrspace(5) %stack
  ret <4 x i32> %reload
}

; A full-vector store of a constant overwrites the whole alloca, so the promoted
; value folds to a constant (not an instruction). Constants are rematerialized
; rather than spilled, so pinning is pointless and none is emitted even though
; the alloca is marked.
; CHECK-LABEL: @promote_full_overwrite_const(
; CHECK-NOT:    call void @llvm.amdgcn.internal.vgpr.pin.v4i32
; CHECK:        ret <4 x i32> <i32 1, i32 2, i32 3, i32 4>
define <4 x i32> @promote_full_overwrite_const() {
entry:
  %stack = alloca [4 x i32], align 16, addrspace(5), !amdgpu.vgpr.pin !0
  store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr addrspace(5) %stack
  %reload = load <4 x i32>, ptr addrspace(5) %stack
  ret <4 x i32> %reload
}

!0 = !{}
