//===- AMDGPUAssignIdxToM0.cpp - Copy VGPR-memory indices to M0 ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Copy the register index of a VGPR "as memory" (address space 13)
/// V_LOAD_IDX / V_STORE_IDX pseudo into M0, where V_MOVREL[SD] reads it when
/// the pseudo is expanded (see SIInstrInfo::expandPostRAPseudo). This runs
/// before register allocation so the index computation can be coalesced
/// directly into the write of M0.
///
/// Only movrel-based subtargets use this path; a subtarget without movrel has
/// no register-indirect VGPR access, so VGPR "as memory" is unsupported there
/// and this pass does nothing.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-assign-idx-to-m0"

static bool assignIdxToM0(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasMovrel())
    return false;

  // Instruction selection flags functions that contain a VGPR "as memory"
  // indexed load/store; skip the scan entirely otherwise.
  if (!MF.getInfo<SIMachineFunctionInfo>()->hasVGPRAsMemoryAccess())
    return false;

  const SIInstrInfo *TII = ST.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!SIInstrInfo::isVGPRIdxLoadStore(MI.getOpcode()))
        continue;

      MachineOperand *IdxOp = TII->getNamedOperand(MI, AMDGPU::OpName::idx);
      if (!IdxOp || !IdxOp->isReg() || IdxOp->getReg() == AMDGPU::M0)
        continue;

      // Drop the implicit-def $m0 that instruction selection added (to pin a
      // divergent access inside its waterfall loop); M0 is written for real
      // below.
      int M0DefIdx = MI.findRegisterDefOperandIdx(AMDGPU::M0, /*TRI=*/nullptr);
      if (M0DefIdx >= 0)
        MI.removeOperand(M0DefIdx);

      // Set M0 from the index register and rewrite the pseudo to read M0.
      BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(AMDGPU::COPY), AMDGPU::M0)
          .add(*IdxOp);
      IdxOp->setReg(AMDGPU::M0);
      IdxOp->setIsKill();
      Changed = true;
    }
  }

  return Changed;
}

namespace {

class AMDGPUAssignIdxToM0Legacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUAssignIdxToM0Legacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;
    return assignIdxToM0(MF);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return "AMDGPU Assign Idx To M0"; }
};

} // end anonymous namespace

PreservedAnalyses
AMDGPUAssignIdxToM0Pass::run(MachineFunction &MF,
                             MachineFunctionAnalysisManager &MFAM) {
  if (!assignIdxToM0(MF))
    return PreservedAnalyses::all();
  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

char AMDGPUAssignIdxToM0Legacy::ID = 0;

char &llvm::AMDGPUAssignIdxToM0ID = AMDGPUAssignIdxToM0Legacy::ID;

INITIALIZE_PASS(AMDGPUAssignIdxToM0Legacy, DEBUG_TYPE,
                "AMDGPU Assign Idx To M0", false, false)
