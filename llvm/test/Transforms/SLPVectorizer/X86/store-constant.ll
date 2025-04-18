; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -S -mtriple=x86_64-- -passes=slp-vectorizer -mcpu=x86-64    | FileCheck %s
; RUN: opt < %s -S -mtriple=x86_64-- -passes=slp-vectorizer -mcpu=x86-64-v2 | FileCheck %s
; RUN: opt < %s -S -mtriple=x86_64-- -passes=slp-vectorizer -mcpu=x86-64-v3 | FileCheck %s
; RUN: opt < %s -S -mtriple=x86_64-- -passes=slp-vectorizer -mcpu=x86-64-v4 | FileCheck %s

@arr = global [20 x i64] zeroinitializer, align 16

define void @PR111126() {
; CHECK-LABEL: @PR111126(
; CHECK-NEXT:    store <4 x i64> splat (i64 1), ptr @arr, align 16
; CHECK-NEXT:    store <4 x i64> splat (i64 1), ptr getelementptr inbounds (i8, ptr @arr, i64 32), align 16
; CHECK-NEXT:    store <4 x i64> splat (i64 1), ptr getelementptr inbounds (i8, ptr @arr, i64 64), align 16
; CHECK-NEXT:    store <4 x i64> splat (i64 1), ptr getelementptr inbounds (i8, ptr @arr, i64 96), align 16
; CHECK-NEXT:    store <4 x i64> splat (i64 1), ptr getelementptr inbounds (i8, ptr @arr, i64 128), align 16
; CHECK-NEXT:    ret void
;
  store i64 1, ptr @arr, align 16
  store i64 1, ptr getelementptr inbounds (i8, ptr @arr, i64 8), align 8
  store i64 1, ptr getelementptr inbounds (i8, ptr @arr, i64 16), align 16
  store i64 1, ptr getelementptr inbounds (i8, ptr @arr, i64 24), align 8
  store i64 1, ptr getelementptr inbounds (i8, ptr @arr, i64 32), align 16
  store i64 1, ptr getelementptr inbounds (i8, ptr @arr, i64 40), align 8
  store i64 1, ptr getelementptr inbounds (i8, ptr @arr, i64 48), align 16
  store i64 1, ptr getelementptr inbounds (i8, ptr @arr, i64 56), align 8
  store i64 1, ptr getelementptr inbounds (i8, ptr @arr, i64 64), align 16
  store i64 1, ptr getelementptr inbounds (i8, ptr @arr, i64 72), align 8
  store i64 1, ptr getelementptr inbounds (i8, ptr @arr, i64 80), align 16
  store i64 1, ptr getelementptr inbounds (i8, ptr @arr, i64 88), align 8
  store i64 1, ptr getelementptr inbounds (i8, ptr @arr, i64 96), align 16
  store i64 1, ptr getelementptr inbounds (i8, ptr @arr, i64 104), align 8
  store i64 1, ptr getelementptr inbounds (i8, ptr @arr, i64 112), align 16
  store i64 1, ptr getelementptr inbounds (i8, ptr @arr, i64 120), align 8
  store i64 1, ptr getelementptr inbounds (i8, ptr @arr, i64 128), align 16
  store i64 1, ptr getelementptr inbounds (i8, ptr @arr, i64 136), align 8
  store i64 1, ptr getelementptr inbounds (i8, ptr @arr, i64 144), align 16
  store i64 1, ptr getelementptr inbounds (i8, ptr @arr, i64 152), align 8
  ret void
}
