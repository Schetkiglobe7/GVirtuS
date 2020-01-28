/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Written by: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 */

#include <iostream>
#include <cstdio>
#include <string>

#include "CudnnFrontend.h"

using namespace std;

extern "C" size_t CUDNNWINAPI cudnnGetVersion(){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::Execute("cudnnGetVersion"); 
    return CudnnFrontend::GetExitCode();
}

extern "C" const char * CUDNNWINAPI cudnnGetErrorString(cudnnStatus_t status){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<cudnnStatus_t>(status);
    CudnnFrontend::Execute("cudnnGetErrorString");
    return (const char *) CudnnFrontend::GetOutputHostPointer<char *>();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreate(cudnnHandle_t *handle){
    CudnnFrontend::Prepare();
    CudnnFrontend::AddHostPointerForArguments<cudnnHandle_t>(handle);
    CudnnFrontend::Execute("cudnnCreate");
    if(CudnnFrontend::Success())
        *handle = CudnnFrontend::GetOutputVariable<cudnnHandle_t>();
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroy(cudnnHandle_t handle){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::Execute("cudnnDestroy");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)streamId);
    CudnnFrontend::Execute("cudnnSetStream");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetStream(cudnnHandle_t handle, cudaStream_t *streamId){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::Execute("cudnnGetStream");
    if(CudnnFrontend::Success())
        *streamId = (cudaStream_t) CudnnFrontend::GetOutputVariable<long long int>();
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc){
    CudnnFrontend::Prepare();

    CudnnFrontend::Execute("cudnnCreateTensorDescriptor");
    if (CudnnFrontend::Success()){
        *tensorDesc = CudnnFrontend::GetOutputVariable<cudnnTensorDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptor( cudnnTensorDescriptor_t   tensorDesc,
                            cudnnTensorFormat_t format,
                            cudnnDataType_t dataType,
                            int n,
                            int c, int h, int w ) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)tensorDesc);
    CudnnFrontend::AddVariableForArguments<cudnnTensorFormat_t>(format);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
    CudnnFrontend::AddVariableForArguments<int>(n);
    CudnnFrontend::AddVariableForArguments<int>(c);
    CudnnFrontend::AddVariableForArguments<int>(h);
    CudnnFrontend::AddVariableForArguments<int>(w);

    CudnnFrontend::Execute("cudnnSetTensor4dDescriptor");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptorEx( cudnnTensorDescriptor_t tensorDesc,
                              cudnnDataType_t dataType,
                              int n,
                              int c,
                              int h,
                              int w,
                              int nStride,
                              int cStride,
                              int hStride,
                              int wStride ){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)tensorDesc);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
    CudnnFrontend::AddVariableForArguments<int>(n);
    CudnnFrontend::AddVariableForArguments<int>(c);
    CudnnFrontend::AddVariableForArguments<int>(h);
    CudnnFrontend::AddVariableForArguments<int>(w);

    CudnnFrontend::AddVariableForArguments<int>(nStride);
    CudnnFrontend::AddVariableForArguments<int>(cStride);
    CudnnFrontend::AddVariableForArguments<int>(hStride);
    CudnnFrontend::AddVariableForArguments<int>(wStride);

    CudnnFrontend::Execute("SetTensor4dDescriptorEx");
    return CudnnFrontend::GetExitCode();
}

extern "C"  cudnnStatus_t CUDNNWINAPI cudnnGetTensor4dDescriptor( const cudnnTensorDescriptor_t tensorDesc,
                            cudnnDataType_t *dataType,
                            int *n,
                            int *c,
                            int *h,
                            int *w,
                            int *nStride,
                            int *cStride,
                            int *hStride,
                            int *wStride ){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)tensorDesc);
    CudnnFrontend::Execute("cudnnGetTensor4dDescriptor");

    if(CudnnFrontend::Success()){
        *dataType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
        *n = CudnnFrontend::GetOutputVariable<int>();
        *c = CudnnFrontend::GetOutputVariable<int>();
        *h = CudnnFrontend::GetOutputVariable<int>();
        *w = CudnnFrontend::GetOutputVariable<int>();
        *nStride = CudnnFrontend::GetOutputVariable<int>();
        *cStride = CudnnFrontend::GetOutputVariable<int>();
        *hStride = CudnnFrontend::GetOutputVariable<int>();
        *wStride = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptor( cudnnTensorDescriptor_t tensorDesc,
                            cudnnDataType_t dataType,
                            int nbDims,
                            const int *dimA,
                            const int *strideA){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)tensorDesc);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
    CudnnFrontend::AddVariableForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>((int*)dimA);
    CudnnFrontend::AddHostPointerForArguments<int>((int*)strideA);

    CudnnFrontend::Execute("cudnnSetTensorNdDescriptor");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptorEx(cudnnTensorDescriptor_t tensorDesc,
                                                                  cudnnTensorFormat_t format,
                                                                  cudnnDataType_t dataType,
                                                                  int nbDims,
                                                                  const int *dimA){

     CudnnFrontend::Prepare();
    
     CudnnFrontend::AddVariableForArguments<cudnnTensorFormat_t>(format);
     CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
     CudnnFrontend::AddVariableForArguments<int>(nbDims);
     CudnnFrontend::AddVariableForArguments<int>((int*)dimA);
   
      if(CudnnFrontend::Success()){
        tensorDesc = (cudnnTensorDescriptor_t)CudnnFrontend::GetOutputVariable<long long int>();
    }
    return CudnnFrontend::GetExitCode();      
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t tensorDesc,
                            int nbDimsRequested,
                            cudnnDataType_t *dataType,
                            int *nbDims,
                            int *dimA,
                            int *strideA){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)tensorDesc);
    CudnnFrontend::AddVariableForArguments<int>(nbDimsRequested);
    CudnnFrontend::Execute("cudnnGetTensorNdDescriptor");
    if(CudnnFrontend::Success()){
        *dataType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
        *nbDims = CudnnFrontend::GetOutputVariable<int>();
        dimA = CudnnFrontend::GetOutputHostPointer<int>();
        strideA = CudnnFrontend::GetOutputHostPointer<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetTensorSizeInBytes(const cudnnTensorDescriptor_t tensorDesc,
                                                               size_t *size){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)tensorDesc);

    CudnnFrontend::Execute("cudnnGetTensorSizeInBytes");
    if(CudnnFrontend::Success()){
       *size = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int) tensorDesc);
    CudnnFrontend::Execute("cudnnDestroyTensorDescriptor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnInitTransformDest(const cudnnTensorTransformDescriptor_t transformDesc,
                                                            const cudnnTensorDescriptor_t srcDesc,
                                                            cudnnTensorDescriptor_t destDesc,
                                                            size_t *destSizeInBytes){
    CudnnFrontend::Prepare();
   
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)transformDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)srcDesc);

    CudnnFrontend::Execute("cudnnInitTransformDest");
    
    if(CudnnFrontend::Success()){
       destDesc = (cudnnTensorDescriptor_t)GetOutputVariable<long long int>();
       *destSizeInBytes = GetoutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateTensorTransformDescriptor(cudnnTensorTransformDescriptor_t *transformDesc){
    CudnnFrontend::Prepare();
   
    CudnnFrontend::Execute("cudnnCreateTensorTransformDescriptor");
   
    if(CudnnFrontend::Success()){
       *transformDesc = CudnnFrontend::GetOutputVariable<cudnnTensorTransformDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();   
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc,
                                                                       const uint32_t nbDims,
                                                                       const cudnnTensorFormat_t destFormat,
                                                                       const int32_t *padBeforeA,
                                                                       const int32_t *padAfterA,
                                                                       const uint32_t *foldA,
                                                                       const cudnnFoldingDirection_t direction){

   CudnnFronted::Prepare();
   
   CudnnFrontend::AddVariableForArguments<uint32_t>(nbDims);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)destFormat);
   CudnnFrontend::AddVariableForArguments<int32_t>((int32_t*)padBeforeA);
   CudnnFrontend::AddVariableForArguments<int32_t>((int32_t*)padAfterA);
   CudnnFrontend::AddVariableForArguments<uint32_t>((uint32_t*)foldA);
   CudnnFrontend::AddVariableForArguemtns<long long int>((long long int)direction);

   CudnnFrontend::Execute("cudnnSetTensorTransformDescriptor");
   if(CudnnFrontend::Success()){
      transformDesc = CudnnFrontend::GetOutputVariable<cudnnFoldingDirection_t>();
  }
   return CudnnFrontend::GetExitCode(); 
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc,
                                                                       uint32_t nbDimsRequested,
                                                                       cudnnTensorFormat_t *destFormat,
                                                                       int32_t *padBeforeA,
                                                                       int32_t *padAfterA,
                                                                       uint32_t *foldA,
                                                                       cudnnFoldingDirection_t *direction){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)transformDesc);
    CudnnFrontend::AddVariableForArguments<uint32_t>(nbDimsRequested);

    CudnnFrontend::Execute("cudnnGetTensorTransformDescriptor");
    if(CudnnFrontend::Success()){
        *destFormat = CudnnFrontend::GetOutputVariable<cudnnTensorFormat_t>();
        *padBeforeA = CudnnFrontend::GetOutputVariable<int32_t>();
        *padAfterA  = CudnnFrontend::GetOutputVariable<int32_t>();
        *foldA      = CudnnFrontend::GetOutputVariable<uint32_t>();
        *direction  = CudnnFrontend::GetOutputVariable<cudnnFoldingDirection_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int >((long long int)transformDesc);

    cudnnFrontend::Execute("cudnnDestroyTensorTransformDescriptor");
 
    return cudnnFrontend::GetExitCode();   
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnTransformTensor( cudnnHandle_t                  handle,
                                                const void                    *alpha,
                                                const cudnnTensorDescriptor_t  xDesc,
                                                const void                    *x,
                                                const void                    *beta,
                                                const cudnnTensorDescriptor_t  yDesc,
                                                void                          *y ){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddHostPointerForArguments(x);
    CudnnFrontend::AddHostPointerForArguments(beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddHostPointerForArguments(y);

    CudnnFrontend::Execute("cudnnTransformTensor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnTransformTensorEx(cudnnHandle_t handle,
                                                            const cudnnTensorTransformDescriptor_t transDesc,
                                                            const void *alpha,
                                                            const cudnnTensorDescriptor_t srcDesc,
                                                            const void *srcData,
                                                            const void *beta,
                                                            const cudnnTensorDescriptor_t destDesc,
                                                            void *destData){

     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)transDesc);
     CudnnFrontend::AddHostPointerForArguments(alpha);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)srcDesc);
     CudnnFrontend::AddHostPointerForArguments(srcData);
     CudnnFrontend::AddHostPointerForArguments(beta);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)destDesc);
     CudnnFrontend::AddHostPointerForArguments(void);
    
     CudnFrontend::Execute("cudnnTransformTensorEx");
   
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetFoldedConvBackwardDataDescriptors(const cudnnHandle_t handle,
                                                                               const cudnnFilterDescriptor_t filterDesc,
                                                                               const cudnnTensorDescriptor_t diffDesc,
                                                                               const cudnnConvolutionDescriptor_t convDesc,
                                                                               const cudnnTensorDescriptor_t gradDesc,
                                                                               const cudnnTensorFormat_t transformFormat,
                                                                               cudnnFilterDescriptor_t foldedFilterDesc,
                                                                               cudnnTensorDescriptor_t paddedDiffDesc,
                                                                               cudnnConvolutionDescriptor_t foldedConvDesc,
                                                                               cudnnTensorDescriptor_t foldedGradDesc,
                                                                               cudnnTensorTransformDescriptor_t filterFoldTransDesc,
                                                                               cudnnTensorTransformDescriptor_t diffPadTransDesc,
                                                                               cudnnTensorTransformDescriptor_t gradFoldTransDesc,
                                                                               cudnnTensorTransformDescriptor_t gradUnfoldTransDesc){

     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     
     CudnnFrontend::Execute("cudnnGetFoldedConvBackwardDataDescriptors");

     if(CudnnFrontend::Success()){
         filterDesc = CudnnFrontend::GetOutputVariable<cudnnFilterDescriptor_t>();
         diffDesc   = CudnnFrontend::GetOutputVariable<cudnnTensorDescriptor_t>();
         convDesc   = CudnnFrontend::GetOutputVariable<cudnnConvolutionDescriptor_t>();
         gradDesc   = CudnnFrontend::GetOutputVariable<cudnnTensorDescriptor_t>();
         transformFormat = CudnnFrontend::GetOutputVariable<cudnnTensorFormat_t>();
         foldedFilterDesc = CudnnFrontend::GetOutputVariable<cudnnFilterDescriptor_t>();
         paddedDiffDesc   = CudnnFrontend::GetOutputVariable<cudnnTensorDescriptor_t>();
         foldedConvDesc   = CudnnFrontend::GetOutputVariable<cudnnConvolutionDescriptor_t>();
         foldedGradDesc   = CudnnFrontend::GetOutputVariable<cudnnTensorDescriptor_t>();
         filterFoldTransDesc = CudnnFrontend::GetOutputVariable<cudnnTensorTransformDescriptor_t>();
         diffPadTransDesc    = CudnnFrontend::GetOutputVariable<cudnnTensorTransformDescriptor_t>();
         gradFoldTransDesc   = CudnnFrontend::GetOutputVariable<cudnnTensorTransformDescriptor_t>();
         gradUnfoldTransDesc = CudnnFrontend::GetOutputVariable<cudnnTensorTransformDescriptor_t>(); 
      }
      return CudnnFrontend::GetExitCode(); 
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnAddTensor(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       aDesc,
                                const void                         *A,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       cDesc,
                                void                               *C ){
    CudnnFrontend::Prepare();
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)aDesc);
    CudnnFrontend::AddHostPointerForArguments((void*)A);
    CudnnFrontend::AddHostPointerForArguments((void*)beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)cDesc);
    CudnnFrontend::AddHostPointerForArguments((void*)C);

    CudnnFrontend::Execute("cudnnAddTensor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t *opTensorDesc){

    CudnnFrontend::Prepare();

    CudnnFrontend::Execute("cudnnCreateOpTensorDescriptor");
    if(CudnnFrontend::Success()){
        *opTensorDesc = CudnnFrontend::GetOutputVariable<cudnnOpTensorDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}
     
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc,
                                                                cudnnOpTensorOp_t opTensorOp,
                                                                cudnnDataType_t opTensorCompType,
                                                                cudnnNanPropagation_t opTensorNanOpt){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<cudnnOpTensorOp_t>(opTensorOp);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(opTensorCompType);
    CudnnFrontend::AddVariableForArguments<cudnnNanPropagation_t>(opTensorNanOpt);

    CudnnFrontend::Execute("cudnnSetOpTensorDescriptor");
    if(CudnnFrontend::Success()){
        opTensorDesc = Frontend::GetOutputVariable<cudnnOpTensorDescriptor_t>(opTensorDesc);
    }
    return CudnnFrontend::GetExitCode();
}


extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetOpTensorDescriptor(const cudnnOpTensorDescriptor_t opTensorDesc,
                                                                cudnnOpTensorOp_t *opTensorOp,
                                                                cudnnDataType_t *opTensorCompType,
                                                                cudnnNanPropagation_t *opTensorNanOpt){


   CudnnFrontend::Prepare();
  
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)opTensorDesc);

   Cudnnfrontend::Execute("cudnnGetOpTensorDescriptor");
   if(CudnnFrontend::Success()){
      *opTensorOp = CudnnFrontend::GetOutputVariable<cudnnOpTensorOp_t>();
      *opTensorCompType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
      *opTensorNanOpt   = CudnnFrontend::GetOutputVariable<cudnnNanPropagation_t>();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)opTensorDesc);

    CudnnFrontend::Execute("cudnnDestroyOpTensorDescriptor");

    CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnOpTensor(
                                cudnnHandle_t                       handle,
                                const cudnnOpTensorDescriptor_t     opTensorDesc,
                                const void                         *alpha1,
                                const cudnnTensorDescriptor_t       aDesc,
                                const void                         *A,
                                const void                         *alpha2,
                                const cudnnTensorDescriptor_t       bDesc,
                                const void                         *B,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       cDesc,
                                void                               *C ){
    CudnnFrontend::Prepare();
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)opTensorDesc);
    CudnnFrontend::AddHostPointerForArguments(alpha1);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)aDesc);
    CudnnFrontend::AddHostPointerForArguments((void*)A);
    CudnnFrontend::AddHostPointerForArguments((void*)alpha2);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)bDesc);
    CudnnFrontend::AddHostPointerForArguments((void*)B);
    CudnnFrontend::AddHostPointerForArguments((void*)beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)cDesc);
    CudnnFrontend::AddHostPointerForArguments((void*)C);

    CudnnFrontend::Execute("cudnnOpTensor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t *reduceTensorDesc){

    CudnnFrontend::Prepare();
    
    CudnnFrontend::Execute("cudnnCreateReduceTensorDescriptor");
    if(CudnnFrontend::Success()){
        *reduceTensorDesc = CudnnFrontend::GetOutputVariable<cudnnReduceTensorDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                                                    cudnnReduceTensorOp_t reduceTensorOp,
                                                                    cudnnDataType_t reduceTensorCompType,
                                                                    cudnnNanPropagation_t reduceTensorNanOpt,
                                                                    cudnnReduceTensorIndices_t reduceTensorIndices,
                                                                    cudnnIndicesType_t reduceTensorIndicesType){

    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)reduceTensorDesc);
    CudnnFrontend::AddVariableForArguments<cudnnReduceTensorOp_t>(reduceTensorOp);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(reduceTensorCompType);
    CudnnFrontend::AddVariableForArguments<cudnnNanPropagation_t>(reduceTensorNanOpt);
    CudnnFrontend::AddVariableForArguments<cudnnReduceTensorIndices_t>(reduceTensorIndices);
    CudnnFrontend::AddVariableForArguments<cudnnIndicesType_t>(reduceTensorIndicesType);

    CudnnFrontend::Execute("cudnnSetReduceTensorDescriptor");
    if(CudnnFrontend::Success()){
       reduceTensorDesc = CudnnFrontend::GetOutputVariable<cudnnReduceTensorDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetReduceTensorDescriptor(const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                                                    cudnnReduceTensorOp_t *reduceTensorOp,
                                                                    cudnnDataType_t *reduceTensorCompType,
                                                                    cudnnNanPropagation_t *reduceTensorNanOpt,
                                                                    cudnnReduceTensorIndices_t *reduceTensorIndices,
                                                                    cudnnIndicesType_t *reduceTensorIndicesType){

    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)reduceTensorDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)reduceTensorNanOpt);
    
    CudnnFrontend::Execute("cudnnGetReduceTensorDescriptor");
    if(CudnnFrontend::Success()){
        *reduceTensorOp = CudnnFrontend::GetOutputVariable<cudnnReduceTensorOp_t>();
        *reduceTensorCompType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
        *reduceTensorIndices  = CudnnFrontend::GetOutputVariable<cudnnReduceTensorIndices_t>();
        *reduceTensorIndicesType = CudnnFrontend::GetOutputVariable<cudnnIndicesType_t>();
    }
    return cudnnFrontend::GetExitCode();
}

extern "C" cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc){

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)reduceTensorDesc);

    CudnnFrontend::Execute("cudnnDestroyReduceTensorDescriptor");  

    return CudnnFrontend::GetExitcode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetReductionIndicesSize(cudnnHandle_t handle,
                                                                  const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                                                  const cudnnTensorDescriptor_t aDesc,
                                                                  const cudnnTensorDescriptor_t cDesc,
                                                                  size_t *sizeInBytes){

    CudnnFrontend::Prepare();
   
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)reduceTensorDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)aDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)cDesc);
    
   CudnnFrontend::Execute("cudnnGetReductionIndicesSize");
   if(CudnnFrontend::Success()){
       *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetReductionWorkspaceSize(cudnnHandle_t handle,
                                                                    const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                                                    const cudnnTensorDescriptor_t aDesc,
                                                                    const cudnnTensorDescriptor_t cDesc,
                                                                    size_t *sizeInBytes){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)reduceTensorDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)aDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)cDesc);
    
    CudnnFrontend::Execute("cudnnGetReductionWorkspaceSize");
    if(CudnnFrontend::Success()){
        *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnReduceTensor(cudnnHandle_t handle,
                                                       const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                                       void *indices,
                                                       size_t indicesSizeInBytes,
                                                       void *workspace,
                                                       size_t workspaceSizeInBytes,
                                                       const void *alpha,
                                                       const cudnnTensorDescriptor_t aDesc,
                                                       const void *A,
                                                       const void *beta,
                                                       const cudnnTensorDescriptor_t cDesc,
                                                       void *C){


    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)reduceTensorDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)indicesSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(workspace);
    CudnnFrontend::AddVariableForArguments<size_t>(workspaceSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)aDesc);
    CudnnFrontend::AddHostPointerForArguments(A);
    CudnnFrontend::AddHostPointerForArguments(beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)cDesc);
    CudnnFrontend::AddHostPointerForArguments(C); 

    CudnnFrontend::Execute("cudnnReduceTensor");
    if(CudnnFrontend::Success()){
       indices = CudnnFrontend::GetOutputHostPointer();
       C       = CudnnFrontend::GetOutputHostPointer();
    }  
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensor(
                            cudnnHandle_t                 handle,
                            const cudnnTensorDescriptor_t yDesc,
                            void                          *y,
                            const void                    *valuePtr ){
    CudnnFrontend::Prepare();
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddHostPointerForArguments(y);
    CudnnFrontend::AddHostPointerForArguments((void*)valuePtr);

    CudnnFrontend::Execute("cudnnSetTensor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnScaleTensor( cudnnHandle_t                 handle,
                                            const cudnnTensorDescriptor_t yDesc,
                                            void                          *y,
                                            const void                    *alpha){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddHostPointerForArguments(y);
    CudnnFrontend::AddHostPointerForArguments((void*)alpha);

    CudnnFrontend::Execute("cudnnScaleTensor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddHostPointerForArguments<cudnnFilterDescriptor_t>(filterDesc);
    CudnnFrontend::Execute("cudnnCreateFilterDescriptor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetFilter4dDescriptor( cudnnFilterDescriptor_t filterDesc,
                                                    cudnnDataType_t dataType,
                                                    cudnnTensorFormat_t  format,
                                                    int k,
                                                    int c, int h, int w ){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dataType);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)format);

    CudnnFrontend::AddVariableForArguments<int>(k);
    CudnnFrontend::AddVariableForArguments<int>(c);
    CudnnFrontend::AddVariableForArguments<int>(h);
    CudnnFrontend::AddVariableForArguments<int>(w);

    CudnnFrontend::Execute("cudnnSetFilter4dDescriptor");

    return CudnnFrontend::GetExitCode();
}

extern "C"  CUDNNWINAPI cudnnStatus_t cudnnGetFilter4dDescriptor( cudnnFilterDescriptor_t filterDesc,
                                                                                cudnnDataType_t *dataType,
                                                                                cudnnTensorFormat_t  *format,
                                                                                int *k,
                                                                                int *c,
                                                                                int *h,
                                                                                int *w ){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);

    CudnnFrontend::Execute("cudnnGetFilter4dDescriptor");

    if(CudnnFrontend::Success()){
        *dataType = (cudnnDataType_t) CudnnFrontend::GetOutputVariable<long long int>();
        *format = (cudnnTensorFormat_t) CudnnFrontend::GetOutputVariable<long long int>();
        *k = CudnnFrontend::GetOutputVariable<int>();
        *c = CudnnFrontend::GetOutputVariable<int>();
        *h = CudnnFrontend::GetOutputVariable<int>();
        *w = CudnnFrontend::GetOutputVariable<int>();
    }

    return CudnnFrontend::GetExitCode();
}

#if CUDNN_VERSION < 6000
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetFilter4dDescriptor_v3( cudnnFilterDescriptor_t filterDesc,
                                                    cudnnDataType_t dataType,
                                                    int k,
                                                    int c, int h, int w ){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dataType);

    CudnnFrontend::AddVariableForArguments<int>(k);
    CudnnFrontend::AddVariableForArguments<int>(c);
    CudnnFrontend::AddVariableForArguments<int>(h);
    CudnnFrontend::AddVariableForArguments<int>(w);

    CudnnFrontend::Execute("cudnnSetFilter4dDescriptor_v3");

    return CudnnFrontend::GetExitCode();
}

extern "C"  CUDNNWINAPI cudnnStatus_t cudnnGetFilter4dDescriptor_v3( cudnnFilterDescriptor_t filterDesc,
                                                                    cudnnDataType_t *dataType,
                                                                    int *k,
                                                                    int *c,
                                                                    int *h,
                                                                    int *w ){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);

    CudnnFrontend::Execute("cudnnGetFilter4dDescriptor_v3");

    if(CudnnFrontend::Success()){
        *dataType = (cudnnDataType_t) CudnnFrontend::GetOutputVariable<long long int>();
        *k = CudnnFrontend::GetOutputVariable<int>();
        *c = CudnnFrontend::GetOutputVariable<int>();
        *h = CudnnFrontend::GetOutputVariable<int>();
        *w = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetFilter4dDescriptor_v4( cudnnFilterDescriptor_t filterDesc,
                                                    cudnnDataType_t dataType,
                                                    cudnnTensorFormat_t  format,
                                                    int k,
                                                    int c, int h, int w ){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dataType);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)format);

    CudnnFrontend::AddVariableForArguments<int>(k);
    CudnnFrontend::AddVariableForArguments<int>(c);
    CudnnFrontend::AddVariableForArguments<int>(h);
    CudnnFrontend::AddVariableForArguments<int>(w);

    CudnnFrontend::Execute("cudnnSetFilter4dDescriptor_v4");

    return CudnnFrontend::GetExitCode();
}

extern "C"  CUDNNWINAPI cudnnStatus_t cudnnGetFilter4dDescriptor_v4( cudnnFilterDescriptor_t filterDesc,
                                                                                cudnnDataType_t *dataType,
                                                                                cudnnTensorFormat_t  *format,
                                                                                int *k,
                                                                                int *c,
                                                                                int *h,
                                                                                int *w ){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);

    CudnnFrontend::Execute("cudnnGetFilter4dDescriptor_v4");

    if(CudnnFrontend::Success()){
        *dataType = (cudnnDataType_t) CudnnFrontend::GetOutputVariable<long long int>();
        *format = (cudnnTensorFormat_t) CudnnFrontend::GetOutputVariable<long long int>();
        *k = CudnnFrontend::GetOutputVariable<int>();
        *c = CudnnFrontend::GetOutputVariable<int>();
        *h = CudnnFrontend::GetOutputVariable<int>();
        *w = CudnnFrontend::GetOutputVariable<int>();
    }

    return CudnnFrontend::GetExitCode();
}
#endif

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetFilterNdDescriptor( cudnnFilterDescriptor_t filterDesc,
                                                                cudnnDataType_t  dataType,
                                                                cudnnTensorFormat_t  format,
                                                                int nbDims,
                                                                const int* filterDimA){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dataType);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)format);

    CudnnFrontend::AddVariableForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)filterDimA);

    CudnnFrontend::Execute("cudnnSetFilterNdDescriptor");
    if(CudnnFrontend::Success())
        filterDesc = (cudnnFilterDescriptor_t) CudnnFrontend::GetOutputVariable<long long int>();
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetFilterNdDescriptor( const cudnnFilterDescriptor_t wDesc,
                                                                int nbDimsRequested,
                                                                cudnnDataType_t *dataType,
                                                                cudnnTensorFormat_t  *format,
                                                                int *nbDims,
                                                                int *filterDimA ){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
    CudnnFrontend::AddVariableForArguments<int>(nbDimsRequested);
    CudnnFrontend::AddHostPointerForArguments(dataType);

    CudnnFrontend::AddHostPointerForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>(filterDimA);

    CudnnFrontend::Execute("cudnnGetFilterNdDescriptor");
    if(CudnnFrontend::Success()){
        *format = (cudnnTensorFormat_t) CudnnFrontend::GetOutputVariable<long long int>();
    }

    return CudnnFrontend::GetExitCode();
}

#if CUDNN_VERSION < 6000
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetFilterNdDescriptor_v3( cudnnFilterDescriptor_t filterDesc,
                                                                cudnnDataType_t  dataType,
                                                                int nbDims,
                                                                const int* filterDimA){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dataType);

    CudnnFrontend::AddVariableForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)filterDimA);

    CudnnFrontend::Execute("cudnnSetFilterNdDescriptor_v3");
    if(CudnnFrontend::Success())
        filterDesc = (cudnnFilterDescriptor_t) CudnnFrontend::GetOutputVariable<long long int>();
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetFilterNdDescriptor_v3( const cudnnFilterDescriptor_t wDesc,
                                                                int nbDimsRequested,
                                                                cudnnDataType_t *dataType,
                                                                int *nbDims,
                                                                int *filterDimA ){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
    CudnnFrontend::AddVariableForArguments<int>(nbDimsRequested);
    CudnnFrontend::AddHostPointerForArguments(dataType);

    CudnnFrontend::AddHostPointerForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>(filterDimA);

    CudnnFrontend::Execute("cudnnGetFilterNdDescriptor_v3");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetFilterNdDescriptor_v4( cudnnFilterDescriptor_t filterDesc,
                                                                cudnnDataType_t  dataType,
                                                                cudnnTensorFormat_t  format,
                                                                int nbDims,
                                                                const int* filterDimA){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dataType);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)format);

    CudnnFrontend::AddVariableForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)filterDimA);

    CudnnFrontend::Execute("cudnnSetFilterNdDescriptor_v4");
    if(CudnnFrontend::Success())
        filterDesc = (cudnnFilterDescriptor_t) CudnnFrontend::GetOutputVariable<long long int>();
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetFilterNdDescriptor_v4( const cudnnFilterDescriptor_t wDesc,
                                                                int nbDimsRequested,
                                                                cudnnDataType_t *dataType,
                                                                cudnnTensorFormat_t  *format,
                                                                int *nbDims,
                                                                int *filterDimA ){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
    CudnnFrontend::AddVariableForArguments<int>(nbDimsRequested);
    CudnnFrontend::AddHostPointerForArguments(dataType);

    CudnnFrontend::AddHostPointerForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>(filterDimA);

    CudnnFrontend::Execute("cudnnGetFilterNdDescriptor_v4");
    if(CudnnFrontend::Success()){
        *format = (cudnnTensorFormat_t) CudnnFrontend::GetOutputVariable<long long int>();
    }
    return CudnnFrontend::GetExitCode();
}
#endif

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetFilterSizeInBytes(const cudnnFilterDescriptor_t filterDesc, size_t *size){


    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);

    CudnnFrontend::Execute("cudnnGetFilterSizeInBytes");
    if(CudnnFrontend::Success()){
       *size = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int) filterDesc);
    CudnnFrontend::Execute("cudnnDestroyFilterDescriptor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnTransformFilter(cudnnHandle_t handle,
                                                          const cudnnTensorTransformDescriptor_t transDesc,
                                                          const void *alpha,
                                                          const cudnnFilterDescriptor_t srcDesc,
                                                          const void *srcData,
                                                          const void *beta,
                                                          const cudnnFilterDescriptor_t destDesc,
                                                          void *destData){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)transDesc);
    CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)srcDesc);
    CudnnFrontend::AddHostPointerForArguments(srcData);
    CudnnFrontend::AddHostPointerForArguments(beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)destDesc);
    CudnnFrontend::AddHostPointerForArguments(destData);

    CudnnFrontend::Execute("cudnnTransformFilter");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnReorderFilterAndBias(cudnnHandle_t handle,
                                                               const cudnnFilterDescriptor_t filterDesc,
                                                               cudnnReorderType_t reorderType,
                                                               const void *filterData,
                                                               void *reorderedFilterData,
                                                               int reorderBias,
                                                               const void *biasData,
                                                               void *reorderedBiasData){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
   CudnnFrontend::AddVariableForArguments<cudnnReorderType_t>(reorderType);
   CudnnFrontend::AddHostPointerForArguments(filterData);
   CudnnFrontend::AddHostPointerForArguments(reorderedFilterData);
   CudnnFrontend::AddVariableForArguments<int>(reorderBias);
   CudnnFrontend::AddHostPointerForArguments(biasData);
   CudnnFrontend::AddHostPointerForArguments(reorderedBiasData);

   CudnnFrontend::Execute("cudnnReorderFilterAndBias");
   
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc){
    CudnnFrontend::Prepare();

    CudnnFrontend::Execute("cudnnCreateConvolutionDescriptor");
    if(CudnnFrontend::Success())
        *convDesc = CudnnFrontend::GetOutputVariable<cudnnConvolutionDescriptor_t>();
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType){

    CudnnFrontend::Prepare();
   
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<cudnnMathType_t>(mathType);

    CudnnFrontend:::Execute("cudnnSetConvolutionMathType");

    return CudnnFrontend::GetExitCode(); 
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t *mathType){

    CudnnFrontend::Prepare();
   
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);

    CudnnFrontend::Execute("cudnnGetConvolutionMathType");
    if(CudnnFrontend::Success()){
       *mathType = CudnnFrontend::GetOutputVariable<cudnnMathType_t>();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int groupCount){

     CudnnFrontend::Prepare();
     
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
     CudnnFrontend::AddVariableForArguments<int>(groupCount);
     
     CudnnFrontend::Execute("cudnnSetConvolutionGroupCount");
     
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int *groupCount){

     CudnnFrontend::Prepare();
    
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);

     CudnnFrontend::Execute("cudnnGetConvolutionGroupCount");
     if(CudnnFrontend::Success()){
         *groupCount = CudnnFrontend::GetOutputVariable<int>();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc,
                                                                    cudnnReorderType_t reorderType){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<cudnnReorderType_t>(reorderType);

    CudnnFrontend::Execute("cudnnSetConvolutionReorderType");
 
    CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc,
                                                                    cudnnReorderType_t *reorderType){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);

    CudnnFrontend::Execute("cudnnGetConvolutionReorderType");
    if(CudnnFrontend::Success()){
        *reorderType = CudnnFrontend::GetOutputVariable<cudnnReorderType_t>();
    }
    return CudnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI  cudnnSetConvolution2dDescriptor( cudnnConvolutionDescriptor_t convDesc,
                                                                    int pad_h,
                                                                    int pad_w,
                                                                    int u,
                                                                    int v,
                                                                    int upscalex,
                                                                    int upscaley,
                                                                    cudnnConvolutionMode_t mode,
                                                                    cudnnDataType_t computeType){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<int>(pad_h);
    CudnnFrontend::AddVariableForArguments<int>(pad_w);
    CudnnFrontend::AddVariableForArguments<int>(u);
    CudnnFrontend::AddVariableForArguments<int>(v);
    CudnnFrontend::AddVariableForArguments<int>(upscalex);
    CudnnFrontend::AddVariableForArguments<int>(upscaley);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionMode_t>(mode);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(computeType);

    CudnnFrontend::Execute("cudnnSetConvolution2dDescriptor");
    if(CudnnFrontend::Success())
        convDesc = (cudnnConvolutionDescriptor_t)CudnnFrontend::GetOutputVariable<long long int>();
    return CudnnFrontend::GetExitCode();
}
 
extern "C" CUDNNWINAPI cudnnStatus_t cudnnGetConvolution2dDescriptor( const cudnnConvolutionDescriptor_t convDesc,
                                                                        int* pad_h,
                                                                        int* pad_w,
                                                                        int* u,
                                                                        int* v,
                                                                        int* upscalex,
                                                                        int* upscaley,
                                                                        cudnnConvolutionMode_t *mode,
                                                                        cudnnDataType_t* computeType ){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);

    CudnnFrontend::Execute("cudnnGetConvolution2dDescriptor");
    if(CudnnFrontend::Success()){
        *pad_h = CudnnFrontend::GetOutputVariable<int>();
        *pad_w = CudnnFrontend::GetOutputVariable<int>();
        *u = CudnnFrontend::GetOutputVariable<int>();
        *v = CudnnFrontend::GetOutputVariable<int>();
        *upscalex = CudnnFrontend::GetOutputVariable<int>();
        *upscaley = CudnnFrontend::GetOutputVariable<int>();
        *mode = CudnnFrontend::GetOutputVariable<cudnnConvolutionMode_t>();
        *computeType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t cudnnGetConvolution2dForwardOutputDim( const cudnnConvolutionDescriptor_t convDesc,
                                                                const cudnnTensorDescriptor_t inputTensorDesc,
                                                                const cudnnFilterDescriptor_t filterDesc,
                                                                int *n,
                                                                int *c,
                                                                int *h,
                                                                int *w ){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)inputTensorDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);

    CudnnFrontend::Execute("cudnnGetConvolution2dForwardOutputDim");
    if(CudnnFrontend::Success()){
        *n = CudnnFrontend::GetOutputVariable<int>();
        *c = CudnnFrontend::GetOutputVariable<int>();
        *h = CudnnFrontend::GetOutputVariable<int>();
        *w = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                                                                   int arrayLength,
                                                                                   const int *padA,
                                                                                   const int *filterStrideA,
                                                                                   const int *dilationA,
                                                                                   cudnnConvolutionMode_t mode,
                                                                                   cudnnDataType_t computeType){

    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<int>(arrayLength);
    CudnnFrontend::AddVariableForArguments<int>((int*)padA);
    CudnnFrontend::AddVariableForArguments<int>((int*)filterStrideA);
    CudnnFrontend::AddVariableForArguments<int>((int*)dilationA);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionMode_t>(mode);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(computeType);

    CudnnFrontend::Execute("cudnnSetConvolutionNdDescriptor");
    if(CudnnFrontend::Success()){
        convDesc = CudnnFrontend::GetOutputVariable<cudnnConvolutionDescriptor_t>(convDesc);
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdDescriptor(const cudnnConvolutionDescriptor_t convDesc,
                                                                     int arrayLengthRequested,
                                                                     int *arrayLength,
                                                                     int *padA,
                                                                     int *strideA,
                                                                     int *dilationA,
                                                                     cudnnConvolutionMode_t *mode,
                                                                     cudnnDataType_t *computeType){

    CudnnFrontend::Prepare();
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<int>(arrayLengthRequested);
    
    CudnnFrontend::Execute("cudnnGetConvolutionNdDescriptor");
    if(CudnnFrontend::Success()){
        convDesc = CudnnFrontend::GetOutputVariable<cudnnConvolutionDescriptor_t>(convDesc);
        arrayLengthRequested = CudnnFrontend::GetOutputVariable<int>();
        *arrayLength = CudnnFrontend::GetOutputVariable<int>();
        *strideA     = CudnnFrontend::GetOutputVariable<int>();
        *dilationA   = CudnnFrontend::GetOutputVariable<int>();
        *mode        = CudnnFrontend::GetOutputVariable<cudnnConvolutionMode_t>();
        *computeType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc,
                                                                           const cudnnTensorDescriptor_t inputTensorDesc,
                                                                           const cudnnFilterDescriptor_t filterDesc,
                                                                           int nbDims,
                                                                           int *tensorOuputDimA){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)inputTensorDesc);
    Cudnnfrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
    CudnnFrontend::AddVariableForArguments<int>(nbDims);

    CudnnFrontend::Execute("cudnnGetConvolutionNdForwardOutputDim");
    if(CudnnFrontend::Success()){
       *tensorOuputDimA =  CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc){

    CudnnFrontend::Prepare();
   
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);

    CudnnFrontend::Execute("cudnnDestroyConvolutionDescriptor");

    CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t handle, int *count){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int >((long long int)handle);

    CudnnFrontend::Execute("cudnnGetConvolutionForwardAlgorithmMaxCount");
    if(CudnnFrontend::Success()){
        *count = CudnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionForwardAlgorithm(cudnnHandle_t handle,
                                                                          const cudnnTensorDescriptor_t xDesc,
                                                                          const cudnnFilterDescriptor_t wDesc,
                                                                          const cudnnConvolutionDescriptor_t convDesc,
                                                                          const cudnnTensorDescriptor_t yDesc,
                                                                          const int requestedAlgoCount,
                                                                          int *returnedAlgoCount,
                                                                          cudnnConvolutionFwdAlgoPerf_t *perfResults){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
    Cudnnfrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddVariableForArguments<int>(returnedAlgoCount);

    CudnnFrontend::Execute("cudnnFindConvolutionForwardAlgorithm"):
    if(CudnnFrontend::Success()){
         *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
         *perfResults       = CudnnFrontend::GetOutputVariable<cudnnConvolutionFwdAlgoPerf_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionForwardAlgorithmEx(cudnnHandle_t handle,
                                                                            const cudnnTensorDescriptor_t xDesc,
                                                                            const void *x,
                                                                            const cudnnFilterDescriptor_t wDesc,
                                                                            const void *w,
                                                                            const cudnnConvolutionDescriptor_t convDesc,
                                                                            const cudnnTensorDescriptor_t yDesc,
                                                                            void *y,
                                                                            const int requestedAlgoCount,
                                                                            int *returnedAlgoCount,
                                                                            cudnnConvolutionFwdAlgoPerf_t *perfResults,
                                                                            void *workSpace,
                                                                            size_t workSpaceSizeInBytes){

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddHostPointerForArguments(x);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
    CudnnFrontend::AddHostPointerForArguments(w);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddHostPointerForArguments(y);
    CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);
    CudnnFrontend::AddHostPointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnFindConvolutionForwardAlgorithmEx");
    if(CudnnFrontend::Success()){
        y = CudnnFrontend::AddOutputHostPointer();
       *returnedAlgoCount = CudnnFrontend::AddOutputVariable<int>();
        perfResults = CudnnFrontend::AddOutputVariable<cudnnConvolutionFwdAlgoPerf_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithm(cudnnHandle_t handle,
                                                                              const cudnnTensorDescriptor_t xDesc,
                                                                              const cudnnFilterDescriptor_t wDesc,
                                                                              const cudnnConvolutionDescriptor_t convDesc,
                                                                              const cudnnTensorDescriptor_t yDesc,
                                                                              cudnnConvolutionFwdPreference_t preference,
                                                                              size_t memoryLimitInBytes,
                                                                              cudnnConvolutionFwdAlgo_t *algo){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((convDesc));
    CudnnFrontend::AddVariableForArguments<long long int>((yDesc));
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionFwdPreference_t>(preference);
    CudnnFrontend::AddVariableForArguments<size_t>();
    
    CudnnFrontend::Execute("cudnnGetConvolutionForwardAlgorithm");
    if(CudnnFrontend::Success()){
        *algo = CudnnFrontend::GetOutputVariable<cudnnConvolutionFwdAlgo_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle_t handle,
                                                                            const cudnnTensorDescriptor_t srcDesc,
                                                                            const cudnnFilterDescriptor_t filterDesc,
             						                    const cudnnConvolutionDescriptor_t convDesc,
								            const cudnnTensorDescriptor_t destDesc,
									    const int requestedAlgoCount,
 									    int *returnedAlgoCount,
								     	    cudnnConvolutionFwdAlgoPerf_t *perfResults){


    CudnnFrontend::Prepare();
   
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)srcDesc); 
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)destDesc);
    CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);

    CudnnFrontend::Execute("cudnnGetConvolutionForwardAlgorithm_v7");
    if(CudnnFrontend::Success()){
        *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
        *perfResults       = CudnnFrontend::GetOutputVariable<cudnnConvolutionFwdAlgoPerf_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t handle,
                                                                                  const cudnnTensorDescriptor_t xDesc,
                                                                                  const cudnnFilterDescriptor_t wDesc,
                                                                                  const cudnnConvolutionDescriptor_t convDesc,
                                                                                  const cudnnTensorDescriptor_t yDesc,
                                                                                  cudnnConvolutionFwdAlgo_t algo,
                                                                                   size_t *sizeInBytes){
    
    CudnnFrontend::Prepare();
   
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionFwdAlgo_t>(algo);

    CudnnFrontend::Execute("cudnnGetConvolutionForwardWorkspaceSize");
    if(CudnnFrontend::Success()){
        *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnConvolutionForward(cudnnHandle_t handle,
                                                             const void *alpha,
                                                             const cudnnTensorDescriptor_t xDesc,
                                                             const void *x,
                                                             const cudnnFilterDescriptor_t wDesc,
                                                             const void *w,
                                                             const cudnnConvolutionDescriptor_t convDesc,
                                                             cudnnConvolutionFwdAlgo_t algo,
                                                             void *workSpace,
                                                             size_t workSpaceSizeInBytes,
                                                             const void *beta,
                                                             const cudnnTensorDescriptor_t yDesc,
                                                             void *y){

    CudnnFrontend::Prepare();
  
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddHostPointerForArguments(x);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
    CudnnFrontend::AddHostPointerForArguments(w);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionFwdAlgo_t>(algo);
    CudnnFrontend::AddHostPointerForArguments()workSpace;
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(beta);
    CudnnFrontend::AddVariableForArguments<long long int>((longlong int)yDesc);
    CudnnFrontend::AddHostPointerForArguments(y);

    CudnnFrontend::Execute(""cudnnConvolutionForward);
    if(CudnnFrontend::Success()){
        y = CudnnFrontend::GetOutputHostPointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnConvolutionBiasActivationForward(cudnnHandle_t handle,
                                                                           const void *alpha1,
                                                                           const cudnnTensorDescriptor_t xDesc,
                                                                           const void *x,
                                                                           const cudnnFilterDescriptor_t wDesc,
                                                                           const void *w,
                                                                           const cudnnConvolutionDescriptor_t convDesc,
                                                                           cudnnConvolutionFwdAlgo_t algo,
                                                                           void *workSpace,
                                                                           size_t workSpaceSizeInBytes,
                                                                           const void *alpha2,
                                                                           const cudnnTensorDescriptor_t zDesc,
                                                                           const void *z,
                                                                           const cudnnTensorDescriptor_t biasDesc,
                                                                           const void *bias,
                                                                           const cudnnActivationDescriptor_t activationDesc,
                                                                           const cudnnTensorDescriptor_t yDesc,
                                                                           void *y){

     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddHostPointerForArguments(alpha1);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
     CudnnFrontend::AddHostPointerForArguments(x);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
     CudnnFrontend::AddHostPointerForArguments(w);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
     CudnnFrontend::AddVariableForArguments<cudnnConvolutionFwdAlgo_t>(algo);
     CudnnFrontend::AddHostPointerForArguments(workSpace);
     CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
     CudnnFrontend::AddHostPointerForArguments(alpha2);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)zDesc);
     CudnnFrontend::AddHostPointerArguments(z);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)biasDesc);
     CudnnFrontend::AddHostPointerForArguments(bias);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)activationDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
     CudnnFrontend::AddHostPointerForArguments(y);
     
     CudnnFrontend::Execute("cudnnConvolutionBiasActivationForward");
     if(Cudnnfrontend::Success()){
         y = CudnnFrontend::GetOutputHostPointer();
      }
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardBias(cudnnHandle_t handle,
                                                                  const void *alpha,
                                                                  const  cudnnTensorDescriptor_t dyDesc,
                                                                  const void *dy,
                                                                  const void *beta,
                                                                  const cudnnTensorDescriptor_t dbDesc,
                                                                  void *db){

     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)hndle);
     CudnnFrontend::AddHostPointerForArguments(alpha);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
     CudnnFrontend::AddHostPointerForArguments(dy);
     CudnnFrontend::AddHostPointerForArguments(beta);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dbDesc);
     
     CudnnFrontend::Execute("cudnnConvolutionBackwardBias");
     if(CudnnFrontend::Success()){
        db = CudnnFrontend::GetOutputHostPointer();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle_t handle, int *count){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);

    CudnnFrontend::Execute("cudnnGetConvolutionBackwardFilterAlgorithmMaxCount");
    if(CudnnFrontend::Success()){
       *count = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardFilterAlgorithm(cudnnHandle_t handle,
                                                                                 const cudnnTensorDescriptor_t xDesc,
                                                                                 const cudnnTensorDescriptor_t dyDesc,
                                                                                 const cudnnConvolutionDescriptor_t convDesc,
                                                                                 const cudnnFilterDescriptor_t dwDesc,
                                                                                 const int requestedAlgoCount,
                                                                                 int *returnedAlgoCount,
                                                                                 cudnnConvolutionBwdFilterAlgoPerf_t *perfResults){

    CudnnFrontend::Prepare();
  
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dwDesc);
    CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);

    CudnnFrontend::Execute("cudnnFindConvolutionBackwardFilterAlgorithm");
    if(CudnnFrontend::Success()){
       *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
       *perfResults = CudnnFrontend::GetOutputVariable<cudnnConvolutionBwdFilterAlgoPerf_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardFilterAlgorithmEx(cudnnHandle_t handle,
                                                                                   const cudnnTensorDescriptor_t xDesc,
                                                                                   const void *x,
                                                                                   const cudnnTensorDescriptor_t dyDesc,
                                                                                   const void *y,
                                                                                   const cudnnConvolutionDescriptor_t convDesc,
                                                                                   const cudnnFilterDescriptor_t dwDesc,
                                                                                   void *dw,
                                                                                   const int requestedAlgoCount,
                                                                                   int *returnedAlgoCount,
                                                                                   cudnnConvolutionBwdFilterAlgoPerf_t *perfResults,
                                                                                   void *workSpace,
                                                                                   size_t workSpaceSizeInBytes){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddHostPointerForArguments(x);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
    CudnnFrontend::AddHostPointerForArguments(y);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dwDesc);
    CudnnFrontend::AddHostPointerForArguments(dw);
    CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);
    CudnnFrontend::AddHostPointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnFindConvolutionBackwardFilterAlgorithmEx");
    if(CudnnFrontend::Success()){
       dw = CudnnFrontend::GetOutputHostPointer();
       *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
       *perfResults = CudnnFrontend::GetOutputVariable<cudnnConvolutionBwdFilterAlgoPerf_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle_t handle,
                                                                                const cudnnTensorDescriptor_t xDesc,
                                                                                const cudnnTensorDescriptor_t dyDesc,
                                                                                const cudnnConvolutionDescriptor_t convDesc,
                                                                                const cudnnFilterDescriptor_t dwDesc,
                                                                                cudnnConvolutionBwdFilterPreference_t preference,
                                                                                size_t memoryLimitInBytes,
                                                                                cudnnConvolutionBwdFilterAlgo_t *algo){

     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dwDesc);
     CudnnFrontend::AddVariableForArguments<cudnnConvolutionBwdFilterPreference_t>(preference);
     CudnnFrontend::AddVariableForArguments<size_t>(memoryLimitInBytes);

     Cudnnfrontend::Execute("cudnnGetConvolutionBackwardFilterAlgorithm");
     if(CudnnFrontend::Success()){
         *algo = CudnnFrontend::GetOutputVariable<cudnnConvolutionBwdFilterAlgo_t>();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnnHandle_t handle,
                                                                                   const cudnnTensorDescriptor_t srcDesc,
                                                                                   const cudnnTensorDescriptor_t diffDesc,
                                                                                   const cudnnConvolutionDescriptor_t convDesc,
                                                                                   const cudnnFilterDescriptor_t gradDesc,
                                                                                   const int requestedAlgoCount,
                                                                                   int *returnedAlgoCount,
                                                                                   cudnnConvolutionBwdFilterAlgoPerf_t *perfResults){

      CudnnFrontend::Prepare();

      CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)srcDesc);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)diffDesc);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)gradDesc);
      CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);
      
      CudnnFrontend::Execute("cudnnGetConvolutionBackwardFilterAlgorithm_v7");
      if(CudnnFrontend::Success()){
          *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
          *perfResults       = CudnnFrontend::GetOutputVariable<cudnnConvolutionBwdFilterAlgoPerf_t>();
      }
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle_t handle,
                                                                                    const cudnnTensorDescriptor_t xDesc,
                                                                                    const cudnnTensorDescriptor_t dyDesc,
                                                                                    const cudnnConvolutionDescriptor_t convDesc,
                                                                                    const cudnnFilterDescriptor_t gradDesc,
                                                                                    cudnnConvolutionBwdFilterAlgo_t algo,
                                                                                    size_t *sizeInBytes){


     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)gradDesc);
     CudnnFrontend::AddVariableForArguments<cudnnConvolutionBwdFilterAlgo_t>(algo);
     

    CudnnFrontend::Execute("cudnnGetConvolutionBackwardFilterWorkspaceSize");
    if(CudnnFrontend::Success()){
      *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardFilter(cudnnHandle_t handle,
                                                                    const void *alpha,
                                                                    const cudnnTensorDescriptor_t xDesc,
                                                                    const void *x,
                                                                    const cudnnTensorDescriptor_t dyDesc,
                                                                    const void *dy,
                                                                    const cudnnConvolutionDescriptor_t convDesc,
                                                                    cudnnConvolutionBwdFilterAlgo_t algo,
                                                                    void *workSpace,
                                                                    size_t workSpaceSizeInBytes,
                                                                    const void *beta,
                                                                    const cudnnFilterDescriptor_t dwDesc,
                                                                    void *dw){

     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddHostPointerForArguments(alpha);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
     CudnnFrontend::AddHostPointerForArguments(x);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
     CudnnFrontend::AddHostPointerForArguments(dy);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
     CudnnFrontend::AddVariableForArguments<cudnnConvolutionBwdFilterAlgo_t>(algo);
     CudnnFrontend::AddHostPointerForArguments(workSpace);
     CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
     CudnnFrontend::AddHostPointerForArguments(beta);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dwDesc);
     CudnnFrontend::AddHostPointerForArguments(dw);

     CudnnFrontend::Execute("cudnnConvolutionBackwardFilter");
     if(CudnnFrontend::Success()){
        dw = CudnnFrontend::GetOutputHostPointer();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t handle, int *count){
     
     CudnnFrontend::Prepare();
     
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);

     CudnnFrontend::Execute("cudnnGetConvolutionBackwardDataAlgorithmMaxCount");
     if(CudnnFrontend::Success()){
         *count = CudnnFrontend::GetOutputVariable<int>();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardDataAlgorithm(cudnnHandle_t handle,
                                                                               const cudnnFilterDescriptor_t wDesc,
                                                                               const cudnnTensorDescriptor_t dyDesc,
                                                                               const cudnnConvolutionDescriptor_t convDesc,
                                                                               const cudnnTensorDescriptor_t dxDesc,
                                                                               const int requestedAlgoCount,
                                                                               int *returnedAlgoCount,
                                                                               cudnnConvolutionBwdDataAlgoPerf_t *perfResults){

     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dxDesc);
     CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);

     CudnnFrontend::Execute("cudnnFindConvolutionBackwardDataAlgorithm");
     if(CudnnFrontend::Success()){
         *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
         *perfResults       = CudnnFrontend::GetOutputVariable<cudnnConvolutionBwdDataAlgoPerf_t>();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardDataAlgorithmEx(cudnnHandle_t handle,
                                						 const cudnnFilterDescriptor_t wDesc,
										 const void *w,
 										 const cudnnTensorDescriptor_t dyDesc,
										 const void *dy,
										 const cudnnConvolutionDescriptor_t convDesc,
										 const cudnnTensorDescriptor_t dxDesc,
     										 void *dx,
										 const int requestedAlgoCount,
										 int *returnedAlgoCount,
										 cudnnConvolutionBwdDataAlgoPerf_t *perfResults,
										 void *workSpace,
										 size_t workSpaceSizeInBytes){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
   CudnnFrontend::AddHostPointerForArguments(w);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
   CudnnFrontend::AddHostPointerForArguments(dy);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)dxDesc);
   CudnnFrontend::AddHostPointerForArguments(dx);
   CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);
   CudnnFrontend::AddHostPointerForArguments(workSpace);
   CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);

   CudnnFrontend::Execute("cudnnFindConvolutionBackwardDataAlgorithmEx");
   if(CudnnFrontend::Success()){
      dx = CudnnFrontend::GetOutputHostPointer();
      *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
      *perfResults       = Cudnnfrontend::GetOutputVariable<cudnnConvolutionBwdDataAlgoPerf_t>();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle_t handle,
 									      const cudnnFilterDescriptor_t wDesc,
									      const cudnnTensorDescriptor_t dyDesc,
									      const cudnnConvolutionDescriptor_t convDesc,
									      const cudnnTensorDescriptor_t dxDesc,
									      cudnnConvolutionBwdDataPreference_t preference,
									      size_t memoryLimitInBytes,
									      cudnnConvolutionBwdDataAlgo_t *algo){

     CudnnFrontend::Prepare();
    
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dxDesc);
     CudnnFrontend::AddVariableForArguments<cudnnConvolutionBwdDataPreference_t>(preference);
     CudnnFrontend::AddVariableForArguments<size_t>(memoryLimitInBytes);

     CudnnFrontend::Execute("cudnnGetConvolutionBackwardDataAlgorithm");
     if(CudnnFrontend::Success()){
          *algo = CudnnFrontend::GetOutputVariable<cudnnConvolutionBwdDataAlgo_t>();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnnHandle_t handle,
										 const cudnnFilterDescriptor_t filterDesc,
           								  	 const cudnnTensorDescriptor_t diffDesc,
										 const cudnnConvolutionDescriptor_t convDesc,
										 const cudnnTensorDescriptor_t gradDesc,
										 const int requestedAlgoCount,
										 int *returnedAlgoCount,
										 cudnnConvolutionBwdDataAlgoPerf_t *perfResults){

     CudnnFrontend::Prepare();
   
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)diffDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)gradDesc);
     CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);

     CudnnFrontend::Execute("cudnnGetConvolutionBackwardDataAlgorithm_v7");
     if(CudnnFrontend::Success()){
          *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
          *perfResults       = CudnnFrontend::GetOutputVariable<cudnnConvolutionBwdDataAlgoPerf_t>();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle_t handle,
										  const cudnnFilterDescriptor_t wDesc,
										  const cudnnTensorDescriptor_t dyDesc,
										  const cudnnConvolutionDescriptor_t convDesc,
										  const cudnnTensorDescriptor_t dxDesc,
										  cudnnConvolutionBwdDataAlgo_t algo,
										  size_t *sizeInBytes){


    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dxDesc);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionBwdDataAlgo_t>(algo);
 
    CudnnFrontend::Execute("cudnnGetConvolutionBackwardDataWorkspaceSize");
    if(CudnnFrontend::Success()){
          *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>(); 
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardData(cudnnHandle_t handle,
								  const void *alpha,
								  const cudnnFilterDescriptor_t wDesc,
								  const void *w,
								  const cudnnTensorDescriptor_t *dyDesc,
								  const void *dy,
								  const cudnnConvolutionDescriptor_t convDesc,
								  cudnnConvolutionBwdDataAlgo_t algo,
								  void *workSpace,
								  size_t workSpaceSizeInBytes,
								  const void *beta,
								  const cudnnTensorDescriptor_t dxDesc,
								  void *dx){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
    CudnnFrontend::AddHostPointerForArguments(w);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
    CudnnFrontend::AddHostPointerForArguments(dy);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionBwdDataAlgo_t>(algo);
    CudnnFrontend::AddHostPointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dxDesc);
    CudnnFrontend::AddHostPointerForArguments(dx);

    CudnnFrontend::Execute("cudnnConvolutionBackwardData");
    if(CudnnFrontend::Success()){
        dx = CudnnFrontend::GetOutputHostPointer();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnIm2Col(cudnnHandle_t handle,
						 const cudnnTensorDescriptor_t xDesc,
						 const void *x,
						 const cudnnFilterDescriptor_t wDesc,
						 const cudnnConvolutionDescriptor_t convDesc,
						 void *colBuffer){

     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
     CudnnFrontend::AddHostPointerForArguments(x);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);

     CudnnFrontend::Execute("cudnnIm2Col");
     if(CudnnFrontend::Success()){
         colBuffer = CudnnFrontend::GetOutputHostPointer();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSoftmaxForward(cudnnHandle_t handle,
							 cudnnSoftmaxAlgorithm_t algo,
							 cudnnSoftmaxMode_t mode,
							 const void *alpha,
							 const cudnnTensorDescriptor_t xDesc,
							 const void *x,
							 const void *beta,
							 const cudnnTensorDescriptor_t yDesc,
							 void *y){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<cudnnSoftmaxAlgorithm_t>(algo);
    CudnnFrontend::AddVariableForArguments<cudnnSoftmaxMode_t>(mode);
    CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddHostPointerForArguments(x);
    CudnnFrontend::AddHostPointerForArguments(beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);

    CudnnFrontend::Execute("cudnnSoftmaxForward");
    if(CudnnFrontend::Success()){
         y = CudnnFrontend::GetOutputHostPointer();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSoftmaxBackward(cudnnHandle_t handle,
							  cudnnSoftmaxAlgorithm_t algo,
							  cudnnSoftmaxMode_t mode,
							  const void *alpha,
							  const cudnnTensorDescriptor_t yDesc,
							  const void *y,
							  const cudnnTensorDescriptor_t dyDesc,
							  const void *dy,
							  const void *beta,
							  const cudnnTensorDescriptor_t dxDesc,
							  const void *dx){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<cudnnSoftmaxAlgorithm_t>(algo);
    CudnnFrontend::AddVariableForArguments<cudnnSoftmaxMode_t>(mode);
    CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddHostPointerForArguments(y);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
    CudnnFrontend::AddHostPointerForArguments(dy);
    CudnnFrontend::AddHostPointerForArguments(beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dxDesc);
    

    CudnnFrontend::Execute("cudnnSoftmaxBackward");
    if(CudnnFrontend::Success()){
        dx = CudnnFrontend::GetOutputHostPointer();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc){
   CudnnFrontend::Prepare();

   CudnnFrontend::Execute("cudnnCreatePoolingDescriptor");
   if(CudnnFrontend::Success())
      *poolingDesc = CudnnFrontend::GetOutputVariable<cudnnPoolingDescriptor_t>();
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                                                                 cudnnPoolingMode_t mode,
                                                                 cudnnNanPropagation_t maxpoolingNanOpt,
                                                                 int windowHeight,
                                                                 int windowWidth,
                                                                 int verticalPadding,
                                                                 int horizontalPadding,
                                                                 int verticalStride,
                                                                 int horizontalStride){
   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)poolingDesc);
   CudnnFrontend::AddVariableForArguments<cudnnPoolingMode_t>(mode);
   CudnnFrontend::AddVariableForArguments<cudnnNanPropagation_t>(maxpoolingNanOpt);
   CudnnFrontend::AddVariableForArguments<int>(windowHeight);
   CudnnFrontend::AddVariableForArguments<int>(windowWidth);
   CudnnFrontend::AddVariableForArguments<int>(verticalPadding);
   CudnnFrontend::AddVariableForArguments<int>(horizontalPadding);
   CudnnFrontend::AddVariableForArguments<int>(verticalStride);
   CudnnFrontend::AddVariableForArguments<int>(horizontalStride);

   CudnnFrontend::Execute("cudnnSetPooling2dDescriptor");
   return CudnnFrontend::GetExitCode();
}
 
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetPooling2dDescripto(const cudnnPoolingDescriptor_t poolingDesc,
							        cudnnPoolingMode_t *mode,
								cudnnNanPropagation_t *maxpoolingNanOpt,
								int *windowHeight,
								int *windowWidth,
								int *verticalPadding,
								int *horizontalPadding,
								int *verticalStride,
								int *horizontalStride){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)poolingDesc);

    CudnnFrontend::Execute("cudnnGetPooling2dDescripto");
    if(CudnnFrontend::Success()){
        *mode = CudnnFrontend::GetOutputVariable<cudnnPoolingMode_t>();
        *maxpoolingNanOpt = CudnnFrontend::GetOutputVariable<cudnnNanPropagation_t>();
        *windowHeight     = CudnnFrontend::GetOutputVariable<int>();
        *windowWidth      = CudnnFrontend::GetOutputVariable<int>();
        *verticalPadding  = CudnnFrontend::GetOutputVariable<int>();
        *horizontalPadding = CudnnFrontend::GetOutputVariable<int>();
        *verticalStride    = CudnnFrontend::GetOutputVariable<int>();
        *horizontalStride  = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                                                                 const cudnnPoolingMode_t mode,
                                                                 const cudnnNanPropagation_t maxpoolingNanOpt,
                                                                 int nbDims,
                                                                 const int *windowDimA,
                                                                 const int *paddingA,
                                                                 const int *strideA){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)poolingDesc);
    CudnnFrontend::AddVariableForArguments<cudnnPoolingMode_t>(mode);
    CudnnFrontend::AddVariableForArguments<cudnnNanPropagation_t>(maxpoolingNanOpt);
    CudnnFrontend::AddVariableForArguments<int>(nbDims);
    CudnnFrontend::AddVariableForArguments<int>((int*)windowDimA);
    CudnnFrontend::AddVariableForArguments<int>((int*)paddingA);
    CudnnFrontend::AddVariableForArguments<int>((int*)strideA);

    CudnnFrontend::Execute("cudnnSetPoolingNdDescriptor");
    if(CudnnFrontend::Success()){
       poolingDesc = CudnnFrontend::GetOutputVariable<cudnnPoolingDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetPoolingNdDescriptor(const cudnnPoolingDescriptor_t poolingDesc,
								 int nbDimsRequested,
								 cudnnPoolingMode_t *mode,
								 cudnnNanPropagation_t *maxpoolingNanOpt,
								 int *nbDims,
								 int *windowDimA,
								 int *paddingA,
								 int *strideA){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)poolingDesc);
    CudnnFrontend::AddVariableForArguments<int>(nbDimsRequested);
    CudnnFrontend::AddVariableForArguments<cudnnNanPropagation_t>(maxpoolingNanOpt);
    
    CudnnFrontend::Execute("cudnnGetPoolingNdDescriptor");
    if(CudnnFrontend::Success()){
       *mode = CudnnFrontend::GetOutputVariable<cudnnPoolingMode_t>();
       *nbDims = CudnnFrontend::GetOutputVariable<int>();
       *windowDimA = (int*)CudnnFrontend::GetOutputVariable<int>();
       *paddingA   = (int*)CudnnFrontend::GetOutputVariable<int>();
       *strideA    = (int*)CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
								       const cudnnTensorDescriptor_t inputTensorDesc,
								       int nbDims,
								       int *outputTensorDimA){

     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)poolingDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)inputTensorDesc);
     CudnnFrontend::AddVariableForArguments<int>(nbDims);

    CudnnFrontend::Execute("cudnnGetPoolingNdForwardOutputDim");
    if(CudnnFrontend::Success()){
        *outputTensorDimA = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
   								       const cudnnTensorDescriptor_t inputTensorDesc,
								       int *n,
								       int *c,
      								       int *h,
								       int *w){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)poolingDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)inputTensorDesc);
    
    CudnnFrontend::Execute("cudnnGetPooling2dForwardOutputDim");
    if(CudnnFrontend::Success()){
       *n = CudnnFrontend::GetOutputVaribale<int>();
       *c = CudnnFrontend::GetOutputVaribale<int>();
       *h = CudnnFrontend::GetOutputVaribale<int>();
       *w = CudnnFrontend::GetOutputVaribale<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc){

    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)poolingDesc);

    CudnnFrontend::Execute("cudnnDestroyPoolingDescriptor");
   
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnPoolingForward(cudnnHandle_t handle,
							 const cudnnPoolingDescriptor_t poolingDesc,
							 const void *alpha,
							 const cudnnTensorDescriptor_t xDesc,
							 const void *x,
							 const void *beta,
							 const cudnnTensorDescriptor_t yDesc,
							 void *y){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)poolingDesc);
    CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddHostPointerForArguments(x);
    CudnnFrontend::AddHostPointerForArguments(beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    

   CudnnFrontend::Execute("cudnnPoolingForward");
   if(CudnnFrontend::Success()){
       y = CudnnFrontend::GetOutputHostPointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnPoolingBackward(cudnnHandle_t handle,
							  const cudnnPoolingDescriptor_t poolingDesc,
							  const void *alpha,
							  const cudnnTensorDescriptor_t yDesc,
							  const void *y,
							  const cudnnTensorDescriptor_t dyDesc,
							  const *dy,
							  const cudnnTensorDescriptor_t xDesc,
							  const void *x,
							  const void *beta,
							  const cudnnTensorDescriptor_t dxDesc,
							  void *dx){


    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)poolingDesc);
    CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddHostPointerForArguments(y);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
    CudnnFrontend::AddHostPointerForArguments(dy);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddHostPointerForArguments(x);
    CudnnFrontend::AddHostPointerForArguments(beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dxDesc);

    CudnnFrontend::Execute("cudnnPoolingBackward");
    if(CudnnFrontend::Success()){
        dx = CudnnFrontend::GetOutputHostPointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t *activationDesc){
   CudnnFrontend::Prepare();

   CudnnFrontend::Execute("cudnnCreateActivationDescriptor");
   if(CudnnFrontend::Success())
      *activationDesc = CudnnFrontend::GetOutputVariable<cudnnActivationDescriptor_t>();
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc,
                                                                  cudnnActivationMode_t mode,
                                                                  cudnnNanPropagation_t reluNanOpt,
                                                                  double coef){
    CudnnFrontend::Prepare();
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)activationDesc);
    CudnnFrontend::AddVariableForArguments<cudnnActivationMode_t>(mode);
    CudnnFrontend::AddVariableForArguments<cudnnNanPropagation_t>(reluNanOpt);
    CudnnFrontend::AddVariableForArguments<double>(coef);
    CudnnFrontend::Execute("cudnnSetActivationDescriptor");
    return CudnnFrontend::GetExitCode();
}
        
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetActivationDescriptor(const cudnnActivationDescriptor_t activationDesc,
                                                                  cudnnActivationMode_t *mode,
                                                                  cudnnNanPropagation_t *reluNanOpt,
                                                                  double *coef){
    CudnnFrontend::Prepare();
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)activationDesc);
    CudnnFrontend::Execute("cudnnGetActivationDescriptor");
    if(CudnnFrontend::Success()){
        *mode = CudnnFrontend::GetOutputVariable<cudnnActivationMode_t>();
        *reluNanOpt = CudnnFrontend::GetOutputVariable<cudnnNanPropagation_t>();
        *coef = CudnnFrontend::GetOutputVariable<double>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc){
    CudnnFrontend::Prepare();
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)activationDesc);
    CudnnFrontend::Execute("cudnnDestroyActivationDescriptor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnActivationForward(cudnnHandle_t handle,
							    cudnnActivationDescriptor_t activationDesc,
							    const void *alpha,
							    const cudnnTensorDescriptor_t xDesc,
							    const void *x,
						            const void *beta,
							    const cudnnTensorDescriptor_t yDesc,
							    void *y){


    CudnnFrontend::Prepare();


    CudnnFrontend::AddVariableForArugments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArugments<long long int>((long long int)activationDesc);
    CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddVariableForArugments<long long int>((long long int)xDesc);
    CudnnFrontend::AddHostPointerForArguments(x);
    CudnnFrontend::AddHostPointerForArguments(beta);
    CudnnFrontend::AddVariableForArugments<long long int>((long long int)yDesc);
    

    CudnnFrontend::Execute("cudnnActivationForward");
    if(CudnnFrontend::Success()){
       y = CudnnFrontend::GetOutputHostPointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnActivationBackward(cudnnHandle_t handle,
                                                             cudnnActivationDescriptor_t activationDesc,
							     const void *alpha,
							     const cudnnTensorDescriptor_t yDesc,
							     const void *y,
							     const cudnnTensorDescriptor_t dyDesc,
							     const void *dy,
							     const cudnnTensorDescriptor_t xDesc,
							     const void *x,
							     const void *beta,
						             const cudnnTensorDescriptor_t dxDesc,
							     void *dx){


    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)activationDesc);
    CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddHostPointerForArguments(y);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
    CudnnFrontend::AddHostPointerForArguments(dy);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddHostPointerForArguments(x);
    CudnnFrontend::AddHostPointerForArguments(beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dxDesc);
    

    CudnnFrontend::Execute("cudnnActivationBackward");
    if(CudnnFrontend::Success()){
       dx = CudnnFrontend::GetOutputHostPointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t *normDesc){
    
    CudnnFrontend::Prepare();
   
    CudnnFrontend::Execute("cudnnCreateLRNDescriptor");
    if(CudnnFrontend::Success()){
       *normDesc = CudnnFrontend::GetOutputVariable<cudnnLRNDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetLRNDescriptor(cudnnLRNDescriptor_t normDesc,
               						   unsigned lrnN,
							   double lrnAlpha,
							   double lrnBeta,
							   double lrnK){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<unsigned>(lrnN);
    CudnnFrontend::AddVariableForArguments<double>(lrnAlpha);
    CudnnFrontend::AddVariableForArguments<double>(lrnBeta);
    CudnnFrontend::AddVariableForArguments<double>(lrnK);

    CudnnFrontend::Execute("cudnnSetLRNDescriptor");
    if(CudnnFrontend::Success()){
        normDesc = CudnnFrontend::GetOutputVariable<cudnnLRNDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetLRNDescriptor(cudnnLRNDescriptor_t normDesc,
						           unsigned *lrnN,
							   double *lrnAlpha,
							   double *lrnBeta,
							   double *lrnK){

   CudnnFrontend::Prepare();

   CudnnFrontend::Execute("cudnnGetLRNDescriptor");
   if(CudnnFrontend::Success()){
       normDesc = CudnnFrontend::GetOutputVariable<cudnnLRNDescriptor_t>();
       *lrnN    = CudnnFrontend::GetOutputVariable<unsigned>();
       *lrnAlpha = CudnnFrontend::GetOutputVariable<double>();
       *lrnBeta  =  CudnnFrontend::GetOutputVariable<double>();
       *lrnK     =  CudnnFrontend::GetOutputVariable<double>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t lrnDesc){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)lrnDesc);
   
    CudnnFrontend::Execute("cudnnDestroyLRNDescriptor");
   
    return CudnnFrontend::GetExitCode();
}














  




















































































 




























