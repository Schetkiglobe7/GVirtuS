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
 *
*/

#include <cstring>
#include <map>
#include <errno.h>
#include <cuda_runtime_api.h>
#include "CudnnHandler.h"

using namespace std;
using namespace log4cplus;

std::map<string, CudnnHandler::CudnnRoutineHandler> * CudnnHandler::mspHandlers = NULL;

extern "C" std::shared_ptr<CudnnHandler> create_t() {
    return std::make_shared<CudnnHandler>();
}


extern "C" int HandlerInit() {
    return 0;
}

CudnnHandler::CudnnHandler() {
    logger=Logger::getInstance(LOG4CPLUS_TEXT("CudnnHandler"));
    setLogLevel(&logger);
    Initialize();
}

CudnnHandler::~CudnnHandler() {

}

void CudnnHandler::setLogLevel(Logger *logger) {
	log4cplus::LogLevel logLevel=log4cplus::INFO_LOG_LEVEL;
	char * val = getenv("GVIRTUS_LOGLEVEL");
	std::string logLevelString =(val == NULL ? std::string("") : std::string(val));
	if(logLevelString != "") {
		logLevel=std::stoi(logLevelString);
	}
	logger->setLogLevel(logLevel);
}

bool CudnnHandler::CanExecute(std::string routine) {
    return mspHandlers->find(routine) != mspHandlers->end();

}

std::shared_ptr<Result> CudnnHandler::Execute(std::string routine, std::shared_ptr<Buffer> input_buffer) {
    LOG4CPLUS_DEBUG(logger,"Called " << routine);
    map<string, CudnnHandler::CudnnRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if (it == mspHandlers->end())
        throw "No handler for '" + routine + "' found!";
    try {
        return it->second(this, input_buffer);
    } catch (const char *ex) {
        cout << ex << endl;
        cout << strerror(errno) << endl;
    }
    return NULL;
}

void CudnnHandler::Initialize(){
   if (mspHandlers != NULL)
        return;
    mspHandlers = new map<string, CudnnHandler::CudnnRoutineHandler> ();

    /* CublasHandler Query Platform Info */
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetVersion));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetErrorString));   
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(Create));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(Destroy));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetStream));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetStream));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetTensor4dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetTensor4dDescriptorEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetTensor4dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetTensorNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetTensorNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetTensorSizeInBytes));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(InitTransformDest));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateTensorTransformDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetTensorTransformDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetTensorTransformDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyTensorTransformDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(TransformTensor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(TransformTensorEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANLDER_PAIR(GetFoldedConvBackwardDataDescriptors));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(AddTensor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateOpTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetOpTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetOpTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyOpTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(OpTensor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateReduceTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetReduceTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetReduceTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyReduceTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetReductionIndicesSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetReductionWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ReduceTensor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetTensor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ScaleTensor)); 
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateFilterDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilter4dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilter4dDescriptor));
    #if CUDNN_VERSION < 6000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilter4dDescriptor_v3));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilter4dDescriptor_v3));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilter4dDescriptor_v4));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilter4dDescriptor_v4));
    #endif
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilterNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilterNdDescriptor));
    #if CUDNN_VERSION < 6000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilterNdDescriptor_v3));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilterNdDescriptor_v3));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilterNdDescriptor_v4));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilterNdDescriptor_v4));
    #endif
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilterSizeInBytes));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyFilterDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(TransformFilter));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ReorderFilterAndBias));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateConvolutionDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetConvolutionMathType));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionMathType));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetConvolutionGroupCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionGroupCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetConvolutionReorderType));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionReorderType));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetConvolution2dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolution2dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolution2dForwardOutputDim));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetConvolutionNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionNdForwardOutputDim));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyConvolutionDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionForwardAlgorithmMaxCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindConvolutionForwardAlgorithm));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindConvolutionForwardAlgorithmEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionForwardAlgorithm));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionForwardAlgorithm_v7));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionForwardWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ConvolutionForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ConvolutionBiasActivationForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ConvolutionBackwardBias));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardFilterAlgorithmMaxCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindConvolutionBackwardFilterAlgorithm));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindConvolutionBackwardFilterAlgorithmEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardFilterAlgorithm));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardFilterAlgorithm_v7));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardFilterWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ConvolutionBackwardFilter));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardDataAlgorithmMaxCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindConvolutionBackwardDataAlgorithm));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindConvolutionBackwardDataAlgorithmEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardDataAlgorithm));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardDataAlgorithm_v7));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardDataWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ConvolutionBackwardData));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(Im2Col));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SoftmaxForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SoftmaxBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreatePoolingDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetPooling2dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetPooling2dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetPoolingNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetPoolingNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetPoolingNdForwardOutputDim));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetPooling2dForwardOutputDim));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyPoolingDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(PoolingForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(PoolingBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateActivationDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetActivationDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetActivationDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyActivationDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ActivationForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ActivationBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateLRNDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetLRNDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetLRNDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyLRNDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(LRNCrossChannelForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(LRNCrossChannelBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DivisiveNormalizationForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DivisiveNormalizationBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DeriveBNTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetBatchNormalizationForwardTrainingExWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetBatchNormalizationBackwardExWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetBatchNormalizationTrainingExReserveSpaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(BatchNormalizationForwardTraining));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(BatchNormalizationForwardTrainingEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(BatchNormalizationForwardInference));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(BatchNormalizationBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(BatchNormalizationBackwardEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateSpatialTransformerDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetSpatialTransformerNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroySpatialTransformerDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SpatialTfGridGeneratorForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SpatialTfGridGeneratorBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SpatialTfSamplerForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SpatialTfSamplerBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateDropoutDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyDropoutDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DropoutGetStatesSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DropoutGetReserveSpaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetDropoutDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RestoreDropoutDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetDropoutDescriptor)); 
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DropoutForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DropoutBackward));
    mspHAndlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateRNNDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyRNNDescriptor));
    mspHAndlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNMatrixMathType));
    mspHAndlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNMatrixMathType));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNBiasMode));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNBiasMode));
    mspHAndlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNSetClip));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNGetClip));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNProjectionLayers));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNProjectionLayers));
    mspHAndlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreatePersistentRNNPlan));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyPersistentRNNPlan));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetPersistentRNNPlan));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNTrainingReserveSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNParamsSize));   
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNLinLayerMatrixParams));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNLinLayerBiasParams));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNForwardInference));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNForwardTraining));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNBackwardData));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNBackwardWeights));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNPaddingMode));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNPaddingMode));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateRNNDataDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyRNNDataDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNDataDescriptor));
    mspHAndlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNDataDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNForwardTrainingEx));
    mspHAndlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNForwardInferenceEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNBackwardDataEx));
    mspHandlers->isnert(CUDNN_ROUTINE_HANDLER_PAIR(RNNBackwardWeightsEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNAlgorithmDescriptor));
    mspHAndlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNForwardInferenceAlgorithmMaxCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindRNNForwardInferenceAlgorithmEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNForwardTrainingAlgorithmMaxCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindRNNForwardTrainingAlgorithmEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNBackwardDataAlgorithmMaxCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindRNNBackwardDataAlgorithmEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNBackwardWeightsAlgorithmMaxCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindRNNBackwardWeightsAlgorithmEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateSeqDataDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroySeqDataDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetSeqDataDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetSeqDataDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateAttnDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyAttnDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetAttnDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR((GetAttnDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetMultiHeadAttnBuffers));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetMultiHeadAttnWeights));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(MultiHeadAttnForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(MultiHeadAttnBackwardData));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(MultiHeadAttnBackwardWeights));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateCTCLossDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetCTCLossDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetCTCLossDescriptorEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetCTCLossDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetCTCLossDescriptorEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyCTCLossDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CTCLoss));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetCTCLossWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateAlgorithmDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetAlgorithmDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetAlgorithmDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CopyAlgorithmDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyAlgorithmDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateAlgorithmPerformance));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetAlgorithmPerformance));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetAlgorithmPerformance));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyAlgorithmPerformance));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetAlgorithmSpaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SaveAlgorithm));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RestoreAlgorithm));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetCallback));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetCallback));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateFusedOpsConstParamPack));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyFusedOpsConstParamPack));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFusedOpsConstParamPackAttribute));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFusedOpsConstParamPackAttribute));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateFusedOpsVariantParamPack));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyFusedOpsVariantParamPack));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFusedOpsVariantParamPackAttribute));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFusedOpsVariantParamPackAttribute));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateFusedOpsPlan));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyFusedOpsPlan));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FusedOpsExecute));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNDescriptor_v6));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNDescriptor_v5)); 

}

CUDNN_ROUTINE_HANDLER(GetVersion){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetVersion"));

    size_t version = cudnnGetVersion();
    cout << "DEBUG - cudnnGetVersion Executed"<<endl;
    return std::make_shared<Result>(version);
}

CUDNN_ROUTINE_HANDLER(GetErrorString){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetErrorString"));
    cudnnStatus_t cs = in->Get<cudnnStatus_t>();
    const char * s = cudnnGetErrorString(cs);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add((char *)s);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUDNN_STATUS_EXECUTION_FAILED);
    }
    cout << "DEBUG - cudnnGetErrorString Executed"<<endl;
    return std::make_shared<Result>(CUDNN_STATUS_SUCCESS,out);
}

CUDNN_ROUTINE_HANDLER(Create){

    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Create"));
    cudnnHandle_t handle;
    cudnnStatus_t cs = cudnnCreate(&handle);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
         out->Add<cudnnHandle_t>(handle);
    } catch (string e){
                        LOG4CPLUS_DEBUG(logger,e);
                        return std::make_shared<Result>(CUDNN_STATUS_EXECUTION_FAILED);
    }
    std::cout << "DEBUG - cudnnCreate Executed"<<endl;
    return std::make_shared<Result>(cs,out);

}

CUDNN_ROUTINE_HANDLER(Destroy){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Destroy"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnStatus_t cs = cudnnDestroy(handle);
    cout << "DEBUG - cudnnDestroy Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetStream){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetStream"));
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudaStream_t streamId = (cudaStream_t) in->Get<long long int>();

    cudnnStatus_t cs = cudnnSetStream(handle,streamId);
    cout << "DEBUG - cudnnSetStream Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetStream){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetStream"));
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudaStream_t *streamId;
    cudnnStatus_t cs = cudnnGetStream(handle,streamId);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->Add<long long int>((long long int)*streamId);
    } catch (string e){
         LOG4CPLUS_DEBUG(logger,e);
         return std::make_shared<Result>(cs);
    }
    cout << "DEBUG - cudnnGetStream Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(CreateTensorDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateTensorDescriptor"));
    cudnnTensorDescriptor_t tensorDesc;
    cudnnStatus_t cs = cudnnCreateTensorDescriptor(&tensorDesc);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->Add<cudnnTensorDescriptor_t>(tensorDesc);
    } catch (string e){
         LOG4CPLUS_DEBUG(logger,e);
         return std::make_shared<Result>(cs);
    }
    cout << "DEBUG - cudnnCreateTensorDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(SetTensor4dDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetTensor4dDescriptor"));
    cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnTensorFormat_t format = in->Get<cudnnTensorFormat_t>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();                                                                                          int n = in->Get<int>();
    int c = in->Get<int>();
    int h = in->Get<int>();
    int w = in->Get<int>();
    cudnnStatus_t cs = cudnnSetTensor4dDescriptor(tensorDesc,format,dataType,n,c,h,w);                                                         cout << "DEBUG - cudnnSetTensor4dDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetTensor4dDescriptorEx){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetTensor4dDescriptor"));

    cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();

    int n = in->Get<int>();
    int c = in->Get<int>();
    int h = in->Get<int>();
    int w = in->Get<int>();

    int nStride = in->Get<int>();
    int cStride = in->Get<int>();
    int hStride = in->Get<int>();
    int wStride = in->Get<int>();

    cudnnStatus_t cs = cudnnSetTensor4dDescriptorEx(tensorDesc,dataType,n,c,h,w,nStride,cStride,hStride,wStride);
    cout << "DEBUG - cudnnSetTensor4dDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetTensor4dDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetTensor4dDescriptor"));
    cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();

    cudnnDataType_t dataType;
    int n,c,h,w;
    int nStride,cStride,hStride,wStride;

    cudnnStatus_t cs = cudnnGetTensor4dDescriptor(tensorDesc,&dataType,&n,&c,&h,&w,&nStride,&cStride,&hStride,&wStride);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnDataType_t>(dataType);
        out->Add<int>(n);
        out->Add<int>(c);
        out->Add<int>(h);
        out->Add<int>(w);
        out->Add<int>(nStride);
        out->Add<int>(cStride);
        out->Add<int>(hStride);
        out->Add<int>(wStride);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    cout << "DEBUG - cudnnGetTensor4dDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(SetTensorNdDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetTensorNdDescriptor"));

    cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
    int nbDims = in->Get<int>();
    int *dimA = in->Assign<int>();
    int *strideA = in->Assign<int>();

    cudnnStatus_t cs = cudnnSetTensorNdDescriptor(tensorDesc,dataType,nbDims,dimA,strideA);
    cout << "DEBUG - cudnnSetTensorNdDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetTensorNdDescriptorEx){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetTensorNdDescriptorEx"));

    cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnTensorFormat_t format = in->Get<cudnnTensorFormat_t>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
    int nbDims = in->Get<int>();
    int *dimA = in->Get<int>();

    cudnnStatus_t cs = cudnnSetTensorNdDescriptorEx(tensorDesc, format, dataType, nbDims, dimA);
    cout << "DEBUG - cudnnSetTensorNdDescriptorEx Executed"<<endl;
    return std::make_shared<Result>(cs);  
}

CUDNN_ROUTINE_HANDLER(GetTensorNdDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetTensorNdDescriptor"));

    cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    int nbDimsRequested = in->Get<int>();
    cudnnDataType_t dataType;
    int *nbDims;
    int *dimA;
    int *strideA;

    cudnnStatus_t cs = cudnnGetTensorNdDescriptor(tensorDesc,nbDimsRequested,&dataType,nbDims,dimA,strideA);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnDataType_t>(dataType);
        out->Add<int>(nbDims);
        out->Add<int>(dimA);
        out->Add<int>(strideA);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    cout << "DEBUG - cudnnGetTensorNdDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(GetTensorSizeInBytes){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetTensorSizeInBytes"));

   cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   size_t *size = in->Get<size_t>();

   cudnnStatus_t cs = cudnnGetTensorSizeInBytes(tensorDesc, size);
  
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
        out->Add<size_t>(size);
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   cout << "DEBUG - cudnnGetTensorSizeInBytes Executed"<<endl;
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyTensorDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyTensorDescriptor"));

    cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnStatus_t cs = cudnnDestroyTensorDescriptor(tensorDesc);
    cout << "DEBUG - DestroyTensorDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(InitTransformDest){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("InitTransformDest"));
    
    cudnnTensorTransformDescriptor_t transformDesc = (cudnnTensorTransformDescriptor_t)in->Get<long long int >();
    cudnnTensorDescriptor_t srcDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnTensorDescriptor_t destDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    size_t *destSizeInBytes = in->Get<size_t>();
  
    cudnnStatus_t cs = cudnnInitTransformDest(transformDesc, srcDesc, destDesc, destSizeInBytes);
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<size_t>(destSizeInBytes);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    cout << " DEBUG - cudnnInitTransformDest Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateTensorTransformDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateTensorTransformDescriptor"));

    cudnnTensorTransformDescriptor_t transformDesc;
    
    cudnnStatus_t cs = cudnnCreateTensorTransformDescriptor(&transformDesc);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnTensorTransformDescriptor_t>(transformDesc);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    cout << " DEBUG - cudnnCreateTensorTransformDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetTensorTransformDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetTensorTransformDescriptor"));
    
    cudnnTensorTransformDescriptor_t transformDesc = (cudnnTensorTransformDescriptor_t)in->Get<long long int>();
    uint32_t nbDims = in->Get<uint32_t>();
    cudnnTensorFormat_t destFormat = in->Get<cudnnTensorFormat_t>();
    int32_t *padBeforeA = in->Get<int32_t>();
    int32_t *padAfterA = in->Get<int32_t>();
    uint32_t *foldA = in->Get<uint32_t>();
    cudnnFoldingDirection_t direction = in->Get<cudnnFoldingDirection_t>();

    cudnnStatus_t cs = cudnnSetTensorTransformDescriptor(transformDesc, nbDims, destFormat, padBeforeA, padAfterA, foldA, direction); 
    cout << "DEBUG - cudnnSetTensorTransformDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetTensorTransformDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetTensorTransformDescriptor"));

    cudnnTensorTransformDescriptor_t transformDesc = (cudnnTensorTransformDescriptor_t)in->Get<long long int >();
    uint32_t nbDimsRequested = in->Get<uint32_t>();
    cudnnTensorFormat_t destFormat;
    int32_t padBeforeA;
    int32_t padAfterA;
    uint32_t foldA;
    cudnnFoldingDirection_t direction;
   
    cudnnStatus_t cs = cudnnGetTensorTransformDescriptor(transformDesc, nbDimsRequested, &destFormat, &padBeforeA, &padAfterA, &foldA, &direction);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnTensorFormat_t>(destFormat);
        out->Add<int32_t>(padBeforeA);
        out->Add<int32_t>(padAfterA);
        out->Add<uint32_t>(foldA);
        out->Add<cudnnFoldingDirection_t>(direction);   
    } catch(string e){
        log4cplus_debug(logger, e);
        return make_shared<Result>(cs);
    }
    cout << "DEBUG - cudnnGetTensorTransformDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyTensorTransformDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyTensorTransformDescriptor"));
    
    cudnnTensorTransformDescriptor_t transformDesc = (cudnnTensorTransformDescriptor_t)in->Get<long long int>();
    
    cudnnStatus_t cs = cudnnDestroyTensorTransformDescriptor(transformDesc);
    cout << " DEBUG - cudnnDestroyTensorTransformDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(TransformTensor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("TransformTensor"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    void * alpha = in->Assign<void>();
    cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void * x = in->Assign<void>();
    void * beta = in->Assign<void>();
    cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void * y = in->Assign<void>();

    cudnnStatus_t cs = cudnnTransformTensor(handle,alpha,xDesc,x,beta,yDesc,y);
    cout << "DEBUG - cudnnTransformTensor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(TransformTensorEx){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("TransformTensorEx"));
   
   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   void *alpha = in->Assign<void>();
   cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *x = in->Assign<void>();
   void *beta = in->Assign<void>();
   cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<cudnnTensorDescriptor_t>();
   void *y = in->Assign<void>();
  
   cudnnStatus_t cs = cudnnTransformTensorEx(handle, alpha, xDesc, x, beta, yDesc, y);
   
   cout << "DEBUG - cuddTransformTensorEx Executed"<<endl;
   return std::make_shared<Result>(cs);
}

/* NON SONO SICURO DI QUESTA FUNZIONE DA FAR VEDERE A MONTELLA!!! */
CUDNN_ROUTINE_HANDLER(GetFoldedConvBackwardDataDescriptors){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFoldedConvBackwardDataDescriptors"));
   
   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   cudnnTensorDescriptor_t diffDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
   cudnnTensorDescriptor_t gradDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   cudnnTensorFormat_t transformFormat = in->Get<cudnnTensorFormat_t>();
   cudnnFilterDescriptor_t foldedFilterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   cudnnTensorDescriptor_t paddedDiffDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   cudnnConvolutionDescriptor_t foldedConvDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
   cudnnTensorDescriptor_t foldedGradDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   cudnnTensorTransformDescriptor_t filterFoldTransDesc = (cudnnTensorTransformDescriptor_t)in->Get<long long int>();
   cudnnTensorTransformDescriptor_t diffPadTransDesc = (cudnnTensorTransformDescriptor_t)in->Get<long long int>();
   cudnnTensorTransformDescriptor_t gradFoldTransDesc = (cudnnTensorTransformDescriptor_t)in->Get<long long int>();
   cudnnTensorTransformDescriptor_t gradUnfoldTransDesc = (cudnnTensorTransformDescriptor_t)in->Get<long long int>();


   cudnnStatus_t cs = cudnnGetFoldedConvBackwardDataDescriptors
}

CUDNN_ROUTINE_HANDLER(AddTensor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("AddTensor"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    void * alpha = in->Assign<void>();
    cudnnTensorDescriptor_t aDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void * A = in->Assign<void>();
    void * beta = in->Assign<void>();
    cudnnTensorDescriptor_t cDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void * C = in->Assign<void>();

    cudnnStatus_t cs = cudnnAddTensor(handle,alpha,aDesc,A,beta,cDesc,C);
    cout << "DEBUG - cudnnAddTensor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(CreateOpTensorDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateOpTensorDescriptor"));
   
    cudnnOpTensorDescriptor_t opTensorDesc;
    
    cudnnStatus_t cs = cudnnCreateOpTensorDescriptor(&opTensorDesc);
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnOpTensorDescriptor_t>(opTensorDesc);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    cout << " DEBUG - cudnnCreateOpTensorDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetOpTensorDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetOpTensorDescriptor"));

    cudnnOpTensorDescriptor_t opTensorDesc = (cudnnOpTensorDescriptor_t)in->Get<long long int>();
    cudnnOpTensorOp_t opTensorOp = in->Get<cudnnOpTensorOp_t>();
    cudnnDataType_t opTensorCompType = in->Get<cudnnDataType_t>();
    cudnnNanPropagation_t opTensorNanOpt = in->Get<cudnnNanPropagation_t>();

   cudnnStatus_t cs = cudnnSetOpTensorDescriptor(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);
   
   cout << " DEBUG - cudnnSetOpTensorDescriptor Executed"<<endl;
   return std::make_shared<Result>(cs); 
}

CUDNN_ROUTINE_HANDLER(GetOpTensorDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetOpTensorDescriptor"));
   
   cudnnOpTensorDescriptor_t opTensorDesc = (cudnnOpTensorDescriptor_t)in->Get<long long int>();
   cudnnOpTensorOp_t opTensorOp;
   cudnnDataType_t opTensorCompType;
   cudnnNanPropagation_t opTensorNanOpt;

   cudnnStatus_t cs = cudnnGetOpTensorDescriptor(opTensorDesc, &opTensorOp, &opTensorCompType, &opTensorNanOpt);
   
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<cudnnOpTensorOp_t>(opTensorOp);
       out->Add<opTensorCompType>(opTensorCompType);
       out->Add<cudnnNanPropagation_t>(opTensorNanOpt);
   } catch(string e)
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   cout << " DEBUG - cudnnGetOpTensorDescriptor"<<endl;
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyOpTensorDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyOpTensorDescriptor"));

   cudnnOpTensorDescriptor_t opTensorDesc = (cudnnOpTensorDescriptor_t)in->Get<long long int>();
   
   cudnnStatus_t cs = cudnnDestroyOpTensorDescriptor(opTensorDesc);
   
   cout << "DEBUG - cudnnDestroyOpTensorDescriptor Executed"<<endl;
   return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(OpTensor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("OpTensor"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnOpTensorDescriptor_t opTensorDesc = (cudnnOpTensorDescriptor_t)in->Get<long long int>();
    const void * alpha1 = in->Assign<void>();
    cudnnTensorDescriptor_t aDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void * A = in->Assign<void>();
    void * alpha2 = in->Assign<void>();
    cudnnTensorDescriptor_t bDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void * B = in->Assign<void>();
    void * beta = in->Assign<void>();
    cudnnTensorDescriptor_t cDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void * C = in->Assign<void>();

    cudnnStatus_t cs = cudnnOpTensor(handle,opTensorDesc,alpha1,aDesc,A,alpha2,bDesc,B,beta,cDesc,C);
    cout << "DEBUG - cudnnOpTensor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(CreateReduceTensorDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateReduceTensorDescriptor"));

    cudnnReduceTensorDescriptor_t reduceTensorDesc;
   
    cudnnStatus_t cs = cudnnCreateReduceTensorDescriptor(& reduceTensorDesc);
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnReduceTensorDescriptor_t>(reduceTensorDesc);
    } catch(string e){
        LOG4CPLUS_TEXT(logger, e);
        return std::make_shared<Result>(cs);
    }
    cout << " DEBUG - cudnnCreateReduceTensorDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}
 
CUDNN_ROUTINE_HANDLER(SetReduceTensorDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetReduceTensorDescriptor"));

    cudnnReduceTensorDescriptor_t reduceTensorDesc = (cudnnReduceTensorDescriptor_t)in->Get<long long int>();
    cudnnReduceTensorOp_t reduceTensorOp = in->Get<cudnnReduceTensorOp_t>();
    cudnnDataType_t reduceTensorCompType = in->Get<cudnnDataType_t>();
    cudnnNanPropagation_t reduceTensorNanOpt = in->Get<cudnnNanPropagation_t>();
    cudnnReduceTensorIndices_t reduceTensorIndices = in->Get<cudnnReduceTensorIndices_t>();
    cudnnIndicesType_t reduceTensorIndicesType = in->Get<cudnnIndicesType_t>();

    cudnnStatus_t cs = cudnnSetReduceTensorDescriptor(reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType);
    cout << " DEBUG - cudnnSetReduceTensorDescriptor"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetReduceTensorDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetReduceTensorDescriptor"));
 
    cudnnReduceTensorDescriptor_t reduceTensorDesc = (cudnnReduceTensorDescriptor_t)in->Get<long long int>();
    cudnnReduceTensorOp_t reduceTensorOp;
    cudnnDataType_t reduceTensorCompType;
    cudnnNanPropagation_t reduceTensorNanOpt;
    cudnnReduceTensorIndices_t reduceTensorIndices;
    cudnnIndicesType_t reduceTensorIndicesType;
  
    cudnnStatus_t cs = cudnnGetReduceTensorDescriptor(reduceTensorDesc, &reduceTensorOp, &reduceTensorCompType, &reduceTensorNanOpt, &reduceTensorIndices, &reduceTensorIndicesType);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnReduceTensorOp_t>(reduceTensorOp);
        out->Add<cudnnDataType_t>(reduceTensorCompType);
        out->Add<cudnnNanPropagation_t>(reduceTensorNanOpt);
        out->Add<cudnnReduceTensorIndices_t>(reduceTensorIndices);
        out->Add<cudnnIndicesType_t>(reduceTensorIndicesType);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    cout << " DEBUG - cudnnGetReduceTensorDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyReduceTensorDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyReduceTensorDescriptor"));

    cudnnReduceTensorDescriptor_t reduceTensorDesc = (cudnnReduceTensorDescriptor_t)in->Get<long long int>();
    cudnnStatus_t cs = cudnnDestroyReduceTensorDescriptor(reduceTensorDesc);
    cout << "DEBUG - cudnnDestroyReduceTensorDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetReductionIndicesSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetReductionIndicesSize"));
    
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnReduceTensorDescriptor_t reduceTensorDesc = (cudnnReduceTensorDescriptor_t)in->Get<long long int>()
    cudnnTensorDescriptor_t aDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnTensorDescriptor_t cDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    size_t *sizeInBytes;
   
    cudnnStatus_t cs = cudnnGetReductionIndicesSize(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
     
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<size_t>(sizeInBytes);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
   }
   cout << " DEBUG - cuddGetReductionIndicesSize Executed"<<endl;
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetReductionWorkspaceSize){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetReductionWorkspaceSize"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnReduceTensorDescriptor_t reduceTensorDesc = (cudnnReduceTensorDescriptor_t)in->Get<long long int>();
   cudnnTensorDescriptor_t aDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   cudnnTensorDescriptor_t cDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   size_t *sizeInBytes;

   cudnnStatus_t cs = cudnnGetReductionWorkspaceSize(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
    
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Get<size_t>(sizeInBytes);
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   cout << " DEBUG - cudnnGetReductionWorkspaceSize Executed"<<endl;
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(ReduceTensor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ReduceTensor"));
 
   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>(); //INPUT
   cudnnReduceTensorDescriptor_t reduceTensorDesc = (cudnnReduceTensorDescriptor_t)in->Get<long long int>(); //INPUT
   void *indices = in->Assign<void>();  //OUTPUT
   size_t indicesSizeInBytes = in->Get<size_t>(); //INPUT
   void *workspace = in->Assign<void>(); //INPUT
   size_t workspaceSizeInBytes = in->Get<size_t>(); //INPUT
   void *alpha = in->Assign<void>(); //INPUT
   cudnnTensorDescriptor_t aDesc = (cudnnTensorDescriptor_t)in->Get<long long int>(); //INPUT
   void *A = in->Assign<void>(); //INPUT
   void *beta = in->Assign<void>(); //INPUT
   cudnnTensorDescriptor_t cDesc = (cudnnTensorDescriptor_t)in->Get<long long int> //INPUT
   void *C = in->Assign<void>(); //INPUT/OUTPUT

   cuddStatus_t cs = cudnnReduceTensor(handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace, workspaceSizeInBytes, alpha, cudnnTensorDescriptor_t, A, beta, cDesc, C);
   
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<void>(indicies);
       out->Add<void>(C);
   } catch(string e){
      LOG4CPLUS_DEBUG(logger, e);
      return std::make_shared<Result>(cs);
   }
   cout << " DEBUG - cudnnReduceTensor Executed"<<endl;
   return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetTensor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetTensor"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void * y = in->Assign<void>();
    void * valuePtr = in->Assign<void>();

    cudnnStatus_t cs = cudnnSetTensor(handle,yDesc,y,valuePtr);
    cout << "DEBUG - cudnnSetTensor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(ScaleTensor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ScaleTensor"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    const cudnnTensorDescriptor_t yDesc = (const cudnnTensorDescriptor_t)in->Get<long long int>();
    void * y = in->Assign<void>();
    void * alpha = in->Assign<void>();

    cudnnStatus_t cs = cudnnScaleTensor(handle,yDesc,y,alpha);
    cout << "DEBUG - cudnnScaleTensor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(CreateFilterDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS("CreateFilterDescriptor"));
    
    cudnnFilterDescriptor_t filterDesc;

    cudnnStatus_t cs = cudnnCreateFilterDescriptor(&filterDesc);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnFilterDescriptor_t>(filterDesc);
   } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
   }
   cout << " DEBUG - cudnnCreateFilterDescriptor Executed"<<endl;
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetFilter4dDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilter4dDescriptor"));
   
   cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
   cudnnTensorFormat_t format = in->Get<cudnnTensorFormat_t>();
   int k = in->Get<int>();
   int c = in->Get<int>();
   int h = in->Get<int>();
   int w = in->Get<int>();

   cudnnStatus_t cs = cudnnSetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w);
   
   cout << " DEBUG - cudnnSetFilter4dDescriptor Executed"<<endl;
   return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetFilter4dDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilter4dDescriptor"));

   cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   cudnnDataType_t dataType;
   cudnnTensorFormat_t format;
   int k;
   int c;
   int h;
   int w;

   cudnnStatus_t cs = cudnnGetFilter4dDescriptor(filterDesc, &dataTYpe, &format, &k, &c, &h, &w);
   
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<cudnnDataType_t>(dataType);
       out->Add<cudnnTensorFormat_t>(format);
       out->Add<int>();
       out->Add<int>();
       out->Add<int>();
       out->Add<int>();
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   cout << " DEBUG - cudnnGetFilter4dDescriptor Executed"<<endl;
   return std::make_shared<Result>(cs, out);
}

#if CUDNN_VERSION < 6000
CUDNN_ROUTINE_HANDLER(SetFilter4dDescriptor_v3){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilter4dDescriptor_v3"));

    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType = (cudnnDataType_t) in->Get<long long int>();

    int k = in->Get<int>();
    int c = in->Get<int>();
    int h = in->Get<int>();
    int w = in->Get<int>();

    cudnnStatus_t cs = cudnnSetFilter4dDescriptor_v3(filterDesc,dataType,k,c,h,w);
    cout << "DEBUG - cudnnSetFilter4dDescriptor_v3 Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetFilter4dDescriptor_v3){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilter4dDescriptor_v3"));

    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType;

    int k,c,h,w;

    cudnnStatus_t cs = cudnnGetFilter4dDescriptor_v3(filterDesc,&dataType,&k,&c,&h,&w);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try{
        out->Add<long long int>((long long int)dataType);
        out->Add<int>(k);
        out->Add<int>(c);
        out->Add<int>(h);
        out->Add<int>(w);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    cout << "DEBUG - cudnnGetFilter4dDescriptor_v3 Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(SetFilter4dDescriptor_v4){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilter4dDescriptor_v4"));

    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType = (cudnnDataType_t) in->Get<long long int>();
    cudnnTensorFormat_t  format = (cudnnTensorFormat_t) in->Get<long long int>();

    int k = in->Get<int>();
    int c = in->Get<int>();
    int h = in->Get<int>();
    int w = in->Get<int>();

    cudnnStatus_t cs = cudnnSetFilter4dDescriptor_v4(filterDesc,dataType,format,k,c,h,w);
    cout << "DEBUG - cudnnSetFilter4dDescriptor_v4 Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetFilter4dDescriptor_v4){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilter4dDescriptor_v4"));

    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType;
    cudnnTensorFormat_t  format;

    int k,c,h,w;

    cudnnStatus_t cs = cudnnGetFilter4dDescriptor_v4(filterDesc,&dataType,&format,&k,&c,&h,&w);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try{
        out->Add<long long int>((long long int)dataType);
        out->Add<long long int>((long long int)format);
        out->Add<int>(k);
        out->Add<int>(c);
        out->Add<int>(h);
        out->Add<int>(w);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    cout << "DEBUG - cudnnGetFilter4dDescriptor_v4 Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}
#endif

CUDNN_ROUTINE_HANDLER(SetFilterNdDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilterNdDescriptor"));
    
    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
    cudnnTensorFormat_t format = in->Get<cudnnTensorFormat_t>();
    int nbDims = in->Get<int>();
    int *filterDimA = in>Assign<int>();
    
    cudnnStatus_t cs = cudnnSetFilterNdDescriptor(filterDesc, dataType, format, nbDims, filterDimA);
    
    cout << " DEBUG - cudnnSetFilterNdDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetFilterNdDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilterNdDescriptor"));

    cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    int nbDimsRequested = in->Get<int>();
    cudnnDataType_t *dataType = in->Assign<cudnnDataType_t>();
    int *nbDims = in->Assign<int>();
    int *filterDimA = in->Assign<int>();

    cudnnTensorFormat_t  format;

    cudnnStatus_t cs = cudnnGetFilterNdDescriptor(wDesc,nbDimsRequested,dataType,&format,nbDims,filterDimA);

    std:shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try{
        out->Add<long long int>(format);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    cout << "DEBUG - cudnnGetFilterNdDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}

#if CUDNN_VERSION < 6000
CUDNN_ROUTINE_HANDLER(SetFilterNdDescriptor_v3){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilterNdDescriptor_v3"));

    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType = (cudnnDataType_t) in->Get<long long int>();

    int nbDims = in->Get<int>();
    int * filterDimA = in->Assign<int>();

    cudnnStatus_t cs = cudnnSetFilterNdDescriptor_v3(filterDesc,dataType,nbDims,filterDimA);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add<long long int>((long long int)filterDesc);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    cout << "DEBUG - cudnnSetFilterNdDescriptor_v3 Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(GetFilterNdDescriptor_v3){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilterNdDescriptor"));

    cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    int nbDimsRequested = in->Get<int>();
    cudnnDataType_t *dataType = in->Assign<cudnnDataType_t>();
    int *nbDims = in->Assign<int>();
    int *filterDimA = in->Assign<int>();


    cudnnStatus_t cs = cudnnGetFilterNdDescriptor_v3(wDesc,nbDimsRequested,dataType,nbDims,filterDimA);
    cout << "DEBUG - cudnnGetFilterNdDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetFilterNdDescriptor_v4){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilterNdDescriptor_v4"));

    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType = (cudnnDataType_t) in->Get<long long int>();
    cudnnTensorFormat_t  format = (cudnnTensorFormat_t) in->Get<long long int>();

    int nbDims = in->Get<int>();
    int * filterDimA = in->Assign<int>();

    cudnnStatus_t cs = cudnnSetFilterNdDescriptor_v4(filterDesc,dataType,format,nbDims,filterDimA);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add<long long int>((long long int)filterDesc);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    cout << "DEBUG - cudnnSetFilterDescriptor_v4 Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(GetFilterNdDescriptor_v4){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilterNdDescriptor_v4"));

    cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    int nbDimsRequested = in->Get<int>();
    cudnnDataType_t *dataType = in->Assign<cudnnDataType_t>();
    int *nbDims = in->Assign<int>();
    int *filterDimA = in->Assign<int>();

    cudnnTensorFormat_t  format;

    cudnnStatus_t cs = cudnnGetFilterNdDescriptor_v4(wDesc,nbDimsRequested,dataType,&format,nbDims,filterDimA);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try{
        out->Add<long long int>(format);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    cout << "DEBUG - cudnnGetFilterDescriptor_v4 Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}
#endif

CUDNN_ROUTINE_HANDLER(GetFilterSizeInBytes){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilterSizeInBytes"));
    
    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();

    size_t *size = in->Get<size_t>();
    
    cudnnStatus_t cs = cudnnGetFilterSizeInBytes(filterDesc, size);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<size_t>(size);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    cout << " DEBUG - cudnnGetFilterSizeInBytes Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyFilterDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestoryFilterDescriptor"));

    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();

    cudnnStatus_t cs = cudnnDestroyFilterDescriptor(filterDesc);
    cout << "DEBUG - cudnnDestroyFilterDescriptor Executed"<<endl;
    return make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(TransformFilter){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("TransformFilter"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>(); //INPUT
   cudnnTensorTransformDescriptor_t transDesc = (cudnnTensorTransformDescriptor_t)in->Get<long long int>(); //INPUT
   void *alpha = in->Assign<void>(); //INPUT
   cudnnFilterDescriptor_t srcDesc = (cudnnFilterDescriptor_t)in->Get<long long int>(); //INPUT
   void *srcData = in->Assign<void>(); //INPUT
   void *beta = in->Assign<void>(); //INPUT
   cudnnFilterDescriptor_t destDesc = (cudnnFilterDescriptor_t)in->Get<long long int>(); //INPUT
   void *destData = in->Assign<void>(); //OUTPUT
   
   cuddStatus_t cs = cudnnTransformFilter(handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData);
   
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<void>(destData);
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   cout << " DEBUG - cudnnTransformFilter Executed"<<endl;
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(ReorderFilterAndBias){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ReorderFilterAndBias"));
   
   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   cudnnReorderType_t reorderType = in->Get<cudnnReorderType_t>();
   void *filterData = in->Assign<void>();
   void *reorderedFilterData = in->Assign<void>();
   int reorderBias = in->Get<int>();
   void *biasData = in->Assign<void>();
   void *reorderedBiasData = in->Assign<void>();


   cudnnStatus_t cs = cudnnReorderFilterAndBias(handle, filterDesc, reorderType, filterData, reorderedFilterData, reorderBias, reorderBiasbiasData, biasData);
   cout << " DEBUG - cudnnReorderFilterAndBias Executed"<<endl;
   return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(CreateConvolutionDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateConvolutionDescriptor"));

    cudnnConvolutionDescriptor_t convDesc;
    cudnnStatus_t cs = cudnnCreateConvolutionDescriptor(&convDesc);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add<cudnnConvolutionDescriptor_t>(convDesc);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    cout << "DEBUG - cudnnCreateConvolutionDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(SetConvolutionMathType){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetConvolutionMathType"));
    
    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
    cudnnMathType_t mathType = in->Get<cudnnMathType_t>();

    cudnnStatus_t cs = cudnnSetConvolutionMathType(convDesc, mathType);

    cout << " DEBUG - cudnnSetConvolutionMathType Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetConvolutionMathType){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionMathType"));

    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
    cudnnMathType_t mathType;

    cudnnStatus_t cs = cudnnGetConvolutionMathType(convDesc, &mathType);

    std::shared_ptr<Buffer>() out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnMathType_t>(mathType);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    cout << " DEBUG - cudnnGetConvolutionMathType Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetConvolutionGroupCount){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetConvolutionGroupCount"));
    
    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
    int groupCount = in->Get<int>();

    cudnnStatus_t cs = cudnnSetConvolutionGroupCount(convDesc, groupCount);
    
    cout << " DEBUG - cudnnSetConvolutionGroupCount Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetConvolutionGroupCount){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionGroupCount"));
   
    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
    int groupCount;

    cudnnStatus_t cs = cudnnGetConvolutionGroupCount(convDesc, &groupCount);
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<int>(groupCount);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    cout << " DEBUG - cudnnGetConvolutionGroupCount Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetConvolutionReorderType){
    Logger logger = Logger::getInstance(LOG4CPLUS("SetConvolutionReorderType"));
    
    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
    cudnnReorderType_t reorderType = in->Get<cudnnReorderType_t>();
    
    cudnnStatus_t cs = cuddnSetConvolutionReorderType(convDesc, reorderType);
  
    cout << " DEBUG - cudnnSetConvolutionReorderType Executed"<<endl;
    return std::make_shared<Result>(cs); 
}

CUDNN_ROUTINE_HANDLER(GetConvolutionReorderType){
    Logget logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionReorderType"));

    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
    cudnnReorderType_t reorderType;

    cudnnStatus_t cs = cudnnGetConvolutionReorderType(convDesc, &reorderType);
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnReorderType_t>(reorderType);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    cout << " DEBUG - cudnnGetConvolutionReorderType Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetConvolution2dDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetConvolution2dDescriptor"));

    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
    int padh = in->Get<int>();
    int padw = in->Get<int>();
    int u = in->Get<int>();
    int v = in->Get<int>();
    int upscalex = in->Get<int>();
    int upscaley = in->Get<int>();
    cudnnConvolutionMode_t mode = in->BackGet<cudnnConvolutionMode_t>();

    cudnnStatus_t cs = cudnnSetConvolution2dDescriptor(convDesc,padh,padw,u,v,upscalex,upscaley,mode,cudnnDataType_t::CUDNN_DATA_FLOAT);


    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try{
        out->Add<cudnnConvolutionDescriptor_t>(convDesc);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    cout << "DEBUG - cudnnSetConvolution2dDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(GetConvolution2dDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolution2dDescriptor"));

    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
    int padh,padw,u,v,upscalex,upscaley;
    cudnnConvolutionMode_t mode;
    cudnnDataType_t computeType = CUDNN_DATA_FLOAT;

    cudnnStatus_t cs = cudnnGetConvolution2dDescriptor(convDesc,&padh,&padw,&u,&v,&upscalex,&upscaley,&mode,&computeType);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try{
        out->Add(padh);
        out->Add(padw);
        out->Add(u);
        out->Add(v);
        out->Add(upscalex);
        out->Add(upscaley);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    cout << "DEBUG - cudnnGetConvolution2dDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(GetConvolution2dForwardOutputDim){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolution2dForwardOutputDim"));

    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
    cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t) in->Get<long long int>();
    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t) in->Get<long long int>();

    int n,c,h,w;

    cudnnStatus_t cs = cudnnGetConvolution2dForwardOutputDim(convDesc,tensorDesc,filterDesc,&n,&c,&h,&w);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add(n);
        out->Add(c);
        out->Add(h);
        out->Add(w);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    cout << "DEBUG - cudnnGetConvolution2dForwardOutputDim Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(SetConvolutionNdDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetConvolutionNdDescriptor"));

    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
    int arrayLength = in->Get<int>();
    int *padA = in->Assign<int>();
    int *filterStrideA = in->Assign<int>();
    int *dilationA = in->Assign<int>();
    cudnnConvolutionMode_t mode = in->Get<cudnnConvolutionMode_t>();
    cudnnDataType_t computeType = in->Get<cudnnDataType_t>();
   
    cudnnStatus_t cs = cudnnSetConvolutionNdDescriptor(convDesc, arrayLength, padA, filterStrideA, dilationA, mode, computeType);
  
    cout << " DEBUG - cudnnSetConvolutionNdDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs);
}


CUDNN_ROUTINE_HANDLER(GetConvolutionNdDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionNdDescriptor"));

    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>(); //INPUT/OUTPUT
    int arrayLengthRequested = in->Get<int>(); //INPUT
    int arrayLength; //OUTPUT
    int *padA = in->Assign<int>(); //OUTPUT
    int *strideA = in->Assign<int>(); //OUTPUT
    int *dilationA = in->Assign<int>(); //OUTPUT
    cudnnConvolutionMode_t mode; //OUTPUT
    cudnnDataType_t computeType; //OUTPUT

    cudnnStatus_t cs = cudnnGetConvolutionNdDescriptor(&convDesc, arrayLengthRequested, &arrayLength, padA, strideA, dilationA, &mode, &computeType);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
         out->Add<cudnnConvolutionDescriptor_t>(convDesc);
         out->Add<int>(arrayLength);
         out->Add<int>(padA);
         out->Add<int>(strideA);
         out->Add<int>(dilationA);
         out->Add<cudnnConvolutionMode_t>(mode);
         out->Add<cudnnDataType_t>(computeType);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    cout << " DEBUG - cudnnGetConvolutionNdDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetConvolutionNdForwardOutputDim){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionNdForwardOutputDim"));

    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
    cudnnTensorDescriptor_t inputTensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    int nbDims = in->Get<int>();
    int *tensorOutputDimA = in->Assign<int>();

    cudnnStatus_t cs = cudnnGetConvolutionNdForwardOutputDim(convDesc, inputTensorDesc, filterDesc, nbDims, tensorOutputDimA);
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<int>(tensorOutputDimA);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    cout << " DEBUG - cudnnGetConvolutionNdForwardOutputDim Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyConvolutionDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyConvolutionDescriptor"));

    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
    
    cudnnStatus_t cs = cudnnDestroyConvolutionDescriptor(convDesc);
    
    cout << " DEBUG - cudnnDestroyConvolutionDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs); 
}

CUDNN_ROUTINE_HANDLER(GetConvolutionForwardAlgorithmMaxCount){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionForwardAlgorithmMaxCount"));
    
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    int count;
   
    cudnnStatus_t cs = cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &count);
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<int>(count);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    cout << " DEBUG - cudnnGetConvolutionForwardAlgorithmMaxCount Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(FindConvolutionForwardAlgorithm){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FindConvolutionForwardAlgorithm"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
    cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    int requestAlgoCount = in->Get<int>();
    int requestedAlgCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults;


    cudnnStatus_t cs = cudnnFindConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, requestAlgoCount, &requestedAlgCount, &perfResults);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<int>(requestedAlgCount);
        out->Add<cudnnConvolutionFwdAlgoPerf_t>(perfResults);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    cout << " DEBUG - cudnnFindConvolutionForwardAlgorithm Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(FindConvolutionForwardAlgorithmEx){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FindConvolutionForwardAlgorithmEx"));
    
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>(); //INPUT
    cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor)in->Get<long long int>(); //INPUT
    void *x = in->Assign<void>(); //INPUT
    cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>(); //INPUT
    void *w = in->Assign<void>(); //INPUT
    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>(); //INPUT
    cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>(); //INPUT
    void *y = in->Assign<void>(); //INPUT/OUTPUT
    int requestAlgoCount = in->Get<int>(); //INPUT
    int returnedAlgoCount; //OUTPUT
    cudnnConvolutionFwdAlgoPerf_t perfResults; //OUTPUT
    void *workSpace = in->Assign<void>(); //INPUT
    size_t workSpaceSizeInBytes = in->Get<size_t>(); //INPUT

    cudnnStatus_t cs = cudnnFindConvolutionForwardAlgorithmEx(handle, xDesc, x, w, convDesc, yDesc, y, requestAlgoCount, &returnedAlgoCount, &perfResults, workSpace, workSpaceSizeInBytes);
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<void>(y);
        out->Add<int>(returnedAlgoCount);
        out->Add<cudnnConvolutionFwdAlgoPerf_t>(perfResults);       
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std:make_shared<Result>(cs);
    }
    cout << " DEBUG - cudnnFindConvolutionForwardAlgorithmEx Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetConvolutionForwardAlgorithm){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionForwardAlgorithm"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
   cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   cudnnConvolutionFwdPreference_t preference = in->Get<cudnnConvolutionFwdPreference_t>();
   size_t memoryLimitInBytes = in->Get<size_t>();
   cudnnConvolutionFwdAlgo_t algo;

   cudnnStatus_t cs = cudnnGetConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, preference, memoryLimitInBytes, &algo);
   
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<cudnnConvolutionFwdAlgo_t>(algo);
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   cout << " DEBUG - cudnnGetConvolutionForwardAlgorithm Executed"<<endl;
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetConvolutionForwardAlgorithm_v7){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionForwardAlgorithm_v7"));
   
   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnTensorDescriptor_t srcDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
   cudnnTensorDescriptor_t destDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   int requestedAlgoCount = in->Get<int>();
   cudnnConvolutionFwdAlgoPerf_t perfResults;

   cudnnStatus_t cs = cudnnGetConvolutionForwardAlgorithm_v7(handle, srcDesc, filterDesc, convDesc, destDesc, requestedAlgoCount, &perfResults);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<cudnnConvolutionFwdAlgoPerf_t>(perfResults);
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   cout << " DEBUG - cudnnGetConvolutionForwardAlgorithm_v7 Executed"<<endl;
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetConvolutionForwardWorkspaceSize){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionForwardWorkspaceSize"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
   cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   cudnnConvolutionFwdAlgo_t algo = in->Get<cudnnConvolutionFwdAlgo_t>();
   size_t sizeInBytes;

   cudnnStatus_t cs = cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc, algo, &sizeInBytes);
   
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<size_t>(sizeInBytes);
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   cout << " DEBUG - cudnnGetConvolutionForwardWorkspaceSize Executed"<<endl;
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(ConvolutionForward){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ConvolutionForward"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   void *alpha = in->Assign<void>();
   cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *x = in->Get<void>();
   cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   void *w = in->Assign<void>();
   cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
   cudnnConvolutionFwdAlgo_t algo = in->Get<cudnnConvolutionFwdAlgo_t>();
   void *workspace = in->Assign<void>();
   size_t workspaceSizeInBytes = in->Get<size_t>();
   void *beta = in->Assign<void>();
   cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *y = in->Assign<void>();

   cudnnStatus_t cs = cudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workspace, workspaceSizeInBytes, beta, yDesc, y);
   
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<void>(y);
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   cout << " DEBUG - cudnnConvolutionForward Executed"<<endl;
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(ConvolutionBiasActivationForward){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ConvolutionBiasActivationForward"));
  
   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   void alpha1 = in->Assign<void>();
   cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *x = in->Assign<void>();
   cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   void *w = in->Assign<void>();
   cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
   cudnnConvolutionFwdAlgo_t algo = in->Get<cudnnConvolutionFwdAlgo_t>();
   void *workspace = in->Assign();
   size_t workSpaceSizeInBytes = in->Get<size_t>();
   void *alpha2 = in->Assign<void>();
   cudnnTensorDescriptor_t zDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *z = in->Get<void>();
   cudnnTensorDescriptor_t biasDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *bias = in->Assign<void>();
   cudnnActivationDescriptor_t activationDesc = (cudnnActivationDescriptor_t)in->Assign<long long int>();
   cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *y = in->Assign<void>();

   cudnnStatus_t cs = cudnnConvolutionBiasActivationForward(handle, alpha1, xDesc, x, wDesc, w, convDesc, algo, workspace, workSpaceSizeInBytes, alpha2, zDesc, z, biasDesc, bias, activationDesc, yDesc, y);
   
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<void>(y);
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   cout << " DEBUG - cudnnConvolutionBiasActivationForward Executed"<<endl;
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(ConvolutionBackwardBias){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ConvolutionBackwardBias"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   void *alpha = in->Assign<void>();
   cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t )in->Get<long long int>();
   void *dy = in->Assign<void>();
   void *beta = in->Assign<void>();
   cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *db = in->Assign<void>();

   cudnnStatus_t cs = cudnnConvolutionBackwardBias(handle, alpha, dyDesc, dy, beta, yDesc, db);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<void>(db);
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   cout << " DEBUG - cudnnConvolutionBackwardBias Executed"<<endl;
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardFilterAlgorithmMaxCount){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionBackwardFilterAlgorithmMaxCount"));
  
   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   int count = in->Get<int>;

   cudnnStatus_t cs = cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle, &count);
 
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<int>(count);
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   cout << " DEBUG - cudnnGetConvolutionBackwardFilterAlgorithmMaxCount Executed"<<endl;
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(FindConvolutionBackwardFilterAlgorithm){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FindConvolutionBackwardFilterAlgorithm"));
   
   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   cudnnTensorDescriptor_t DyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
   cudnnFilterDescriptor_t dwDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   int requestedAlgoCount = in->Get<int>();
   int returnedAlgoCount;
   cudnnConvolutionBwdFilterAlgoPerf_t perfResults;

   cudnnStatus_t cs = cudnnFindConvolutionBackwardFilterAlgorithm(handle, xDesc, DyDesc, convDesc, dwDesc, requestedAlgoCount, &returnedAlgoCount, &perfResults);
   
   std::shared_pointer<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<int>(returnedAlgoCount);
       out->Add<cudnnConvolutionBwdFilterAlgoPerf_t>(perfResults);
   } catch (string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   cout << " DEBUG - cudnnFindConvolutionBackwardFilterAlgorithm Executed"<<endl;
   return std::make_shared<Result>(cs, out);  
}

CUDNN_ROUTINE_HANDLER(FindConvolutionBackwardFilterAlgorithmEx){
   Logger logger = Logger::getInstance();
}

















































CUDNN_ROUTINE_HANDLER(CreatePoolingDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreatePoolingDescriptor"));
   cudnnPoolingDescriptor_t poolingDesc;
   cudnnStatus_t cs = cudnnCreatePoolingDescriptor(&poolingDesc);
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<cudnnPoolingDescriptor_t>(poolingDesc);
   }catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return make_shared<Result>(CUDNN_STATUS_EXECUTION_FAILED);
   }
   std::cout << "DEBUG - cudnnCreatePoolingDescriptor Executed"<<endl;
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetPooling2dDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetPooling2dDescriptor"));
   
   cudnnPoolingDescriptor_t poolingDesc = (cudnnPoolingDescriptor_t)in->Get<long long int>();
   cudnnPoolingMode_t mode = in->Get<cudnnPoolingMode_t>();
   cudnnNanPropagation_t maxpoolingNanOpt = in->Get<cudnnNanPropagation_t>();

    int windowHeight = in->Get<int>();
    int windowWidth = in->Get<int>();
    int verticalPadding = in->Get<int>();
    int horizontalPadding = in->Get<int>();
    int verticalStride = in->Get<int>();
    int horizontalStride = in->Get<int>();

   cudnnStatus_t cs = cudnnSetPooling2dDescriptor(poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride);
   cout << "DEBUG - cudnnSetPooling2dDescriptor Executed"<<endl;
   return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(CreateActivationDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateActivationDescriptor"));
   cudnnActivationDescriptor_t activationDesc;
   cudnnStatus_t cs = cudnnCreateActivationDescriptor(&activationDesc);
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<cudnnActivationDescriptor_t>(activationDesc);
   } catch (string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(CUDNN_STATUS_EXECUTION_FAILED);
   }
   std::cout << "DEBUG - cudnnCreateActivationDescriptor Executed"<<endl;
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetActivationDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetActivationDescriptor"));
   
   cudnnActivationDescriptor_t activationDesc = (cudnnActivationDescriptor_t)in->Get<long long int>();
   cudnnActivationMode_t mode = in->Get<cudnnActivationMode_t>();
   cudnnNanPropagation_t reluNanOpt = in->Get<cudnnNanPropagation_t>();

   double coef = in->Get<double>();

   cudnnStatus_t cs = cudnnSetActivationDescriptor(activationDesc, mode, reluNanOpt, coef);
   cout << "DEBUG - cudnnSetActivationDescriptor Executed"<<endl;
   return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetActivationDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetActivationDescriptor"));
   cudnnActivationDescriptor_t activationDesc;
   cudnnActivationMode_t mode;
   cudnnNanPropagation_t reluNanOpt;
   double coef;
   cudnnStatus_t cs = cudnnGetActivationDescriptor(activationDesc, &mode, &reluNanOpt, &coef);
   cout <<"DEBUG - cudnnGetActivationDescriptor Executed"<<endl;
   return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(DestroyActivationDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyActivationDescriptor"));
    cudnnActivationDescriptor_t activationDesc;
    cudnnStatus_t cs = cudnnDestroyActivationDescriptor(activationDesc);
    cout <<"DEBUG - cudnnDestroyActivationDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(DestroyFilterDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestoryFilterDescriptor"));

    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();

    cudnnStatus_t cs = cudnnDestroyFilterDescriptor(filterDesc);
    cout << "DEBUG - cudnnDestroyFilterDescriptor Executed"<<endl;
    return make_shared<Result>(cs);
}

