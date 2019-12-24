//THIS IS OUR CUSTOM PLUGIN
#include "maskingPluginLSTM.h"
#include "NvInfer.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdio.h>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::MaskingPlugin;
using nvinfer1::plugin::MaskingPluginCreator;








// plugin specific constants
namespace
{
static const char* MASKING_PLUGIN_VERSION{"1"};
static const char* MASKING_PLUGIN_NAME{"MASKING_layer"};
static const float RPN_STD_SCALING{1.0f};
} // namespace

// Static class fields initialization
//PluginFieldCollection layerPluginCreator::mFC{};
//std::vector<PluginField> layerPluginCreator::mPluginAttributes;





///Serialization/Deserialization APIs: serialize, constructor
//serializing plugin
template <typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

//deserializing plugin
template <typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

size_t layerPlugin::getSerializationSize() const
{
    return sizeof(size_t) * 9 + sizeof(float) * 3 + sizeof(float) * mAnchorSizeNum + sizeof(float) * mAnchorRatioNum;
}

void layerPlugin::serialize(void* buffer) const
{
    char* d = reinterpret_cast<char*>(buffer);
    char* a = d;
    writeToBuffer<size_t>(a, mInputHeight);
    writeToBuffer<size_t>(a, mInputWidth);
    writeToBuffer<size_t>(a, mRpnHeight);
    writeToBuffer<size_t>(a, mRpnWidth);
    writeToBuffer<size_t>(a, mRpnStride);
    writeToBuffer<size_t>(a, mPreNmsTopN);
    writeToBuffer<size_t>(a, mMaxBoxNum);
    writeToBuffer<size_t>(a, mAnchorSizeNum);
    writeToBuffer<size_t>(a, mAnchorRatioNum);
    writeToBuffer<float>(a, mRpnStdScaling);
    writeToBuffer<float>(a, mBboxMinSize);
    writeToBuffer<float>(a, mNmsIouThreshold);

    for (int i = 0; i < mAnchorSizeNum; ++i)
    {
        writeToBuffer<float>(a, mAnchorSizes[i]);
    }

    for (int i = 0; i < mAnchorRatioNum; ++i)
    {
        writeToBuffer<float>(a, mAnchorRatios[i]);
    }

    ASSERT(a == d + getSerializationSize());
}

IPluginV2Ext* MaskingPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed,
    IPluginV2Ext* plugin = new layerPlugin(name, serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}









///Builder configurations APIs: configurePlugin, supportsFormat, constructor, destructor, etc.
MaskingPlugin::MaskingPlugin(const std::string name)
    : mLayerName(name)
{
}

MaskingPlugin::MaskingPlugin(const std::string name, int samples, int timesteps, int features)
    : mLayerName(name)
    , msamples(samples)
    , mtimesteps(timesteps)
    , mfeatures(features)
{
    
}

MaskingPlugin::MaskingPlugin(const std::string name, bool supports_masking, int mask_value, bool compute_output_and_mask_jointly)
    : mLayerName(name)
    , msupports_masking(supports_masking)
    , mmask_value(mask_value)
    , mcompute_output_and_mask_jointly(compute_output_and_mask_jointly)
{
}
MaskingPlugin::~MaskingPlugin() {}
bool MaskingPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    // This plugin only supports ordinary floats, and NCHW input format
    if (type == DataType::kFLOAT && format == PluginFormat::kNCHW)
    {
        return true;
    }
    else
    {
        return false;
    }
}
void MaskingPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    ASSERT(
        inputTypes[0] == DataType::kFLOAT && inputTypes[1] == DataType::kFLOAT && floatFormat == PluginFormat::kNCHW);

    ASSERT(nbInputs == 2);
    ASSERT(nbOutputs == 1);

    mRpnHeight = inputDims->d[1];
    mRpnWidth = inputDims->d[2];
}
IPluginV2Ext* MaskingPlugin::clone() const
{
    IPluginV2Ext* plugin = new layerPlugin(mLayerName, mInputHeight, mInputWidth, mRpnHeight, mRpnWidth,
        mRpnStdScaling, mRpnStride, mBboxMinSize, mNmsIouThreshold, mPreNmsTopN, mMaxBoxNum, &mAnchorSizes[0],
        mAnchorSizeNum, &mAnchorRatios[0], mAnchorRatioNum);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
MaskingPluginCreator::MaskingPluginCreator()
{
    /*mPluginAttributes.emplace_back(PluginField("input_height", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("input_width", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("rpn_stride", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("roi_min_size", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("nms_iou_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("pre_nms_top_n", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("post_nms_top_n", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("anchor_sizes", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("anchor_ratios", nullptr, PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();*/
}

MaskingPluginCreator::~MaskingPluginCreator() {}

IPluginV2Ext* MaskingPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    int input_height = 0, input_width = 0, rpn_stride = 0, pre_nms_top_n = 0, post_nms_top_n = 0;
    float roi_min_size = 0.0f, nms_iou_threshold = 0.0f;
    std::vector<float> anchor_sizes;
    std::vector<float> anchor_ratios;

    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;

        if (!strcmp(attr_name, "input_height"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            input_height = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attr_name, "input_width"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            input_width = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attr_name, "rpn_stride"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            rpn_stride = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attr_name, "roi_min_size"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            roi_min_size = *(static_cast<const float*>(fields[i].data));
        }
        else if (!strcmp(attr_name, "nms_iou_threshold"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            nms_iou_threshold = *(static_cast<const float*>(fields[i].data));
        }
        else if (!strcmp(attr_name, "pre_nms_top_n"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            pre_nms_top_n = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attr_name, "post_nms_top_n"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            post_nms_top_n = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attr_name, "anchor_sizes"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            const float* as = static_cast<const float*>(fields[i].data);

            for (int j = 0; j < fields[i].length; ++j)
            {
                ASSERT(*as > 0.0f);
                anchor_sizes.push_back(*as);
                ++as;
            }
        }
        else if (!strcmp(attr_name, "anchor_ratios"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            const float* ar = static_cast<const float*>(fields[i].data);

            // take the square root.
            for (int j = 0; j < fields[i].length; ++j)
            {
                ASSERT(*ar > 0.0f);
                anchor_ratios.push_back(std::sqrt(*ar));
                ++ar;
            }
        }
    }

    ASSERT(input_height > 0 && input_width > 0 && rpn_stride > 0 && pre_nms_top_n > 0 && post_nms_top_n
        && roi_min_size >= 0.0f && nms_iou_threshold > 0.0f);

    IPluginV2Ext* plugin = new layerPlugin(name, input_height, input_width, RPN_STD_SCALING, rpn_stride,
        roi_min_size, nms_iou_threshold, pre_nms_top_n, post_nms_top_n, &anchor_sizes[0], anchor_sizes.size(),
        &anchor_ratios[0], anchor_ratios.size());
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
// Return the DataType of the plugin output at the requested index.
DataType layerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // one outputs
    ASSERT(index == 0);
    return DataType::kFLOAT;
}








//Plugin configuration APIs: getPluginType, getPluginVersion
const char* MaskingPlugin::getPluginType() const
{
    return MASKING_PLUGIN_NAME;
}

const char* MaskingPlugin::getPluginVersion() const
{
    return MASKING_PLUGIN_VERSION;
}

int MaskingPlugin::getNbOutputs() const
{
    return 1;
}

const char* MaskingPluginCreator::getPluginName() const
{
    return PROPOSAL_PLUGIN_NAME;
}

const char* MaskingPluginCreator::getPluginVersion() const
{
    return PROPOSAL_PLUGIN_VERSION;
}

const PluginFieldCollection* MaskingPluginCreator::getFieldNames()
{
    return &mFC;
}






//Execution configurations APIs: enqueue (call to CUDA kernel implementation) 
int MaskingPlugin::enqueue(
    int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
a
{
    int status = -1;
    // Our plugin outputs only one tensor
    void* output = outputs[0];
    status = proposalInference_gpu(stream, inputs[0], inputs[1], batchSize, mInputHeight, mInputWidth, mRpnHeight,
        mRpnWidth, mMaxBoxNum, mPreNmsTopN, &mAnchorSizes[0], mAnchorSizeNum, &mAnchorRatios[0], mAnchorRatioNum,
        mRpnStdScaling, mRpnStride, mBboxMinSize, mNmsIouThreshold, workspace, output);
    return status;
}






//Laavarisss :(
int MaskingPlugin::initialize()
{
    return 0;
}

size_t MaskingPlugin::getWorkspaceSize(int max_batch_size) const
{
    return _get_workspace_size(max_batch_size, mAnchorSizeNum, mAnchorRatioNum, mRpnHeight, mRpnWidth, mMaxBoxNum);
}

void MaskingPlugin::terminate() {}

void MaskingPlugin::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void MaskingPlugin::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* MaskingPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

// Return true if output tensor is broadcast across a batch.
bool MaskingPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool MaskingPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void MaskingPlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void MaskingPlugin::detachFromContext() {
	}




