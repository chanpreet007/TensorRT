
#ifndef Masking_LSTM
#define Masking_LSTM

#include "NvInferPlugin.h"
#include "kernel.h"
#include "plugin.h"
#include <string>
#include <vector>

using namespace nvinfer1::plugin;

namespace nvinfer1
{
namespace plugin
{

class MaskingPlugin : public IPluginV2Ext
{
public:
    MaskingPlugin(const std::string name);

    MaskingPlugin(const std::string name, int samples, int timesteps, int features);

    MaskingPlugin(const std::string name, bool supports_masking, int mask_value, bool compute_output_and_mask_jointly);

    //layerPlugin(const std::string name, const void* serial_buf, size_t serial_size);

    // It doesn't make sense to make ProposalPlugin without arguments, so we delete default constructor.
    MaskingPlugin() = delete;

    ~MaskingPlugin() override;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int) const override;

    int enqueue(
        int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    bool supportsFormat(DataType type, PluginFormat format) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    nvinfer1::IPluginV2Ext* clone() const override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;

    void detachFromContext() override;

private:
   
    
    bool msupports_masking;
    int mmask_value;
    bool mcompute_output_and_mask_jointly ;
    int msamples;
    int mtimesteps;
    int mfeatures;
    const std::string mLayerName;
    std::string mNamespace;

};

class MaskingPluginCreator : public BaseCreator
{
public:
    MaskingPluginCreator();

    ~MaskingPluginCreator() override;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin

} // namespace nvinfer1

#endif // PROPOSAL_PLUGIN_H
REGISTER_TENSORRT_PLUGIN(MASKING_layer_TRT);


