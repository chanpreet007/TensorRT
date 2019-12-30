

#ifndef Masking_LSTM_TRT
#define Masking_LSTM_TRT

#include "NvInferPlugin.h"
#include "kernel.h"
#include "plugin.h"
#include <string>
#include <vector>

typedef struct
{
    
    
    std::vector<vector<bool> > computed_output;
        
}MaskingParams;

using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace nvinfer1
{
namespace plugin
{

class MaskingPlugin : public IPluginV2Ext
{
public:
    MaskingPlugin(const std::string name);

    MaskingPlugin(const std::string name, MaskingParams params, int samples, int timesteps, std::vector<vector<int>> features, bool supports_masking, int mask_value, bool compute_output_and_mask_jointly);

    MaskingPlugin(const std::string name,  MaskingParams params, bool supports_masking, int mask_value, bool compute_output_and_mask_jointly);

    MaskingPlugin( const void* serial_buf, size_t serial_size);

    // It doesn't make sense to make MaskingPlugin without arguments, so we delete default constructor.
    MaskingPlugin() = delete;

    ~MaskingPlugin() override;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

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
     template <typename T>
    void write(char*& buffer, const T& val) const
    {
        std::memcpy(buffer, &val, sizeof(T));
        buffer += sizeof(T);
    }

    template <typename T>
    T read(const char*& buffer)
    {
        T val;
        std::memcpy(&val, buffer, sizeof(T));
        buffer += sizeof(T);
        return val;
    }
   
    MaskingParams mParams
    bool nSupports_masking;
    int mMask_value;
    bool nCompute_output_and_mask_jointly;
    int mSamples;
    int mTimesteps;
    std::vector<vector<int>> mFeatures;
    const std::string mLayerName;
    const char* mPluginNamespace = "";

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
    MaskingParams params;
    bool nSupports_masking;
    int mMask_value;
    bool nCompute_output_and_mask_jointly;
    static std::vector<PluginField> mPluginAttributes;
    std::string mPluginNamespace = "";
    REGISTER_TENSORRT_PLUGIN(MaskingPluginCreator);
};

} 

} 

#endif // Masking_LSTM_TRT


