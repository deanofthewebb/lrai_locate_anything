// packed_varlen_attn_plugin.cu — TensorRT 10 plugin for FlashAttention-2 varlen.
//
// Purpose: replace the SDPA-with-mask fallback in MoonViT's `sdpa_packed` with a
// fused FlashAttention-2 varlen call. The SDPA fallback materialises an O(L²)
// attention mask which dominates wall-clock at L > 2000 (i.e. anything above ~2.5 K
// resolution). The plugin avoids that entirely with the same fused kernel
// flash_attn ships.
//
// Build (assuming TRT 10.x headers and a compiled flash_attn library are available):
//
//     nvcc -shared -Xcompiler -fPIC \
//          -I${TRT_INC} -I${CUDA_HOME}/include \
//          -L${FLASH_ATTN_LIB} -lflash_attn \
//          packed_varlen_attn_plugin.cu \
//          -o packed_varlen_attn_plugin.so
//
// Load + register at runtime BEFORE deserializing the engine:
//
//     import ctypes; ctypes.CDLL('./packed_varlen_attn_plugin.so')
//     # then trt.Runtime(logger).deserialize_cuda_engine(...) picks it up
//
// To emit the plugin node in the exported ONNX, register a custom torch op that
// the ONNX exporter maps to PackedVarlenAttn — see lrai_locate_anything/trt/plugins/BUILD.md
// for the bridging step.

#include "NvInferRuntimePlugin.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

// ---- external symbol from flash-attn ----
extern "C" void flash_attn_varlen_fwd(
    const void*    q,
    const void*    k,
    const void*    v,
    const int32_t* cu_q,
    const int32_t* cu_k,
    int            max_seqlen_q,
    int            max_seqlen_k,
    int            total_q,
    int            total_k,
    int            num_heads,
    int            head_dim,
    void*          out,
    cudaStream_t   stream);

using namespace nvinfer1;

static const char* PLUGIN_NAME    = "PackedVarlenAttn";
static const char* PLUGIN_VERSION = "1";


// ---------------------------------------------------------------------------
// IPluginV3 impl
// ---------------------------------------------------------------------------
class PackedVarlenAttn : public IPluginV3,
                        public IPluginV3OneCore,
                        public IPluginV3OneBuild,
                        public IPluginV3OneRuntime {
public:
    // --- IPluginV3OneCore ---
    const char* getPluginName()      const noexcept override { return PLUGIN_NAME; }
    const char* getPluginVersion()   const noexcept override { return PLUGIN_VERSION; }
    const char* getPluginNamespace() const noexcept override { return ""; }

    // --- IPluginV3OneBuild ---
    int  getNbOutputs() const noexcept override { return 1; }
    int  getOutputDataTypes(DataType* out, int nb_out,
                             const DataType* in, int nb_in) const noexcept override {
        out[0] = in[0]; return 0;
    }
    int  getOutputShapes(const DimsExprs* in, int nb_in,
                          const DimsExprs* shape_in, int nb_shape_in,
                          DimsExprs* out, int nb_out,
                          IExprBuilder& eb) noexcept override {
        // (L, H, D) -> (L, H, D)
        out[0] = in[0];
        return 0;
    }
    bool supportsFormatCombination(int pos, const DynamicPluginTensorDesc* in_out,
                                    int nb_in, int nb_out) noexcept override {
        const auto& d = in_out[pos].desc;
        return d.format == TensorFormat::kLINEAR &&
               (d.type == DataType::kHALF || d.type == DataType::kBF16);
    }
    int configurePlugin(const DynamicPluginTensorDesc*, int,
                         const DynamicPluginTensorDesc*, int) noexcept override { return 0; }
    size_t getWorkspaceSize(const DynamicPluginTensorDesc*, int,
                             const DynamicPluginTensorDesc*, int) const noexcept override { return 0; }

    // --- IPluginV3OneRuntime ---
    int onShapeChange(const PluginTensorDesc*, int,
                       const PluginTensorDesc*, int) noexcept override { return 0; }
    int enqueue(const PluginTensorDesc* in, const PluginTensorDesc* /*out*/,
                 const void* const* ins, void* const* outs,
                 void* /*workspace*/, cudaStream_t stream) noexcept override {
        // in[0]=q, in[1]=k, in[2]=v: (L, H, D)  in[3]=cu_q (N+1,)  in[4]=cu_k (N+1,)
        const auto& Q  = in[0];
        const auto& KK = in[1];
        const int total_q = Q.dims.d[0];
        const int total_k = KK.dims.d[0];
        const int H = Q.dims.d[1];
        const int D = Q.dims.d[2];
        flash_attn_varlen_fwd(
            ins[0], ins[1], ins[2],
            reinterpret_cast<const int32_t*>(ins[3]),
            reinterpret_cast<const int32_t*>(ins[4]),
            /*max_seqlen_q*/ total_q, /*max_seqlen_k*/ total_k,
            total_q, total_k, H, D,
            outs[0], stream
        );
        return 0;
    }

    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return this; }
    IPluginCapability* getCapabilityInterface(PluginCapabilityType c) noexcept override {
        switch (c) {
            case PluginCapabilityType::kCORE:    return static_cast<IPluginV3OneCore*>(this);
            case PluginCapabilityType::kBUILD:   return static_cast<IPluginV3OneBuild*>(this);
            case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }
    IPluginV3* clone() noexcept override { return new PackedVarlenAttn(*this); }

    // Serialise/deserialise: no per-instance state, so empty is fine.
    PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        static PluginFieldCollection empty{0, nullptr};
        return &empty;
    }
};


// ---------------------------------------------------------------------------
// Creator
// ---------------------------------------------------------------------------
class PackedVarlenAttnCreator : public IPluginCreatorV3One {
public:
    const char* getPluginName()      const noexcept override { return PLUGIN_NAME; }
    const char* getPluginVersion()   const noexcept override { return PLUGIN_VERSION; }
    const char* getPluginNamespace() const noexcept override { return ""; }
    PluginFieldCollection const* getFieldNames() noexcept override {
        static PluginFieldCollection empty{0, nullptr};
        return &empty;
    }
    IPluginV3* createPlugin(const char* /*name*/,
                             const PluginFieldCollection* /*fc*/,
                             TensorRTPhase /*phase*/) noexcept override {
        return new PackedVarlenAttn();
    }
};

REGISTER_TENSORRT_PLUGIN(PackedVarlenAttnCreator);
