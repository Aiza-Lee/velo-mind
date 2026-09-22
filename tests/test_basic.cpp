#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "velomind/executable.h"
#include "velomind/graph.h"
#include "velomind/ops.h"
#include "velomind/tensor.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"
#include "velomind/velomind.h"
#include "velomind.h"

#include "test_helpers.h"

#if defined(__SANITIZE_ADDRESS__) || (defined(__has_feature) && __has_feature(address_sanitizer))
// 压制系统驱动层与 libdbus-1 / Vulkan ICD 内部单例未释放导致的 LSan 误报
extern "C" const char* __lsan_default_suppressions() {
    return "leak:libdbus-1\n"
           "leak:libvulkan\n"
           "leak:libnvidia\n"
           "leak:libGLX\n";
}
#endif

TEST_CASE("Storage allocator auto-registration is live", "[backend]") {
    using namespace velomind;

    auto a = TensorStorage::allocate(16, DeviceType::CPU);
    REQUIRE(a != nullptr);
    CHECK(Device::cpu().is_available());
    CHECK(TensorStorage::is_available(DeviceType::CPU));
}

TEST_CASE("Device availability query and error diagnostics", "[backend][devices]") {
    using namespace velomind;

    CHECK(Device::cpu().is_available());
    CHECK(Device(DeviceType::CPU).is_available());
    CHECK(TensorStorage::is_available(DeviceType::CPU));

    bool ispc_avail = Device::ispc().is_available();
    (void)ispc_avail;
    bool cuda_avail = Device::cuda().is_available();
    (void)cuda_avail;

    CHECK_FALSE(Device(static_cast<DeviceType>(99)).is_available());

    Graph g;
    auto in = g.input({4}, DataType::Float32);
    auto out = g.op(Op::Relu, in);
    (void)out;
    REQUIRE_THROWS_WITH(
        g.build(static_cast<DeviceType>(99)),
        Catch::Matchers::ContainsSubstring("invalid device"));
}

TEST_CASE("Executable bind_input wraps an external buffer", "[executable]") {
    using namespace velomind;

    Graph g;
    auto x = g.input({4}, DataType::Float32);
    auto y = g.op(Op::Relu, x);

    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    auto data = std::make_shared<std::vector<float>>(
        std::initializer_list<float>{-1.0f, 0.5f, -2.0f, 3.0f});
    auto owner = std::shared_ptr<void>(data, data->data());
    auto* exec_mut = const_cast<Executable*>(exec.get());
    REQUIRE_NOTHROW(exec_mut->bind_input(x, ExternalBuffer{
        data->data(), data->size() * sizeof(float), DeviceType::CPU, owner}));
    exec->execute();

    std::vector<float> out(y.numel());
    y.copy_to_host(velomind_test::as_writeable_bytes(out));
    REQUIRE(out[0] == 0.0f);
    REQUIRE(out[3] == 3.0f);
}

TEST_CASE("Per-op default attributes lookup", "[ops][attrs]") {
    using namespace velomind;

    SECTION("RMSNorm defaults to RMSNormAttrs") {
        auto attrs = default_op_attrs(Op::RMSNorm);
        REQUIRE(std::holds_alternative<RMSNormAttrs>(attrs));
        const auto& r = std::get<RMSNormAttrs>(attrs);
        REQUIRE(r.epsilon == Catch::Approx(1e-5f));
        REQUIRE(r.axis == -1);
    }

    SECTION("Conv2D defaults to Conv2DAttrs") {
        auto attrs = default_op_attrs(Op::Conv2D);
        REQUIRE(std::holds_alternative<Conv2DAttrs>(attrs));
        const auto& c = std::get<Conv2DAttrs>(attrs);
        REQUIRE(c.padding == std::array<int, 2>{0, 0});
        REQUIRE(c.stride  == std::array<int, 2>{1, 1});
        REQUIRE(c.dilation == std::array<int, 2>{1, 1});
        REQUIRE(c.groups  == 1);
    }

    SECTION("Softmax defaults to SoftmaxAttrs with axis -1") {
        auto attrs = default_op_attrs(Op::Softmax);
        REQUIRE(std::holds_alternative<SoftmaxAttrs>(attrs));
        REQUIRE(std::get<SoftmaxAttrs>(attrs).axis == -1);
    }

    SECTION("MatMul defaults to MatMulAttrs with trans=false") {
        auto attrs = default_op_attrs(Op::MatMul);
        REQUIRE(std::holds_alternative<MatMulAttrs>(attrs));
        const auto& m = std::get<MatMulAttrs>(attrs);
        REQUIRE_FALSE(m.trans_a);
        REQUIRE_FALSE(m.trans_b);
    }

    SECTION("Concat defaults to ConcatAttrs with axis 0") {
        auto attrs = default_op_attrs(Op::Concat);
        REQUIRE(std::holds_alternative<ConcatAttrs>(attrs));
        REQUIRE(std::get<ConcatAttrs>(attrs).axis == 0);
    }

    SECTION("Transpose defaults to TransposeAttrs") {
        auto attrs = default_op_attrs(Op::Transpose);
        REQUIRE(std::holds_alternative<TransposeAttrs>(attrs));
    }

    SECTION("Elementwise ops default to NoAttrs") {
        REQUIRE(std::holds_alternative<NoAttrs>(default_op_attrs(Op::Add)));
        REQUIRE(std::holds_alternative<NoAttrs>(default_op_attrs(Op::Sub)));
        REQUIRE(std::holds_alternative<NoAttrs>(default_op_attrs(Op::Mul)));
        REQUIRE(std::holds_alternative<NoAttrs>(default_op_attrs(Op::Relu)));
        REQUIRE(std::holds_alternative<NoAttrs>(default_op_attrs(Op::Embedding)));
    }
}

TEST_CASE("Graph::op defaults to op-specific attrs without explicit OpDescriptor", "[graph][attrs]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x      = g.input({1, 4}, DataType::Float32);
    auto weight = g.input({4},    DataType::Float32);
    auto y      = g.op(Op::RMSNorm, x, weight);

    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    std::vector<float> x_data = {2.0f, 2.0f, 2.0f, 2.0f};
    std::vector<float> w_data = {1.0f, 1.0f, 1.0f, 1.0f};
    x.copy_from_host(as_bytes(x_data));
    weight.copy_from_host(as_bytes(w_data));

    REQUIRE_NOTHROW(exec->execute());

    std::vector<float> y_data(4);
    y.copy_to_host(as_writeable_bytes(y_data));
    for (float val : y_data) {
        REQUIRE(val == Catch::Approx(1.0f).margin(1e-4f));
    }
}

TEST_CASE("to_string and operator<< formatting helpers for core types", "[print][types]") {
    using namespace velomind;
    using velomind_test::as_bytes;

    // 1. shape_t 与 stride_t 格式化
    shape_t s1 = {2, 3, 4};
    CHECK(to_string(s1) == "[2, 3, 4]");
    shape_t empty_s = {};
    CHECK(to_string(empty_s) == "[]");

    std::ostringstream ss_s;
    ss_s << s1;
    CHECK(ss_s.str() == "[2, 3, 4]");

    // 2. DataType 与 DeviceType 格式化
    CHECK(to_string(DataType::Float32) == "Float32");
    CHECK(to_string(DataType::Int8) == "Int8");
    CHECK(to_string(DeviceType::CPU) == "CPU");

    std::ostringstream ss_dt;
    ss_dt << DataType::Float16 << " on " << DeviceType::CUDA;
    CHECK(ss_dt.str() == "Float16 on CUDA");

    // 3. Op 与 OpDescriptor 格式化
    CHECK(to_string(Op::MatMul) == "MatMul");
    CHECK(to_string(OpDescriptor{Op::RMSNorm}) == "OpDescriptor(op=RMSNorm)");

    std::ostringstream ss_op;
    ss_op << Op::Softmax;
    CHECK(ss_op.str() == "Softmax");

    // 4. TensorStorage 格式化
    auto st = TensorStorage::allocate(16 * sizeof(float), DeviceType::CPU);
    REQUIRE(st != nullptr);
    st->shape = {4, 4};
    st->dtype = DataType::Float32;
    std::string st_str = to_string(*st);
    CHECK_THAT(st_str, Catch::Matchers::ContainsSubstring("shape=[4, 4]"));
    CHECK_THAT(st_str, Catch::Matchers::ContainsSubstring("dtype=Float32"));
    CHECK_THAT(st_str, Catch::Matchers::ContainsSubstring("allocated"));

    // 5. Tensor 格式化（未分配 vs 已分配 + CPU数据预览）
    Graph g;
    auto a = g.input({4}, DataType::Float32);
    // 未编译分配前
    CHECK_THAT(to_string(a), Catch::Matchers::ContainsSubstring("shape=[4]"));
    CHECK_THAT(to_string(a), Catch::Matchers::ContainsSubstring("unallocated"));

    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    // 编译分配后且灌入数据
    std::vector<float> a_vals = {1.5f, 2.5f, 3.5f, 4.5f};
    a.copy_from_host(as_bytes(a_vals));

    std::string a_str = to_string(a);
    CHECK_THAT(a_str, Catch::Matchers::ContainsSubstring("shape=[4]"));
    CHECK_THAT(a_str, Catch::Matchers::ContainsSubstring("dtype=Float32"));
    CHECK_THAT(a_str, Catch::Matchers::ContainsSubstring("data=[1.5, 2.5, 3.5, 4.5]"));

    std::ostringstream ss_tensor;
    ss_tensor << a;
    CHECK(ss_tensor.str() == a_str);
}

TEST_CASE("Umbrella header velomind.h exports all core symbols", "[api][umbrella]") {
    using namespace velomind;

    Graph g;
    auto x = g.input({2, 3}, DataType::Float32);
    auto y = g.op(Op::Relu, x);
    auto exec = g.build(Device::cpu());
    REQUIRE(exec != nullptr);

    CHECK(Device::cpu().is_available());
    CHECK(Device::cpu().name() == "CPU");
    CHECK(data_type_size(DataType::Float32) == 4);
}

TEST_CASE("Device class features, string parsing, memory info, and support queries", "[device]") {
    using namespace velomind;

    // 工厂方法与类型属性校验
    auto cpu = Device::cpu();
    CHECK(cpu.type() == DeviceType::CPU);
    CHECK(cpu.index() == 0);
    CHECK(cpu.is_available());
    CHECK(cpu.name() == "CPU");

    auto cuda0 = Device::cuda(0);
    CHECK(cuda0.type() == DeviceType::CUDA);
    CHECK(cuda0.index() == 0);

    auto cuda1 = Device::cuda(1);
    CHECK(cuda1.type() == DeviceType::CUDA);
    CHECK(cuda1.index() == 1);
    CHECK(cuda1.name() == "CUDA:1");

    auto vulkan = Device::vulkan();
    CHECK(vulkan.type() == DeviceType::VULKAN);
    CHECK(vulkan.index() == 0);

    // 隐式转换为 DeviceType
    DeviceType dt = cpu;
    CHECK(dt == DeviceType::CPU);

    // 字符串解析
    CHECK(Device::from_string("cpu") == Device::cpu());
    CHECK(Device::from_string("CPU") == Device::cpu());
    CHECK(Device::from_string("ispc") == Device::ispc());
    CHECK(Device::from_string("cuda") == Device::cuda(0));
    CHECK(Device::from_string("cuda:0") == Device::cuda(0));
    CHECK(Device::from_string("cuda:2") == Device::cuda(2));
    CHECK(Device::from_string("vulkan") == Device::vulkan(0));
    CHECK(Device::from_string("vulkan:1") == Device::vulkan(1));

    // 非法字符串解析
    CHECK(Device::from_string("") == std::nullopt);
    CHECK(Device::from_string("unknown") == std::nullopt);
    CHECK(Device::from_string("cuda:-1") == std::nullopt);
    CHECK(Device::from_string("cuda:abc") == std::nullopt);

    // 设备枚举与默认设备探测
    auto avail = Device::available_devices();
    CHECK(!avail.empty());
    CHECK(std::find(avail.begin(), avail.end(), Device::cpu()) != avail.end());

    auto default_dev = Device::default_device();
    CHECK(default_dev.is_available());

    // 算子与数据类型支持度查询
    CHECK(cpu.supports(DataType::Float32));
    CHECK(cpu.supports(Op::Add, DataType::Float32));
    CHECK(cpu.supports(Op::MatMul, DataType::Float32));

    // 内存信息探测
    auto mem = cpu.memory_info();
    if (mem.has_value()) {
        CHECK(mem->total_bytes > 0);
        CHECK(mem->free_bytes > 0);
        CHECK(mem->total_bytes >= mem->used_bytes);
    }

    // 描述信息
    CHECK(!cpu.description().empty());

    // 字符串化与输出流
    CHECK(to_string(cpu) == "CPU");
    std::ostringstream oss;
    oss << cpu;
    CHECK(oss.str() == "CPU");

    // 哈希支持
    std::hash<Device> hasher;
    CHECK(hasher(cpu) == hasher(Device::cpu()));
    CHECK(hasher(cpu) != hasher(Device::cuda(0)));
}
