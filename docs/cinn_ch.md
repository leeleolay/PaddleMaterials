# CINN 执行后端

[English](cinn.md)

PaddleMaterials 默认使用 eager，并将 CINN 作为由模型持有的可选运行时。训练、
预测和采样共用同一个配置，模型参数以及公开输入输出协议不会因后端而改变。

## 配置

在任务配置中增加顶层 `Execution`：

```yaml
Execution:
  backend: cinn
  __init_params__:
    full_graph: false
```

`Execution.__init_params__` 可以省略。CINN 默认使用 `full_graph=False`；只有下方
支持表明确标为严格 AST 的模型才应设为 `true`。同样的配置可以通过命令行覆盖：

```bash
python property_prediction/predict.py \
  --model_name megnet_mp2018_train_60k_e_form \
  --weights_name best.pdparams \
  --device gpu:0 \
  --input_path property_prediction/example_data/cifs \
  --input_format cif \
  Execution.backend=cinn
```

不配置 `Execution` 或设置 `Execution.backend=eager` 时，行为与原有 eager 流程
一致。CINN 当前要求 CUDA 设备以及启用 CINN 的 Paddle。AMP 和分布式执行尚未
完成验证，工作流会在开始前拒绝这些组合。

## 运行时协议

公共运行时只承担三类职责：

- `RuntimeMixin` 保存后端、规范化后的后端选项以及惰性的训练/评估缓存；
- `runtime_boundary` 标记一个完整的数值计算操作；
- `CinnBackend` 校验环境并调用 `paddle.jit.to_static`。

数值边界属于模型。Trainer、Predictor 和 Sampler 只负责选择、校验后端，因此
checkpoint 不会增加包装层，eager 与 CINN 的参数键完全相同。

边界缓存名使用一套固定语义：

- 标准数值前向统一为 `forward`；
- 独立于采样流程编译的解码器统一为 `decoder`；
- 单次扩散去噪统一为 `denoise_step`；
- 带 classifier-free guidance 的 prior 采样统一为 `guided_denoise_step`；
- 独立组件使用稳定的角色名，当前为 `graph_encoder` 和 `spectrum_encoder`。

`full_graph=False` 使用 SOT 捕获，可以在不支持的 Python 区域断图；
`full_graph=True` 使用严格 AST，整个边界必须转换为一个静态程序。严格静态程序
仍可能包含多个 CINN kernel，`full_graph` 描述的是捕获范围，不代表单 kernel
融合。

编译在第一次真实模型调用时惰性发生，并按后端、训练/评估模式和边界名缓存。
切换后端或修改运行时选项会清空缓存。

## 支持模型

下表描述模型持有的数值边界，不表示文件解析、图构造、scheduler 或后处理都会
进入 CINN。“已验证”指当前 Paddle 3.3 注册 checkpoint 流程。测试环境、逐权重
数据和时间点结论只在[完整性能矩阵](cinn_performance.md)中维护。

| 模型 | 边界 | 捕获方式 | 已验证流程 |
| --- | --- | --- | --- |
| `MEGNetPlus` | 完整 `_forward` | SOT | 预测和 Trainer 集成 |
| `iComformer` | 完整 `_forward` | SOT | 注册 checkpoint 预测 |
| `DimeNetPlusPlus` | 完整 `_forward` | SOT | 预测和 Trainer 集成 |
| `SphereNet` | 数值 `_runtime_forward` | 属性模型 SOT；势模型严格 AST | 属性预测和能量/力推理 |
| `SFIN` | 完整 `_forward` | 严格 AST | 图像推理 |
| `M3GNet` | 可微 `_runtime_forward` | 严格 AST | 能量/力推理 |
| `CHGNet` | 可微 `_runtime_forward` | 严格 AST | 能量/力/应力/磁矩推理 |
| `DiffCSP` | `_runtime_decode` 去噪步骤 | SOT | 结构采样 |
| `MatterGen`、`MatterGenWithCondition` | `_runtime_denoise` 步骤 | SOT | 无条件和条件采样 |
| `MolecularGraphFormer` | `graph_encoder`、`decoder` 和完整 Tensor `denoise_step` | SOT | 已实现，无独立注册流程 |
| `NMRNetCLIP` | 图编码器和谱编码器 | SOT | 已实现，无独立注册流程 |
| `DiffPrior` | 训练和引导采样的 prior 去噪步骤 | SOT | 已实现，注册 DiffNMR 流程未使用 |
| `DiffNMR` | `decoder`、完整 Tensor `denoise_step` 和可选 prior 去噪 | SOT | 谱条件只做一次 eager 编码；schedule、特征、decoder、softmax 和 posterior 共用一个采样边界 |
| `InfGCN` | 无 | 仅 eager | CINN lowering 未在验证时限内完成 |

InfGCN 仍是正常支持的 eager 模型。它不暴露 runtime 边界或 CINN 构造参数；只有
未来 Paddle 版本能在隔离环境内完成 lowering 后，才应重新接入 CINN。

## 边界放置原则

文件解析、PGL/Python 对象解包、动态拓扑、scheduler 控制流、采样循环、指标和
可视化应保留在 eager；纯 Tensor scheduler 数学可以进入数值边界。编译数值核心
只接收 Tensor 和整数拓扑。当前模型的处理方式包括：

- DiffCSP 在 `_runtime_decode` 前构造可变长度的全连接边；
- MatterGen 在每次编译去噪前重建动态周期邻接图；
- SphereNet 在 eager 中选择离散扭转边，再在 `_runtime_forward` 中计算连续几何；
- CHGNet 在 eager 中批处理离散图索引，把几何、消息传递、能量、力和应力保留在
  同一个严格边界中。
- DiffNMR 只在 eager 中编码一次不变的谱条件，把 schedule 数学、额外特征、
  decoder、softmax 和 posterior 合并为一个反向步骤边界，再在 eager 中完成离散
  随机采样。

这套 `eager 拓扑 -> Tensor 索引 -> 编译数值计算` 是模型协议的一部分，不是额外
的 fallback adapter。

导数模型需要在进入严格捕获前创建可微的坐标或应变叶子张量。力/应力训练还依赖
二阶梯度。当前注册 checkpoint 验证主要覆盖推理和采样；生产使用力/应力训练前，
应针对目标模型和输入形状运行隔离的 eager/CINN 梯度对齐测试。

## 兼容性保证

- `forward`、`predict` 和 `sample` 的输入输出保持不变；
- eager 始终是默认值，旧模型即使没有 runtime hook，也可以接受显式 eager 覆盖；
- 编译后的 callable 不注册为模型子层，因此 eager/CINN checkpoint 参数键一致；
- 训练和评估运行时分别缓存；
- 公共运行时只校验后端环境与工作流能力，不维护模型专属白名单或拒绝分支。

## 接入新模型

模型显式接收 `execution_backend` 和 `runtime_options`，初始化 mixin，并标记一个
完整数值操作。继续遵循仓库的 `forward -> _forward -> predict` 协议：

```python
from ppmat.models.common.runtime import RuntimeMixin, runtime_boundary


class NewModel(RuntimeMixin, paddle.nn.Layer):
    def __init__(
        self,
        execution_backend="eager",
        runtime_options=None,
    ):
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)
        self.encoder = paddle.nn.Linear(16, 32)

    def forward(self, data, return_loss=True, return_prediction=True):
        prediction = self._forward(data)
        return build_output(prediction, data, return_loss, return_prediction)

    @runtime_boundary("forward")
    def _forward(self, data):
        return self.encoder(data["x"])
```

SOT 边界可以接收 Python/PGL 输入并在需要时断图。严格模型应在边界外准备对象，
同时只保留一份数值实现：

```python
def _forward(self, data):
    graph = pack_graph(data["graph"])
    return self._runtime_forward(
        graph["node"], graph["edge"], graph["edge_index"]
    )

@runtime_boundary("forward")
def _runtime_forward(self, node, edge, edge_index):
    return self.network(node, edge, edge_index)
```

不要增加模型专属 CINN wrapper、重复的 `_tensor_forward`，也不要让永远不会编译
的类暴露 runtime 参数。新增边界需要 eager 数值回归、缓存测试、公开工作流测试，
以及隔离的真实 GPU eager/CINN 对齐，才能在支持表中标为已验证。

## 验证与隔离

带日期的运行环境、验证方法、模型族汇总、全部逐权重耗时、回本次数、数值容差和
InfGCN lowering 结果集中维护在 [CINN 完整性能矩阵](cinn_performance.md)。

CINN lowering 可能直接终止当前进程。新模型用于长期服务前，应在一次性子进程中
完成首次编译和代表性推理：

```bash
timeout --signal=KILL 600 \
  python property_prediction/predict.py \
  --model_name megnet_mp2018_train_60k_e_form \
  --weights_name best.pdparams \
  --device gpu:0 \
  --input_path property_prediction/example_data/cifs \
  --input_format cif \
  Execution.backend=cinn
echo "exit=$?"
```

GNU `timeout` 达到时限时返回 124。进程被信号 *n* 终止时，shell 返回 128+*n*；
例如 `SIGABRT` 对应 134。
