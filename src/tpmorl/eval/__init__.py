"""tpmorl.eval —— 评价指标层。

与 `tpmorl.rl` 的分工：`rl` 负责训练与环境（会改变可达上界，改动即作废分母），
`eval` 只**读**既有落盘结果并重算指标，不 import 任何会改写情景全局的东西，
也绝不触碰 `rl/scale.py` 的 REF_MODES / REF_SEEDS / REF_VER。

对外接口见 `tpmorl.eval.metrics`。
"""
from .metrics import (  # noqa: F401
    ScenarioSpec,
    approval_pmf,
    expand_outcomes,
    implementation_metrics,
    three_layer_table,
    LAYER_ROWS,
)
