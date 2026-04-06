# `6_IPTW_WEIGHTED_LOGISTIC_bidirectional.R` 修改说明

**文件版本**：修改后版本  
**修改日期**：2026-03-17  
**修改人**：Claude（Anthropic），经李子航确认  
**涉及行号**：第151–157行、第161–162行、第195–201行（修改后行号）

---

## 背景：为什么需要这次修改？

原始版本在处理 `Cohort`（回顾性 vs 前瞻性）这个变量时，将其纳入了**倾向性评分（PS）模型**的协变量列表。这在方法论上存在一个根本性问题：

**`Cohort` 不是临床混杂变量，而是研究设计变量。**

IPTW 框架的核心逻辑是：倾向性评分模型应只包含那些**同时影响暴露分配（是否破裂型）和结局（是否重吸收）的临床因素**，也就是真正意义上的混杂变量。例如 `Initial_volume`、`Komori`、`RSI` 等，都有明确的生物学机制同时影响突出形态和重吸收潜力，是合格的混杂变量。

但 `Cohort` 不同。它表示一个患者来自哪个研究队列，这是由**抽样设计**决定的，不是由椎间盘突出的生物学特征决定的。把它纳入 PS 模型，会产生以下两个问题：

1. **PS 模型的临床可解释性被破坏**。倾向性评分本应反映"根据患者的临床基线特征，发生破裂型突出的概率"，混入 `Cohort` 后，它实际上同时反映了"这个患者来自哪个队列"，这让 PS 的含义模糊。

2. **两队列间有意义的系统性差异被错误地"平衡掉"**。回顾性和前瞻性队列的差异（纳入标准、随访策略、时代效应等）是需要被**报告和讨论**的，而不是被 IPTW 权重抹平。

**正确的做法**是：把 `Cohort` 从 PS 模型中移除，同时在**双重稳健（Doubly Robust）分析的结局模型**中纳入 `Cohort` 作为调整变量。这样既保持了 PS 模型的临床纯洁性，又在因果效应估计中控制了队列差异的潜在干扰，两全其美。

---

## 具体修改内容

### 修改一：`exclude_cols` 中加入 `Cohort`

**位置**：`run_case` 函数内，第151–157行（修改后）

**修改前：**
```r
exclude_cols <- c("ID","Name","Unified_ID","Source_File","Source_Sheet",
                  "Absorption_type","Last_volume","Absorption_rate",
                  "Months_of_Review","Komori","Reabsorption","Rupture")
```

**修改后：**
```r
# Cohort is a study-design variable, not a clinical confounder.
# Excluding it from ps_covars keeps the PS model clinically interpretable;
# Cohort will instead be forced into the doubly-robust outcome model below.
exclude_cols <- c("ID","Name","Unified_ID","Source_File","Source_Sheet",
                  "Absorption_type","Last_volume","Absorption_rate",
                  "Months_of_Review","Komori","Reabsorption","Rupture",
                  "Cohort")
```

**修改理由**：

`exclude_cols` 是定义"哪些变量不进入 PS 协变量列表"的排除清单。原来只排除了 ID 类字段和结局/暴露变量，`Cohort` 没有被排除，因此会自动进入 `ps_covars` 并被纳入 PS 模型。加入 `Cohort` 到排除清单后，PS 模型就只包含真正的临床协变量（年龄、性别、影像学指标等），符合 IPTW 的方法论要求。

---

### 修改二：`factor_vars` 中移除 `Cohort`

**位置**：`run_case` 函数内，第161–162行（修改后）

**修改前：**
```r
factor_vars <- c("Gender","Herniated_Level","Iwabuchi","Modic",
                 "Spinal_canal_stenosis","Bull_eye","Cohort")
```

**修改后：**
```r
# "Cohort" removed from factor_vars: it is excluded from ps_covars
# (see exclude_cols above).
factor_vars <- c("Gender","Herniated_Level","Iwabuchi","Modic",
                 "Spinal_canal_stenosis","Bull_eye")
```

**修改理由**：

`factor_vars` 的作用是指定哪些变量在进入模型前需要被转换为因子型（`as.factor()`）。由于 `Cohort` 已经从 `ps_covars` 中排除，它不会进入 PS 模型，因此也不需要出现在 `factor_vars` 里。保留它在 `factor_vars` 中虽然不会引发报错，但会造成逻辑上的不一致（声明了要转换一个不会被使用的变量），容易让阅读代码的人产生困惑。移除后代码语义更清晰。

---

### 修改三：`dr_vars` 中强制加入 `Cohort`

**位置**：`run_case` 函数内，第195–201行（修改后）

**修改前（`dr_vars` 的生成逻辑，第188行）：**
```r
dr_vars <- dr_vars[dr_vars %in% names(dat) &
                   sapply(dr_vars, function(v) !all(is.na(dat[[v]])))]
# 直接进入双重稳健分析，Cohort 不在其中
```

**修改后：**
```r
dr_vars <- dr_vars[dr_vars %in% names(dat) &
                   sapply(dr_vars, function(v) !all(is.na(dat[[v]])))]
# Force Cohort into the doubly-robust outcome model to account for
# systematic between-cohort differences (e.g. enrolment period, follow-up
# strategy) that were intentionally kept out of the PS model.
if ("Cohort" %in% names(dat) &&
    length(unique(na.omit(as.character(dat[["Cohort"]])))) >= 2) {
  dr_vars <- unique(c("Cohort", dr_vars))
}
```

**修改理由**：

双重稳健（Doubly Robust）分析的核心优势在于：即使 PS 模型或结局模型中有一个存在轻微误设，估计仍然是一致的。把 `Cohort` 从 PS 模型中移除后，它就不再被 IPTW 权重所控制，因此需要在**结局模型**中显式调整，才能去除队列差异对因果效应估计的干扰。

这段代码的逻辑是：

1. 先按原有规则生成 `dr_vars`（SMD > 0.1 的残余不平衡变量）
2. 再检查 `Cohort` 是否存在于当前数据集且有至少两个水平（避免在单队列敏感性分析中报错）
3. 如果满足条件，将 `Cohort` 强制追加到 `dr_vars` 的**最前面**，确保结局模型公式为 `Reabsorption ~ Rupture + Cohort + [其他残余不平衡变量]`

使用 `unique()` 防止 `Cohort` 因重复出现在公式中。在单队列敏感性分析（`Retrospective_15m` 或 `Prospective_15m`）中，`Cohort` 只有一个水平，条件判断会自动跳过，不影响分层分析的运行。

---

## 修改后的分析逻辑流程

```
数据输入（合并队列 662 例）
        ↓
过滤随访时间 / Reabsorption 缺失 / Komori 无效
        ↓
定义 Rupture（Komori 1 = Contained，Komori 2-4 = Non-contained）
        ↓
构建 ps_covars
  ├── 排除：ID类、结局变量、Komori、Cohort（新增）
  └── 仅保留：临床协变量（年龄、性别、影像学指标等）
        ↓
拟合 PS 模型（logistic regression）
  └── 公式：Rupture ~ 年龄 + 性别 + 影像学指标...（不含 Cohort）
        ↓
计算稳定化权重（sw）→ 权重诊断 → 必要时截断
        ↓
构建加权设计对象（svydesign）
        ↓
主分析（IPTW 加权逻辑回归）
  └── 公式：Reabsorption ~ Rupture
        ↓
双重稳健分析
  └── 公式：Reabsorption ~ Rupture + Cohort（强制）+ [SMD>0.1 的变量]
        ↓
输出 Table 4 / eTable 1-3
```

---

## 对结果的预期影响

| 分析模块 | 修改前 | 修改后 | 影响方向 |
|---|---|---|---|
| PS 模型协变量 | 含 Cohort | 不含 Cohort | PS 更纯粹，临床可解释性更强 |
| IPTW 主分析（OR） | 受 Cohort 平衡影响 | 仅由临床协变量驱动 | 数值可能略有变化 |
| 双重稳健 DR 分析 | Cohort 未被显式调整 | Cohort 被强制纳入结局模型 | DR 对队列差异的控制更严格 |
| 平衡诊断（Love plot）| Cohort 出现在图中 | Cohort 不再出现 | 图更简洁，只展示临床变量的平衡情况 |
| 单队列敏感性分析 | 正常运行 | 正常运行（Cohort 条件判断自动跳过）| 无影响 |

**重要提示**：修改后需要重新运行脚本，用新的输出结果（特别是 `Table4_logistic_models_comparison.csv`）更新解读报告第8节的所有数值。

---

## 论文方法部分对应的表述建议

> "倾向性评分模型纳入所有基线临床及影像学协变量，包括年龄、性别、突出节段、Pfirrmann 分级、Iwabuchi 分型、Modic 改变、MSU 分级、椎管狭窄、Bull's eye 征、矢状位参数（SS）、椎体后缘高度、初始突出体积（Initial volume）、RSI 及 DHI，但不包括队列标识（Cohort）。队列标识属于研究设计变量而非临床混杂因素，不纳入 PS 模型，以保持倾向性评分的临床可解释性。在双重稳健分析的结局模型中，队列标识被强制纳入作为调整变量，以控制回顾性与前瞻性队列间可能存在的系统性差异（如纳入时段、随访策略等）。"

---

*本说明文档由 Claude（Anthropic）于 2026-03-17 生成，记录对 `6_IPTW_WEIGHTED_LOGISTIC_bidirectional.R` 的代码修改内容与方法论依据。*

---

## 补充修改（2026-03-17，第二轮）

### 修改四：Multivariable模型同步加入Cohort

**位置**：`run_case` 函数内，第192–203行（修改后）

**问题背景**：

第一轮修改后，Table4中出现了口径不一致的问题：
- Multivariable行：`Reabsorption ~ Rupture + [ps_covars]`，**不含Cohort**
- Doubly robust行：`Reabsorption ~ Rupture + Cohort + [残余变量]`，**含Cohort**

如果方法学立场是"Cohort作为设计变量应在结局模型中调整"，那Multivariable模型也应体现这个立场，否则两种adjusted结果的调整口径不一致，审稿人会质疑。

**修改前：**
```r
res_multi <- extract_glm(glm(
  as.formula(paste("Reabsorption ~ Rupture +", paste(ps_covars, collapse = " + "))),
  data = dat, family = binomial()
))
```

**修改后：**
```r
# Multivariable model: adjust ps_covars (clinical variables) + Cohort.
# Cohort is added here to keep the adjustment strategy consistent with the
# doubly-robust model — both treat Cohort as a design variable to be
# controlled in the outcome model rather than balanced via IPTW weights.
multi_covars <- ps_covars
if ("Cohort" %in% names(dat) &&
    length(unique(na.omit(as.character(dat[["Cohort"]])))) >= 2) {
  multi_covars <- unique(c("Cohort", multi_covars))
}
res_multi <- extract_glm(glm(
  as.formula(paste("Reabsorption ~ Rupture +", paste(multi_covars, collapse = " + "))),
  data = dat, family = binomial()
))
```

### 修改五：Table4的Method描述同步更新

**修改前：**
```
"Standard logistic regression (glm, unweighted, adjusted by ps_covars)"
```

**修改后：**
```
"Standard logistic regression (glm, unweighted, adjusted by ps_covars + Cohort)"
```

---

## 最终Table4逻辑结构（完整版）

| 模型 | 公式 | Cohort处理方式 |
|---|---|---|
| Unadjusted | `Reabsorption ~ Rupture` | 不调整 |
| Multivariable | `Reabsorption ~ Rupture + Cohort + [ps_covars]` | 结局模型显式调整 |
| IPTW | `Reabsorption ~ Rupture`（加权） | 临床变量经权重平衡；Cohort不纳入PS模型 |
| Doubly robust | `Reabsorption ~ Rupture + Cohort + [SMD>0.1变量]`（加权） | 结局模型显式调整 + 权重双重保障 |

四行递进清晰，Cohort的处理方式在所有结局模型中一致。

### 单队列敏感性分析的自动处理

在 `Retrospective_15m` 或 `Prospective_15m` 分层分析中，`Cohort` 只有一个水平，代码中的条件判断（`length(unique(...)) >= 2`）会自动跳过，`multi_covars` 和 `dr_vars` 中都不会加入 `Cohort`，不影响单队列分析的正常运行。
