"""Generate the four rubric prompts from one shared preamble.

Why generate instead of hand-writing four files: the point of running V1/V2/V3/V4 against
each other is to compare SCORING SCHEMES. If the framing, the reference/image baselines or
the multi-turn instruction drift between versions, the comparison silently becomes a
comparison of prompts and the result tells you nothing about scale or structure. Emitting
all four from one PREAMBLE string makes that part byte-identical by construction.

    python gen_rubric_prompts.py            # writes prompts/judge_equip_pointwise_v{1..4}.txt
    python gen_rubric_prompts.py --check    # verify the preamble is identical in all four
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

OUT_DIR = Path(__file__).parent / "prompts"

# ---------------------------------------------------------------------------------
# Shared by all four versions. Anything version-specific must NOT go in here.
# ---------------------------------------------------------------------------------
PREAMBLE = """请扮演一名公正严格的裁判，评估 AI 助手对军事装备图像问答任务的回答质量。我会提供：装备图片（一张或多张）、用户问题、一份参考答案、以及被测助手的回答。

## 评分基准

- **参考答案**整理自该装备的权威资料。其中装备型号、类别归属、技术参数（尺寸、排水量、航速、航程、射程等）、研制历史、服役情况等信息大多无法从图片直接看出，以参考答案为准。标注 INA 或"未提供"表示原始资料没有该信息，回答里没有它不算错。
- **图片**作为外观基准：涉及外观、涂装、布局、可见部件、舷号/编号的描述以图片为准。
- **只依据参考答案和图片判断对错。** 不要用你自己对该型号装备的记忆去替换或质疑参考答案——参考答案不完整、或与你所知不同时，仍以参考答案为准。

## 通用判定规则

- 回答不必覆盖参考答案的全部内容，只要所问要点正确即可，详略本身不影响判定。
- 措辞差异、单位换算、合理取整不算错误；回答多讲了正确且相关的内容不扣分。
- **长度中性**：不因回答长而更宽容，长但夹带错误或编造的照样扣分；也不因回答短而扣分，简短准确的不吃亏。
- 保持客观，不受回答风格、语气、自信程度影响。
- 问题可能包含若干轮历史问答（"第N轮问/第N轮答"形式）以及"当前问题"。**历史仅是上下文，只评判被测回答对"当前问题"的作答。**
- **编造**指给出参考答案未提供、图片也无法证实的具体内容（数值、型号、事实），不论它是不是碰巧说对了。这是本场景杀伤力最大的错误类型。
"""

OUTPUT_RULE = """
## 输出要求

只输出一个 JSON，无多余文字、无 markdown。`rubric` 字段固定填 `"{version}"`，用于记录本行是哪一版评分标准打的分。
"""

# ---------------------------------------------------------------------------------
# V1 -- binary gate. Deliberately keeps zero resolution: its role in the comparison is
# to be the high-reliability floor. The only change from the original is disambiguating
# the 型号-vs-类别 bar, which the original left to the judge's discretion and is the
# single biggest source of inconsistent V1 verdicts.
# ---------------------------------------------------------------------------------
V1 = PREAMBLE + """
## 评估流程

1. 先读参考答案，建立事实基准；再看图片，建立外观基准。
2. 逐项找出被测回答中的错误：装备认定错误、事实冲突、编造、答非所问。
3. 按下面的标准给出一个二值判定。

## 判定标准

**装备认定这一项单独说清楚**（原标准在这里含糊，导致同类回答判定不一致）：

- 认对**大类和用途**，但细分型号说错（如 052C 说成 052D）、名称有错字、用了非官方别称 —— **不算认错**，`equipment_correct` 仍为 true。
- 认错**大类或用途**（驱逐舰说成航母、运输机说成轰炸机、自行火炮说成坦克）—— 算认错，`equipment_correct` 为 false。

**`correct = true`** 需同时满足：

1. `equipment_correct` 为 true（装备大类和用途认定正确）。
2. 所问要点回答正确，与参考答案、图片无实质性冲突。
3. 没有编造。

**`correct = false`**：上面三条任一不满足，或回答空泛/答非所问到无法判断它是否理解了图与问题。

## 失败归因

判 false 时，`failure_type` 从下面选一个**最主要**的（判 true 时填 `null`）。这个字段**不参与打分**，只用于事后聚类看模型主要错在哪：

- `"equipment"`：认错装备大类或用途
- `"fact"`：非数值事实与参考答案冲突
- `"param"`：数值参数错误
- `"visual"`：外观描述与图片不符
- `"fabrication"`：编造参考和图片都无法证实的具体内容
- `"irrelevant"`：答非所问、空泛、只复述问题
""" + OUTPUT_RULE.format(version="v1") + """
```
{"rubric": "v1", "correct": true 或 false, "equipment_correct": true 或 false, "failure_type": 上面六个字符串之一 或 null, "reason": "一句话理由，指出关键判定依据"}
```
"""

# ---------------------------------------------------------------------------------
# V2 -- coarse 1-5 Likert on three axes. Two fixes:
#   (a) all five anchors are defined. The original defined only 1/3/5, so 2 and 4 had no
#       meaning and judges collapsed onto three levels -- a "five-point" scale that was
#       really a three-point one.
#   (b) param_score becomes nullable. The original awarded a free 5 when the question
#       asked for no numbers, and param carries weight 0.50 -- so a totally wrong answer
#       scored 0.60 out of 1.00 on those questions. Higher floor than V3 had.
# ---------------------------------------------------------------------------------
V2 = PREAMBLE + """
## 评分方式

从三个维度分别打分，**每维度 1-5 的整数**。维度之间相互独立评分，不要互相影响。

**如果某个维度在本题不适用，输出 `null`，不要给满分。** "不适用"的判断标准：问题没有要求这个维度的内容，且回答也确实没有涉及。只要回答主动涉及了（哪怕问题没问），就要评分。

### param_score（数值参数准确性）
回答中出现的数值型技术参数。字段随装备种类而定：陆战装备的口径/装甲厚度/公路与越野速度/最大行程/初速/射程/载员数；舰船的排水量/航速/续航力/舰长/舰员数；飞机的翼展/最大速度/升限/航程/载弹量/乘员数；导弹的射程/战斗部重量/飞行速度等。凡是参考答案或问题中出现的具体数值，都按此维度评。

- `5`：提到的每个数值都与参考答案一致（允许单位换算、合理取整、四舍五入误差）。
- `4`：极轻微偏差，不影响对该装备的理解。
- `3`：有一处数值出入较明显（误差超过约 10%，非四舍五入可解释），或漏报了问题明确要求的关键数值。
- `2`：某个关键数值明显错误（数量级错误、张冠李戴到另一装备的参数），或多处轻度错误。
- `1`：编造参考答案未提供且图片无法证实的具体数值，或数值信息系统性错误。
- `null`：**问题不要求数值，且回答也确实没有给出任何具体数值。**

### fact_score（非数值事实准确性）
装备名称/型号、类别归属、所属国家、研制历史、装备关系（"XX 型的改进型"）、服役情况等非数值事实。

- `5`：装备大类与用途认定正确，所问事实要点与参考答案、图片无实质冲突。
- `4`：名称错字或用了非官方别称，所指装备清晰可辨；或次要事实（服役年份、研制单位）有出入。
- `3`：认错细分型号，但装备大类和用途没有本质错误。
- `2`：装备类别或用途认定出现明显偏差，可能造成误导。
- `1`：完全认错装备大类；或关键事实与参考答案直接冲突；或答非所问、空泛到无法判断是否理解了图与问题。
- 这个维度**永远适用**，不允许 null。

### style_score（表达质量）
结构是否清晰、是否覆盖了问题要求的要点、有没有言之有物。**与准确性无关**：不因为长就加分、不因为短就减分、不因为内容错误就在这里扣分。

- `5`：条理清晰，覆盖问题要点，详略得当。
- `4`：基本清晰，个别地方啰嗦或遗漏次要要求。
- `3`：能看懂，但结构松散，或遗漏了问题的部分要求。
- `2`：明显重复、空话较多，或结构混乱。
- `1`：几乎没有回应问题的要求，或严重不知所云。
- 这个维度**永远适用**，不允许 null。

打分完成后，用一句话说明各维度的主要扣分点。**三个字段必须全部出现**：适用的填 1-5 的整数，不适用的显式填 `null`，不允许省略字段。
""" + OUTPUT_RULE.format(version="v2") + """
```
{"rubric": "v2", "param_score": 1-5的整数 或 null, "fact_score": 1-5的整数, "style_score": 1-5的整数, "reasoning": "逐维度简要说明打分依据"}
```
"""

# ---------------------------------------------------------------------------------
# V3 -- 0-100 on five axes, pure weighted mean. Fixes:
#   (a) nullable dimensions instead of a free 100 (param + visual were 0.50 of the total
#       weight, so "完全认错装备" could still clear the 60 pass line at 65).
#   (b) anchors every 20 so interpolation between them is defined.
#   (c) fact_score's 0 anchor no longer conflates "认错装备" with "答非所问/空泛" -- two
#       unrelated failure modes on one anchor made failure-case clustering useless.
# Structurally this is the plain weighted multi-axis design (FLASK / HealthBench shape).
# The contrast against V4 is exactly: no hard gate, no penalty multiplier.
# ---------------------------------------------------------------------------------
V3 = PREAMBLE + """
## 评分方式

从五个维度分别打分，**每维度 0-100 的整数**，分越高越好。维度之间相互独立评分，不要互相影响。锚点如下，可以在锚点之间插值，但要保持锚点代表的严重程度。

**如果某个维度在本题不适用，输出 `null`，不要输出 100。** "不适用"的判断标准：问题没有要求这个维度的内容，且回答也确实没有涉及。只要回答主动涉及了（哪怕问题没问），就要评分。

### param_score（数值参数准确性）
回答中出现的数值型技术参数，字段随装备种类而定（陆战装备的口径/装甲厚度/速度/行程/射程/载员数；舰船的排水量/航速/续航力；飞机的翼展/升限/航程/载弹量；导弹的射程/战斗部重量等）。

- `100`：提到的每个数值都与参考答案一致（允许单位换算、合理取整）。
- `80`：极轻微偏差，不影响理解。
- `60`：一处数值出入较明显（误差超过约 10%），但不是编造。
- `40`：某个关键数值明显错误（数量级错误、张冠李戴到另一装备）。
- `20`：多处数值错误。
- `0`：数值信息系统性错误或几乎全部为编造。
- `null`：**问题不要求数值，且回答也确实没有给出任何具体数值。**

### fact_score（非数值事实准确性）
装备名称/型号、类别归属、所属国家、研制历史、装备关系、服役情况等非数值事实。**这一维度只评事实对错，不评是否切题**（切题性归入 relevance_score）。

- `100`：装备大类与用途认定正确，所问事实要点与参考答案、图片无实质冲突。
- `80`：名称错字或非官方别称，所指装备清晰可辨；或次要事实（服役年份、研制单位）有出入。
- `60`：认错细分型号，装备大类和用途没错。
- `40`：有一处明确的事实与参考答案冲突，可能造成误导。
- `20`：多处事实冲突，或装备类别认定出现明显偏差。
- `0`：完全认错装备大类，或事实内容与参考答案系统性冲突。
- `null`：本题不涉及任何非数值事实（少见）。

### visual_score（图片外观一致性）
只评"看图"这部分：外观、涂装、布局、可见部件、舷号/编号的描述是否与图片吻合。不涉及数值和背景事实。

- `100`：提到的外观特征与图片完全吻合。
- `80`：基本吻合，个别细节表述模糊但不算错。
- `60`：有一处外观细节与图片不符，不影响整体判断。
- `40`：外观描述与图片有明显不符（颜色、部件、编号说错）。
- `20`：严重误读图片内容。
- `0`：描述的外观与图片明显矛盾，或编造了图中不存在的部件/标记。
- `null`：**问题不要求描述外观，且回答也确实没有涉及任何外观特征。**

### fabrication_score（无编造 / 反幻觉）
回答里有没有编造参考答案和图片都无法证实的具体内容。这一维度和 param/visual 里的编造扣分点会有重叠，属于**故意的**——编造值得重复计罚。

- `100`：每一处具体内容都能在参考答案或图片中找到依据。
- `80`：有轻微引申，但没有编造可被证伪的具体细节。
- `60`：基于常识做了泛化描述，边界略模糊但未编造具体数值/型号/事实。
- `40`：编造了个别参考答案和图片都无法证实的具体细节。
- `20`：编造了多处具体细节。
- `0`：大量编造，或编造内容构成了回答的主体。
- 这个维度**永远适用**，不允许 null。

### relevance_score（切题性与表达）
回答是否针对所问、结构是否清晰、有没有言之有物。**与事实对错无关**：内容错误在别的维度扣，这里只评是否回应了问题。

- `100`：直接回应了问题，条理清晰，覆盖要点，详略得当。
- `80`：回应了问题，个别地方啰嗦或遗漏次要要求。
- `60`：基本切题但结构松散，或遗漏了问题的部分要求。
- `40`：明显重复、空话较多，或只回应了问题的一小部分。
- `20`：大体答非所问，只沾到问题的边。
- `0`：完全没有回应问题，只复述问题，或严重不知所云。
- 这个维度**永远适用**，不允许 null。

打分完成后，用一句话说明各维度的主要扣分点。**五个字段必须全部出现**：适用的填 0-100 的整数，不适用的显式填 `null`。即使回答很差导致多个维度都要给低分，也要把每个字段填上——不允许因为"这个问题已经在别的维度提过"就省略。
""" + OUTPUT_RULE.format(version="v3") + """
```
{"rubric": "v3", "param_score": 0-100的整数 或 null, "fact_score": 0-100的整数 或 null, "visual_score": 0-100的整数 或 null, "fabrication_score": 0-100的整数, "relevance_score": 0-100的整数, "reasoning": "逐维度简要说明打分依据"}
```
"""

# ---------------------------------------------------------------------------------
# V4 -- V3's axes plus two structural additions:
#   equipment_correct as a hard cap, and fabrication as a multiplier rather than a
#   weighted term. Tested against V3 this isolates exactly one question: in a domain
#   where misidentification and hallucination are category errors rather than partial
#   credit, does gating beat weighting?
# ---------------------------------------------------------------------------------
V4 = PREAMBLE + """
## 第一步：装备认定（先判这一项）

`equipment_correct`：被测回答有没有认对图中装备？

- `true`：认对**大类和用途**。细分型号说错（052C 说成 052D）、名称错字、非官方别称，仍为 true。
- `false`：认错**大类或用途**（驱逐舰说成航母、运输机说成轰炸机、自行火炮说成坦克），或回答空泛/答非所问到无法判断它是否理解了图与问题。

认错装备会让总分被封顶——在这个领域里认错装备不是扣几分的事。

## 第二步：分维度打分

**每维度 0-100 的整数**，锚点如下，可在锚点之间插值。**不适用的维度输出 `null`，不要输出 100**（判断标准：问题没要求，且回答也确实没涉及）。

### fact_score（非数值事实准确性）
装备名称/型号、类别归属、所属国家、研制历史、装备关系、服役情况等非数值事实。

- `100`：所问事实要点与参考答案、图片完全无冲突。
- `80`：名称错字或非官方别称，所指装备清晰可辨；或次要事实（服役年份、研制单位）有出入。
- `60`：认错细分型号，装备大类和用途没错。
- `40`：有一处明确的事实与参考答案冲突。
- `20`：多处事实冲突。
- `0`：事实内容与参考答案系统性冲突。
- `null`：本题不涉及任何非数值事实（少见）。

### param_score（数值参数准确性）
回答中出现的数值型技术参数，字段随装备种类而定。

- `100`：提到的每个数值都与参考答案一致（允许单位换算、合理取整）。
- `80`：极轻微偏差，不影响理解。
- `60`：一处数值出入较明显（误差超过约 10%）。
- `40`：某个关键数值明显错误（数量级错误、张冠李戴）。
- `20`：多处数值错误。
- `0`：数值信息系统性错误。
- `null`：**问题不要求数值，且回答也确实没有给出任何具体数值。**

### visual_score（图片外观一致性）
只评外观、涂装、布局、可见部件、舷号/编号是否与图片吻合。

- `100`：提到的外观特征与图片完全吻合。
- `80`：基本吻合，个别细节表述模糊但不算错。
- `60`：一处外观细节不符，不影响整体判断。
- `40`：外观描述与图片有明显不符。
- `20`：严重误读图片内容。
- `0`：描述的外观与图片明显矛盾。
- `null`：**问题不要求描述外观，且回答也确实没有涉及任何外观特征。**

### fabrication_score（无编造 / 反幻觉）
有没有编造参考答案和图片都无法证实的具体内容。**这一维度独立作用于总分，编造的惩罚很重。**

- `100`：每一处具体内容都能在参考答案或图片中找到依据。
- `80`：有轻微引申，未编造可被证伪的具体细节。
- `60`：基于常识做了泛化描述，边界略模糊。
- `40`：编造了个别无法证实的具体细节。
- `20`：编造了多处具体细节。
- `0`：大量编造，或编造内容构成回答的主体。
- 这个维度**永远适用**，不允许 null。

### style_score（表达质量）
结构是否清晰、是否覆盖问题要点、有没有言之有物。**与准确性无关**，默认不计入总分，只作记录。

- `100`：条理清晰，覆盖问题要点，详略得当。
- `70`：能看懂，但结构松散或遗漏部分要求。
- `40`：明显重复、空话较多，或结构混乱。
- `0`：完全没有回应问题要求，或严重不知所云。
- 这个维度**永远适用**，不允许 null。

打分完成后，用一句话说明各维度的主要扣分点。**7 个字段必须全部出现**，适用的填整数，不适用的显式填 `null`。
""" + OUTPUT_RULE.format(version="v4") + """
```
{"rubric": "v4", "equipment_correct": true 或 false, "fact_score": 0-100的整数 或 null, "param_score": 0-100的整数 或 null, "visual_score": 0-100的整数 或 null, "fabrication_score": 0-100的整数, "style_score": 0-100的整数, "reasoning": "逐维度简要说明打分依据"}
```
"""

# ---------------------------------------------------------------------------------
# V3B / V4B -- native binary versions of V3 and V4, for the 2x2:
#
#                        continuous            binary
#   weighted (V3)        judge..._v3.txt       judge..._v3b.txt
#   gated    (V4)        judge..._v4.txt       judge..._v4b.txt
#
# These ask the judge for a coarse verdict directly, which is a different experiment from
# thresholding the continuous score afterwards. Post-hoc thresholding is exactly
# equivalent to the continuous run and needs no new judge calls (the `hit` column already
# is quality_score >= 60), so it measures only what the REPORT loses. These prompts
# measure what the JUDGE does differently when asked for one bit instead of five numbers
# -- coarse judgments can be more reliable per call even though they carry less.
#
# Both keep the same criteria and the same anchors as their continuous counterparts, so
# the only thing varying down each column of the 2x2 is output granularity.
# ---------------------------------------------------------------------------------

_ASPECT_LIST = """
`aspect_failures` 列出所有**出现实质问题**的方面（没有则填 `[]`）。这个字段**不参与判定**，
只用于事后聚类看模型主要错在哪：

- `"param"`：数值参数明显错误（误差超过约 10%、数量级错误、张冠李戴）
- `"fact"`：非数值事实与参考答案冲突
- `"visual"`：外观描述与图片不符
- `"fabrication"`：编造参考答案和图片都无法证实的具体内容
- `"relevance"`：答非所问、空泛、只复述问题
"""

V3B = PREAMBLE + """
## 判定方式

这一版只要一个二值判定，但**判定的标准与连续版逐条对应**：先在心里过一遍下面五个方面，
只有全部没有实质问题时才判通过。方面在本题不适用时（问题没要求、回答也没涉及）直接跳过，
不要因为"没涉及"就当成问题。

### 逐方面的通过门槛

- **param（数值参数）**：提到的数值与参考答案一致，允许单位换算、合理取整、四舍五入误差。
  出现误差超过约 10% 的出入、数量级错误、把别的装备参数张冠李戴 —— 不通过。
- **fact（非数值事实）**：装备大类与用途认定正确，所问事实要点与参考答案、图片无实质冲突。
  细分型号说错（052C 说成 052D）、名称错字、非官方别称 —— **仍算通过**。
  认错大类或用途、关键事实与参考直接冲突 —— 不通过。
- **visual（图片外观一致性）**：提到的外观、涂装、布局、可见部件、舷号/编号与图片吻合。
  个别细节表述模糊算通过；颜色/部件/编号说错、严重误读图片 —— 不通过。
- **fabrication（无编造）**：每一处具体内容都能在参考答案或图片中找到依据。
  基于常识的泛化描述算通过；编造了可被证伪的具体数值/型号/事实 —— 不通过。
- **relevance（切题性）**：直接回应了所问，结构可读，言之有物。
  大体答非所问、只沾到问题的边、只复述问题 —— 不通过。

### 总判定

**`correct = true`**：所有适用方面都通过。
**`correct = false`**：任一适用方面不通过。
""" + _ASPECT_LIST + OUTPUT_RULE.format(version="v3b") + """
```
{"rubric": "v3b", "correct": true 或 false, "aspect_failures": 字符串数组（无问题填 []）, "reason": "一句话理由，指出关键判定依据"}
```
"""

V4B = PREAMBLE + """
## 判定方式

这一版只要一个二值判定，但采用**门控**结构：按顺序过三道关，任何一道不过则整体不通过。
这与连续版的封顶和乘子逻辑对应 —— 认错装备和编造在本领域是品类错误，不存在部分给分。

### 第一道关：装备认定（`equipment_correct`）

- 认对**大类和用途**即为 true。细分型号说错（052C 说成 052D）、名称错字、非官方别称，仍为 true。
- 认错**大类或用途**（驱逐舰说成航母、运输机说成轰炸机、自行火炮说成坦克），
  或回答空泛/答非所问到无法判断它是否理解了图与问题 —— false。

**这一关不过，整体直接判 `correct = false`**，后面两关不用再看（但仍要如实填写）。

### 第二道关：无编造

每一处具体内容都能在参考答案或图片中找到依据。基于常识的泛化描述可以接受；
一旦编造了可被证伪的具体数值、型号或事实 —— 整体判 false。

### 第三道关：适用方面的准确性

对本题适用的方面逐一检查（不适用则跳过，不要因为"没涉及"就当成问题）：

- **param（数值参数）**：数值与参考答案一致，允许单位换算、合理取整。出现误差超过约 10%
  的出入、数量级错误、张冠李戴 —— 不通过。
- **visual（图片外观一致性）**：提到的外观、涂装、布局、可见部件、舷号/编号与图片吻合。
  个别细节表述模糊算通过；颜色/部件/编号说错、严重误读图片 —— 不通过。
- **fact（非数值事实的其余部分）**：所问事实要点与参考答案无实质冲突。次要事实
  （服役年份、研制单位）有出入 —— 不通过。

### 总判定

**`correct = true`**：三道关全过。
**`correct = false`**：任一道关不过。

表达质量（结构、详略、啰嗦程度）**不参与判定** —— 写得漂亮的错答案不该因此通过，
简短准确的也不该因此不通过。
""" + _ASPECT_LIST + OUTPUT_RULE.format(version="v4b") + """
```
{"rubric": "v4b", "correct": true 或 false, "equipment_correct": true 或 false, "aspect_failures": 字符串数组（无问题填 []）, "reason": "一句话理由，指出关键判定依据"}
```
"""

VERSIONS = {"v1": V1, "v2": V2, "v3": V3, "v3b": V3B, "v4": V4, "v4b": V4B}


def write_all() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in VERSIONS.items():
        path = OUT_DIR / f"judge_equip_pointwise_{name}.txt"
        path.write_text(text.strip() + "\n", encoding="utf-8")
        print(f"wrote {path}  ({len(text.strip())} chars)")


def check() -> int:
    bad = 0
    for name, text in VERSIONS.items():
        if PREAMBLE.strip() not in text:
            print(f"FAIL {name}: shared preamble not intact")
            bad = 1
        if f'"{name}"' not in text:
            print(f"FAIL {name}: rubric tag missing")
            bad = 1
    if not bad:
        print(f"OK: shared preamble byte-identical in all {len(VERSIONS)} versions "
              f"({len(PREAMBLE.strip())} chars), rubric tags present")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.exit(check() if args.check else (write_all() or 0))
