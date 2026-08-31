"""
敏感词实时监测模块（脱敏版核心逻辑）

功能：基于Whisper语音转文字 + AC自动机四大类违禁词库实时匹配 + 三级预警触发
场景：政府采购远程异地评标专家不当言论实时监测

四大类违禁词库：
1. 核心违禁词与高风险表述（放水、抬分、压分、改分等）
2. 敏感信息泄露类（内定、指定、推荐、选这家等）
3. 倾向性言论类（这家不行、那家好、肯定行等）
4. 非正常评价类（凭经验、随便打分、不用看细则等）

三级预警机制：
- 黄色预警：首次轻微违规，音频提示+侧边浮窗
- 橙色警示：重复/明显违规，音频警告+中央弹窗+片段回放
- 红色警报：重大违规（改分、内定等），全屏警告+建议暂停+证据包
"""

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple


# ============================================================
# 一、四大类违禁词库定义
# ============================================================

class SensitiveCategory(Enum):
    """违禁词类别"""
    CORE_FORBIDDEN = "核心违禁词与高风险表述"
    SENSITIVE_LEAK = "敏感信息泄露类"
    TENDENTIOUS_SPEECH = "倾向性言论类"
    ABNORMAL_EVALUATION = "非正常评价类"


class SeverityLevel(Enum):
    """严重等级"""
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"


@dataclass
class SensitiveWord:
    """违禁词定义"""
    word: str
    category: SensitiveCategory
    severity: SeverityLevel
    description: str = ""


# 第一大类：核心违禁词与高风险表述
CORE_FORBIDDEN_WORDS = [
    SensitiveWord("放水", SensitiveCategory.CORE_FORBIDDEN, SeverityLevel.HIGH, "暗示放宽评审标准"),
    SensitiveWord("松点", SensitiveCategory.CORE_FORBIDDEN, SeverityLevel.HIGH, "暗示降低要求"),
    SensitiveWord("严点", SensitiveCategory.CORE_FORBIDDEN, SeverityLevel.MEDIUM, "暗示提高要求（针对特定供应商）"),
    SensitiveWord("抬分", SensitiveCategory.CORE_FORBIDDEN, SeverityLevel.HIGH, "暗示提高某供应商分数"),
    SensitiveWord("压分", SensitiveCategory.CORE_FORBIDDEN, SeverityLevel.HIGH, "暗示压低某供应商分数"),
    SensitiveWord("加分", SensitiveCategory.CORE_FORBIDDEN, SeverityLevel.HIGH, "暗示给某供应商额外加分"),
    SensitiveWord("减分", SensitiveCategory.CORE_FORBIDDEN, SeverityLevel.HIGH, "暗示给某供应商额外减分"),
    SensitiveWord("凑分", SensitiveCategory.CORE_FORBIDDEN, SeverityLevel.HIGH, "暗示拼凑分数达到特定结果"),
    SensitiveWord("改分", SensitiveCategory.CORE_FORBIDDEN, SeverityLevel.HIGH, "暗示修改已评定的分数"),
    SensitiveWord("抹掉", SensitiveCategory.CORE_FORBIDDEN, SeverityLevel.HIGH, "暗示忽略或删除某问题"),
    SensitiveWord("忽略", SensitiveCategory.CORE_FORBIDDEN, SeverityLevel.MEDIUM, "暗示忽略某供应商的问题"),
    SensitiveWord("跳过", SensitiveCategory.CORE_FORBIDDEN, SeverityLevel.MEDIUM, "暗示跳过某评审环节"),
    SensitiveWord("不用看", SensitiveCategory.CORE_FORBIDDEN, SeverityLevel.MEDIUM, "暗示不用仔细评审某内容"),
]

# 第二大类：敏感信息泄露类
SENSITIVE_LEAK_WORDS = [
    SensitiveWord("内定", SensitiveCategory.SENSITIVE_LEAK, SeverityLevel.HIGH, "暗示结果已提前确定"),
    SensitiveWord("指定", SensitiveCategory.SENSITIVE_LEAK, SeverityLevel.HIGH, "暗示指定某供应商"),
    SensitiveWord("推荐", SensitiveCategory.SENSITIVE_LEAK, SeverityLevel.MEDIUM, "暗示推荐某供应商"),
    SensitiveWord("选这家", SensitiveCategory.SENSITIVE_LEAK, SeverityLevel.HIGH, "直接表达选择倾向"),
    SensitiveWord("要这个", SensitiveCategory.SENSITIVE_LEAK, SeverityLevel.HIGH, "直接表达选择倾向"),
    SensitiveWord("定这个", SensitiveCategory.SENSITIVE_LEAK, SeverityLevel.HIGH, "直接表达确定倾向"),
    SensitiveWord("倾向", SensitiveCategory.SENSITIVE_LEAK, SeverityLevel.MEDIUM, "表达倾向性态度"),
    SensitiveWord("偏爱", SensitiveCategory.SENSITIVE_LEAK, SeverityLevel.MEDIUM, "表达偏爱态度"),
    SensitiveWord("看好", SensitiveCategory.SENSITIVE_LEAK, SeverityLevel.MEDIUM, "表达看好态度"),
    SensitiveWord("中意", SensitiveCategory.SENSITIVE_LEAK, SeverityLevel.MEDIUM, "表达中意态度"),
    SensitiveWord("首选", SensitiveCategory.SENSITIVE_LEAK, SeverityLevel.MEDIUM, "暗示首选某供应商"),
    SensitiveWord("必选", SensitiveCategory.SENSITIVE_LEAK, SeverityLevel.HIGH, "暗示必选某供应商"),
    SensitiveWord("淘汰", SensitiveCategory.SENSITIVE_LEAK, SeverityLevel.HIGH, "直接表达淘汰倾向"),
    SensitiveWord("留下", SensitiveCategory.SENSITIVE_LEAK, SeverityLevel.HIGH, "直接表达保留倾向"),
    SensitiveWord("放弃", SensitiveCategory.SENSITIVE_LEAK, SeverityLevel.HIGH, "直接表达放弃倾向"),
]

# 第三大类：倾向性言论类
TENDENTIOUS_WORDS = [
    SensitiveWord("不行", SensitiveCategory.TENDENTIOUS_SPEECH, SeverityLevel.LOW, "简单否定（无依据）"),
    SensitiveWord("不好", SensitiveCategory.TENDENTIOUS_SPEECH, SeverityLevel.LOW, "简单否定（无依据）"),
    SensitiveWord("差", SensitiveCategory.TENDENTIOUS_SPEECH, SeverityLevel.LOW, "简单否定（无依据）"),
    SensitiveWord("烂", SensitiveCategory.TENDENTIOUS_SPEECH, SeverityLevel.MEDIUM, "情绪化否定"),
    SensitiveWord("垃圾", SensitiveCategory.TENDENTIOUS_SPEECH, SeverityLevel.MEDIUM, "情绪化否定"),
    SensitiveWord("不行就换", SensitiveCategory.TENDENTIOUS_SPEECH, SeverityLevel.MEDIUM, "暗示随意更换"),
    SensitiveWord("这家不行", SensitiveCategory.TENDENTIOUS_SPEECH, SeverityLevel.MEDIUM, "直接否定某供应商"),
    SensitiveWord("那家好", SensitiveCategory.TENDENTIOUS_SPEECH, SeverityLevel.MEDIUM, "直接肯定某供应商"),
    SensitiveWord("这家靠谱", SensitiveCategory.TENDENTIOUS_SPEECH, SeverityLevel.LOW, "主观肯定某供应商"),
    SensitiveWord("那家不靠谱", SensitiveCategory.TENDENTIOUS_SPEECH, SeverityLevel.MEDIUM, "主观否定某供应商"),
    SensitiveWord("肯定行", SensitiveCategory.TENDENTIOUS_SPEECH, SeverityLevel.MEDIUM, "绝对化肯定"),
    SensitiveWord("绝对不行", SensitiveCategory.TENDENTIOUS_SPEECH, SeverityLevel.MEDIUM, "绝对化否定"),
    SensitiveWord("没问题", SensitiveCategory.TENDENTIOUS_SPEECH, SeverityLevel.LOW, "简单肯定（无依据）"),
    SensitiveWord("有问题", SensitiveCategory.TENDENTIOUS_SPEECH, SeverityLevel.LOW, "简单否定（无依据）"),
]

# 第四大类：非正常评价类
ABNORMAL_EVALUATION_WORDS = [
    SensitiveWord("凭经验", SensitiveCategory.ABNORMAL_EVALUATION, SeverityLevel.MEDIUM, "暗示不按标准凭经验评审"),
    SensitiveWord("随便", SensitiveCategory.ABNORMAL_EVALUATION, SeverityLevel.MEDIUM, "暗示随意评审"),
    SensitiveWord("不用看细则", SensitiveCategory.ABNORMAL_EVALUATION, SeverityLevel.HIGH, "暗示不按评审标准"),
    SensitiveWord("不用卡太严", SensitiveCategory.ABNORMAL_EVALUATION, SeverityLevel.MEDIUM, "暗示降低标准"),
    SensitiveWord("松一点", SensitiveCategory.ABNORMAL_EVALUATION, SeverityLevel.MEDIUM, "暗示降低标准"),
    SensitiveWord("直接过", SensitiveCategory.ABNORMAL_EVALUATION, SeverityLevel.MEDIUM, "暗示不用评审直接通过"),
    SensitiveWord("直接淘汰", SensitiveCategory.ABNORMAL_EVALUATION, SeverityLevel.MEDIUM, "暗示不用评审直接淘汰"),
    SensitiveWord("小问题忽略", SensitiveCategory.ABNORMAL_EVALUATION, SeverityLevel.MEDIUM, "暗示忽略问题"),
    SensitiveWord("细节不用看", SensitiveCategory.ABNORMAL_EVALUATION, SeverityLevel.MEDIUM, "暗示忽略细节"),
    SensitiveWord("评完走人", SensitiveCategory.ABNORMAL_EVALUATION, SeverityLevel.MEDIUM, "暗示不认真评审"),
    SensitiveWord("不用太认真", SensitiveCategory.ABNORMAL_EVALUATION, SeverityLevel.MEDIUM, "暗示不认真评审"),
    SensitiveWord("打打分就行", SensitiveCategory.ABNORMAL_EVALUATION, SeverityLevel.MEDIUM, "暗示敷衍评审"),
]

# 完整句式违禁词（高风险，需语义匹配）
HIGH_RISK_PATTERNS = [
    (r"让我关照一下", SensitiveCategory.CORE_FORBIDDEN, SeverityLevel.HIGH),
    (r"特别关注这个标", SensitiveCategory.CORE_FORBIDDEN, SeverityLevel.HIGH),
    (r"我给.*打了.*分", SensitiveCategory.CORE_FORBIDDEN, SeverityLevel.HIGH),
    (r"标底是", SensitiveCategory.CORE_FORBIDDEN, SeverityLevel.HIGH),
    (r"入围名单已经定了", SensitiveCategory.CORE_FORBIDDEN, SeverityLevel.HIGH),
    (r"这个标肯定是.*的", SensitiveCategory.SENSITIVE_LEAK, SeverityLevel.HIGH),
    (r"我建议大家都投", SensitiveCategory.SENSITIVE_LEAK, SeverityLevel.HIGH),
    (r"量身定做", SensitiveCategory.SENSITIVE_LEAK, SeverityLevel.HIGH),
    (r"反正没人看", SensitiveCategory.ABNORMAL_EVALUATION, SeverityLevel.MEDIUM),
    (r"快点评", SensitiveCategory.ABNORMAL_EVALUATION, SeverityLevel.MEDIUM),
]


# ============================================================
# 二、AC自动机多模式匹配
# ============================================================

class ACTrieNode:
    """AC自动机节点"""
    def __init__(self):
        self.children = {}
        self.fail = None
        self.output = []  # 存储匹配到的违禁词


class AhoCorasick:
    """AC自动机多模式匹配引擎"""

    def __init__(self):
        self.root = ACTrieNode()
        self.word_map = {}  # word -> SensitiveWord

    def add_word(self, word: SensitiveWord):
        """添加违禁词到Trie树"""
        node = self.root
        for char in word.word:
            if char not in node.children:
                node.children[char] = ACTrieNode()
            node = node.children[char]
        node.output.append(word.word)
        self.word_map[word.word] = word

    def build(self):
        """构建失败指针（BFS）"""
        from collections import deque
        queue = deque()
        for child in self.root.children.values():
            child.fail = self.root
            queue.append(child)

        while queue:
            current_node = queue.popleft()
            for char, child in current_node.children.items():
                queue.append(child)
                fail_node = current_node.fail
                while fail_node and char not in fail_node.children:
                    fail_node = fail_node.fail
                child.fail = fail_node.children[char] if fail_node and char in fail_node.children else self.root
                child.output += child.fail.output

    def search(self, text: str) -> List[Tuple[str, int, SensitiveWord]]:
        """在文本中搜索所有匹配的违禁词，返回(词, 位置, 词定义)"""
        results = []
        node = self.root
        for i, char in enumerate(text):
            while node and char not in node.children:
                node = node.fail
            if not node:
                node = self.root
                continue
            node = node.children[char]
            for word in node.output:
                if word in self.word_map:
                    results.append((word, i - len(word) + 1, self.word_map[word]))
        return results


# ============================================================
# 三、三级预警机制
# ============================================================

class AlertLevel(Enum):
    """预警等级"""
    YELLOW = "黄色预警"
    ORANGE = "橙色警示"
    RED = "红色警报"


@dataclass
class AlertRecord:
    """预警记录"""
    alert_id: str
    level: AlertLevel
    category: SensitiveCategory
    sensitive_word: str
    severity: SeverityLevel
    timestamp: float
    expert_id: str
    context_text: str
    match_position: int
    audio_clip_ref: str = ""  # 音频片段引用（脱敏）
    is_escalated: bool = False
    is_verified: bool = False


@dataclass
class ExpertViolationStats:
    """专家违规统计"""
    expert_id: str
    yellow_count: int = 0
    orange_count: int = 0
    red_count: int = 0
    category_counts: Dict[str, int] = field(default_factory=dict)
    last_alert_time: float = 0
    last_alert_category: Optional[SensitiveCategory] = None


class ThreeLevelAlertSystem:
    """三级预警系统"""

    # 预警间隔（秒），同一专家同类行为的最小预警间隔
    ALERT_INTERVAL = 60

    def __init__(self):
        self.expert_stats: Dict[str, ExpertViolationStats] = {}
        self.alert_history: List[AlertRecord] = []

    def _get_expert_stats(self, expert_id: str) -> ExpertViolationStats:
        if expert_id not in self.expert_stats:
            self.expert_stats[expert_id] = ExpertViolationStats(expert_id=expert_id)
        return self.expert_stats[expert_id]

    def determine_alert_level(
        self,
        sensitive_word: SensitiveWord,
        expert_id: str,
        current_time: float
    ) -> AlertLevel:
        """
        根据违禁词严重等级和专家违规历史，确定预警等级

        规则：
        - 高严重等级（改分、内定等）：首次即橙色或红色
        - 中严重等级：首次黄色，重复橙色，多次红色
        - 低严重等级：首次黄色，重复橙色
        - 同一专家同类违规累计≥2次：升级
        - 同一专家同类违规累计≥3次：再次升级
        """
        stats = self._get_expert_stats(expert_id)
        category_key = sensitive_word.category.value

        # 累计同类违规次数
        category_count = stats.category_counts.get(category_key, 0)

        # 基础等级判定
        if sensitive_word.severity == SeverityLevel.HIGH:
            base_level = AlertLevel.ORANGE
            if category_count >= 1:  # 已有同类高风险违规
                base_level = AlertLevel.RED
        elif sensitive_word.severity == SeverityLevel.MEDIUM:
            base_level = AlertLevel.YELLOW
            if category_count >= 1:
                base_level = AlertLevel.ORANGE
            if category_count >= 2:
                base_level = AlertLevel.RED
        else:  # LOW
            base_level = AlertLevel.YELLOW
            if category_count >= 2:
                base_level = AlertLevel.ORANGE

        # 检查是否在预警间隔内（避免重复预警）
        if (current_time - stats.last_alert_time < self.ALERT_INTERVAL
                and stats.last_alert_category == sensitive_word.category):
            return None  # 间隔内不重复预警

        return base_level

    def trigger_alert(
        self,
        sensitive_word: SensitiveWord,
        expert_id: str,
        context_text: str,
        match_position: int,
        current_time: float,
        audio_clip_ref: str = ""
    ) -> Optional[AlertRecord]:
        """触发预警"""
        alert_level = self.determine_alert_level(sensitive_word, expert_id, current_time)
        if alert_level is None:
            return None

        # 创建预警记录
        alert = AlertRecord(
            alert_id=f"ALERT-{int(current_time)}-{len(self.alert_history)+1:04d}",
            level=alert_level,
            category=sensitive_word.category,
            sensitive_word=sensitive_word.word,
            severity=sensitive_word.severity,
            timestamp=current_time,
            expert_id=expert_id,
            context_text=context_text,
            match_position=match_position,
            audio_clip_ref=audio_clip_ref
        )

        # 更新专家统计
        stats = self._get_expert_stats(expert_id)
        if alert_level == AlertLevel.YELLOW:
            stats.yellow_count += 1
        elif alert_level == AlertLevel.ORANGE:
            stats.orange_count += 1
        elif alert_level == AlertLevel.RED:
            stats.red_count += 1
        stats.category_counts[sensitive_word.category.value] = \
            stats.category_counts.get(sensitive_word.category.value, 0) + 1
        stats.last_alert_time = current_time
        stats.last_alert_category = sensitive_word.category

        self.alert_history.append(alert)

        # 执行预警动作
        self._execute_alert_action(alert)

        return alert

    def _execute_alert_action(self, alert: AlertRecord):
        """
        执行预警动作（脱敏版，实际系统中调用音视频提示接口）

        黄色预警：音频提示"请注意评标纪律，请使用中性表述" + 侧边浮窗5秒
        橙色警示：音频警告"系统多次检测到异常表述，请立即停止违规评审言行" + 中央弹窗+片段回放
        红色警报：音频警告"因涉嫌违规，建议立即暂停评审活动" + 全屏红色+证据包生成
        """
        if alert.level == AlertLevel.YELLOW:
            # 实际调用：audio_player.play("请注意评标纪律，请使用中性表述。")
            # 实际调用：ui.show_sidebar_toast("【合规提示】请注意评标纪律", duration=5000)
            pass
        elif alert.level == AlertLevel.ORANGE:
            # 实际调用：audio_player.play("系统多次检测到异常表述，请立即停止违规评审言行。")
            # 实际调用：ui.show_center_modal("【警告！】系统多次检测到异常表述", alert)
            # 实际调用：notify_host(alert)  # 通知主持人
            pass
        elif alert.level == AlertLevel.RED:
            # 实际调用：audio_player.play("因涉嫌违规，建议立即暂停评审活动...")
            # 实际调用：ui.show_fullscreen_alert("【警告！涉嫌违规，建议立即暂停评审活动】", alert)
            # 实际调用：generate_evidence_package(alert)  # 生成证据包
            # 实际调用：notify_supervisor(alert)  # 通知监管部门
            pass


# ============================================================
# 四、语义消歧（上下文理解）
# ============================================================

class SemanticDisambiguator:
    """
    语义消歧器：结合上下文理解，区分正常讨论和违规言论

    实际系统中使用大语言模型进行语义理解，此处为规则版脱敏实现。
    """

    # 正常场景关键词（出现这些词时，可能是正常讨论）
    NORMAL_CONTEXT_KEYWORDS = [
        "不符合", "不满足", "不达标", "参数", "要求", "标准",
        "评审细则", "招标文件", "评分标准", "条款", "规定",
        "根据", "依据", "按照", "对照"
    ]

    # 违规场景关键词（出现这些词时，更可能是违规言论）
    VIOLATION_CONTEXT_KEYWORDS = [
        "不用看", "直接", "随便", "凭经验", "心里有数",
        "关照", "关注", "定了", "肯定", "绝对"
    ]

    def is_likely_violation(
        self,
        text: str,
        sensitive_word: SensitiveWord,
        match_position: int
    ) -> Tuple[bool, float]:
        """
        判断是否更可能是违规言论，返回(是否违规, 置信度)

        规则：
        - 高风险违禁词（改分、内定等）：直接判定违规，置信度0.9
        - 低风险倾向性词汇（不行、不好等）：检查上下文
          - 有"不符合""不达标"等正常讨论关键词 → 可能正常，置信度0.3
          - 有"不用看""直接"等违规关键词 → 可能违规，置信度0.8
          - 无上下文 → 疑似违规，置信度0.6
        """
        if sensitive_word.severity == SeverityLevel.HIGH:
            return True, 0.9

        # 提取上下文（匹配位置前后各20个字符）
        start = max(0, match_position - 20)
        end = min(len(text), match_position + len(sensitive_word.word) + 20)
        context = text[start:end]

        # 检查正常场景关键词
        normal_score = sum(1 for kw in self.NORMAL_CONTEXT_KEYWORDS if kw in context)
        violation_score = sum(1 for kw in self.VIOLATION_CONTEXT_KEYWORDS if kw in context)

        if sensitive_word.severity == SeverityLevel.LOW:
            if normal_score > 0 and violation_score == 0:
                return False, 0.3  # 更可能是正常讨论
            elif violation_score > 0:
                return True, 0.8
            else:
                return True, 0.6
        else:  # MEDIUM
            if normal_score > 0 and violation_score == 0:
                return True, 0.5  # 中等风险即使有正常上下文也保留预警
            elif violation_score > 0:
                return True, 0.85
            else:
                return True, 0.7


# ============================================================
# 五、敏感词监测主流程
# ============================================================

class SensitiveWordMonitor:
    """敏感词实时监测器"""

    def __init__(self):
        # 初始化AC自动机
        self.ac = AhoCorasick()
        all_words = (
            CORE_FORBIDDEN_WORDS + SENSITIVE_LEAK_WORDS +
            TENDENTIOUS_WORDS + ABNORMAL_EVALUATION_WORDS
        )
        for word in all_words:
            self.ac.add_word(word)
        self.ac.build()

        # 初始化三级预警系统
        self.alert_system = ThreeLevelAlertSystem()

        # 初始化语义消歧器
        self.disambiguator = SemanticDisambiguator()

        print(f"[初始化] 违禁词库加载完成，共 {len(all_words)} 个违禁词")
        print(f"  - 核心违禁词: {len(CORE_FORBIDDEN_WORDS)}")
        print(f"  - 敏感信息泄露: {len(SENSITIVE_LEAK_WORDS)}")
        print(f"  - 倾向性言论: {len(TENDENTIOUS_WORDS)}")
        print(f"  - 非正常评价: {len(ABNORMAL_EVALUATION_WORDS)}")

    def process_transcript(
        self,
        text: str,
        expert_id: str,
        current_time: float = None,
        audio_clip_ref: str = ""
    ) -> List[AlertRecord]:
        """
        处理语音转文字结果，检测违禁词并触发预警

        Args:
            text: ASR转写的文本
            expert_id: 发言人ID（通过声纹分离识别）
            current_time: 当前时间戳
            audio_clip_ref: 音频片段引用

        Returns:
            触发的预警记录列表
        """
        if current_time is None:
            current_time = time.time()

        alerts = []

        # 1. AC自动机匹配违禁词
        matches = self.ac.search(text)

        # 2. 高风险句式正则匹配
        for pattern, category, severity in HIGH_RISK_PATTERNS:
            for match in re.finditer(pattern, text):
                sw = SensitiveWord(
                    word=match.group(),
                    category=category,
                    severity=severity,
                    description="高风险句式匹配"
                )
                matches.append((match.group(), match.start(), sw))

        # 3. 去重（同一位置只保留最严重的）
        unique_matches = {}
        for word, pos, sw in matches:
            key = (pos, word)
            if key not in unique_matches or sw.severity.value > unique_matches[key][2].severity.value:
                unique_matches[key] = (word, pos, sw)

        # 4. 语义消歧 + 预警触发
        for word, pos, sw in unique_matches.values():
            is_violation, confidence = self.disambiguator.is_likely_violation(text, sw, pos)

            if not is_violation and confidence < 0.5:
                continue  # 判定为正常讨论，跳过

            # 提取上下文（用于预警记录）
            ctx_start = max(0, pos - 10)
            ctx_end = min(len(text), pos + len(word) + 10)
            context = text[ctx_start:ctx_end]

            # 触发预警
            alert = self.alert_system.trigger_alert(
                sensitive_word=sw,
                expert_id=expert_id,
                context_text=context,
                match_position=pos,
                current_time=current_time,
                audio_clip_ref=audio_clip_ref
            )
            if alert:
                alerts.append(alert)

        return alerts

    def get_summary(self) -> Dict:
        """获取监测汇总统计"""
        total_experts = len(self.alert_system.expert_stats)
        total_alerts = len(self.alert_system.alert_history)
        yellow = sum(s.yellow_count for s in self.alert_system.expert_stats.values())
        orange = sum(s.orange_count for s in self.alert_system.expert_stats.values())
        red = sum(s.red_count for s in self.alert_system.expert_stats.values())

        return {
            "total_experts_monitored": total_experts,
            "total_alerts": total_alerts,
            "yellow_alerts": yellow,
            "orange_alerts": orange,
            "red_alerts": red,
            "expert_details": {
                eid: {
                    "yellow": s.yellow_count,
                    "orange": s.orange_count,
                    "red": s.red_count,
                    "categories": s.category_counts
                }
                for eid, s in self.alert_system.expert_stats.items()
            }
        }


# ============================================================
# 六、使用示例
# ============================================================

if __name__ == "__main__":
    # 初始化监测器
    monitor = SensitiveWordMonitor()

    # 模拟语音转文字结果
    test_cases = [
        ("专家A", "这家供应商的技术方案明显比其他家强，我觉得选他们比较合适。"),
        ("专家B", "这个参数不符合要求，所以这一项不行。"),  # 正常讨论，应消歧
        ("专家A", "不用看评审细则了，凭经验选就行，这家肯定行。"),
        ("专家C", "我给XX单位打了85分，你们参考下。"),  # 高风险，泄露分数
        ("专家B", "这个技术参数不达标，但我建议通过，松一点没关系。"),
    ]

    print("\n=== 开始监测 ===\n")
    for expert_id, text in test_cases:
        print(f"[{expert_id}] 发言: {text}")
        alerts = monitor.process_transcript(text, expert_id)
        for alert in alerts:
            print(f"  → {alert.level.value}: 检测到「{alert.sensitive_word}」"
                  f"({alert.category.value}, {alert.severity.value})")
        print()

    # 输出汇总
    print("=== 监测汇总 ===")
    summary = monitor.get_summary()
    print(f"监控专家数: {summary['total_experts_monitored']}")
    print(f"预警总数: {summary['total_alerts']}")
    print(f"  黄色: {summary['yellow_alerts']}")
    print(f"  橙色: {summary['orange_alerts']}")
    print(f"  红色: {summary['red_alerts']}")
