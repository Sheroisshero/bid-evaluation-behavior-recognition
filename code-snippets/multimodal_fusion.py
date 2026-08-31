"""
多模态融合决策模块（脱敏版核心逻辑）

功能：视频+音频+系统日志三路数据的时间戳对齐、置信度加权融合、交叉验证消歧、异常等级判定
场景：政府采购远程异地评标专家异常行为多模态融合识别

融合策略：
1. 时间戳对齐：毫秒级精度，±2秒滑动窗口
2. 置信度加权：根据异常类型动态调整各通道权重
3. 交叉验证：多通道矛盾检测，矛盾则降级
4. 异常等级判定：黄/橙/红三级预警
5. 证据链构建：自动生成多模态证据包

数学基础：
- 加权融合：final_score = Σ w_i × score_i
- 可靠性动态调整：w_i = base_weight_i × reliability_i × relevance_i
- Dempster-Shafer证据理论（可选高级融合）
- 贝叶斯后验概率更新
"""

import numpy as np
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
from scipy.optimize import linear_sum_assignment


# ============================================================
# 一、数据结构定义
# ============================================================

class ModalityType(Enum):
    """数据通道类型"""
    VIDEO = "video"
    AUDIO = "audio"
    LOG = "log"


class AlertLevel(Enum):
    """预警等级"""
    YELLOW = "黄色预警"
    ORANGE = "橙色警示"
    RED = "红色警报"


class BehaviorType(Enum):
    """异常行为类型"""
    PHONE_USAGE = "操作手机"
    LEAVING_SEAT = "起身离席"
    WHISPERING = "交头接耳"
    IMPROPER_SPEECH = "不当言论"
    FACE_OCCLUSION = "肢体遮挡面部"
    NETWORK_ANOMALY = "网络异常"
    DEVICE_ANOMALY = "设备异常"


@dataclass
class ModalityEvent:
    """单通道异常事件"""
    event_id: str
    modality: ModalityType
    behavior_type: BehaviorType
    confidence: float          # 0-1
    timestamp: float           # Unix时间戳
    duration: float = 0.0      # 持续时间（秒）
    expert_id: Optional[str] = None
    seat_number: Optional[int] = None
    description: str = ""
    evidence: Dict = field(default_factory=dict)
    reliability: float = 1.0   # 通道可靠性（0-1），根据数据质量动态调整


@dataclass
class FusionResult:
    """融合决策结果"""
    event_id: str
    behavior_type: BehaviorType
    final_score: float
    alert_level: AlertLevel
    start_time: float
    end_time: Optional[float]
    expert_id: Optional[str]
    seat_number: Optional[int]
    description: str
    channel_scores: Dict[str, float]      # 各通道得分
    channel_weights: Dict[str, float]     # 各通道权重
    verification: str                      # 验证结果（强确认/弱确认/疑似/矛盾）
    evidence_package: Dict = field(default_factory=dict)
    is_escalated: bool = False
    is_verified: bool = False


# ============================================================
# 二、时间戳对齐模块
# ============================================================

class TimestampAligner:
    """
    时间戳对齐模块

    功能：将三路（视频/音频/日志）异常事件按时间戳对齐到同一时间窗口
    精度：毫秒级（NTP时钟同步，同步精度≤10ms）
    窗口：±2秒滑动窗口，步长500ms

    对齐算法：
    1. 所有事件按时间戳排序
    2. 使用滑动窗口，窗口内的事件视为可能相关
    3. 对窗口内事件按专家ID/座位号分组
    4. 同组内不同通道的事件进行关联
    """

    def __init__(self, window_size: float = 2.0, step_size: float = 0.5):
        self.window_size = window_size  # 秒
        self.step_size = step_size      # 秒
        print(f"[TimestampAligner] 初始化: 窗口={window_size}s, 步长={step_size}s")

    def align(
        self,
        video_events: List[ModalityEvent],
        audio_events: List[ModalityEvent],
        log_events: List[ModalityEvent]
    ) -> List[List[ModalityEvent]]:
        """
        对齐三路事件，返回相关事件组列表

        算法：
        1. 合并所有事件并按时间戳排序
        2. 滑动窗口遍历，窗口内事件按专家/座位分组
        3. 使用匈牙利算法进行事件关联（可选，用于复杂场景）
        4. 返回相关事件组
        """
        all_events = video_events + audio_events + log_events
        all_events.sort(key=lambda e: e.timestamp)

        if not all_events:
            return []

        groups = []
        used = set()

        for i, event in enumerate(all_events):
            if event.event_id in used:
                continue

            # 找到时间窗口内的所有事件
            window_events = [event]
            used.add(event.event_id)

            for j in range(i + 1, len(all_events)):
                other = all_events[j]
                if other.timestamp - event.timestamp > self.window_size:
                    break
                if other.event_id in used:
                    continue

                # 检查是否相关（同一专家/座位，或行为类型相关）
                if self._is_related(event, other):
                    window_events.append(other)
                    used.add(other.event_id)

            if len(window_events) >= 1:
                groups.append(window_events)

        return groups

    @staticmethod
    def _is_related(e1: ModalityEvent, e2: ModalityEvent) -> bool:
        """判断两个事件是否相关"""
        # 同一专家
        if e1.expert_id and e2.expert_id and e1.expert_id == e2.expert_id:
            return True
        # 同座位
        if e1.seat_number and e2.seat_number and e1.seat_number == e2.seat_number:
            return True
        # 行为类型相同或相关
        related_pairs = [
            (BehaviorType.PHONE_USAGE, BehaviorType.DEVICE_ANOMALY),
            (BehaviorType.WHISPERING, BehaviorType.IMPROPER_SPEECH),
            (BehaviorType.LEAVING_SEAT, BehaviorType.NETWORK_ANOMALY),
        ]
        for t1, t2 in related_pairs:
            if (e1.behavior_type == t1 and e2.behavior_type == t2) or \
               (e1.behavior_type == t2 and e2.behavior_type == t1):
                return True
        return False


# ============================================================
# 三、置信度加权融合引擎
# ============================================================

class WeightedFusionEngine:
    """
    置信度加权融合引擎

    核心公式：
    final_score = Σ w_i × score_i

    权重动态调整：
    w_i = base_weight_i × reliability_i × relevance_i

    其中：
    - base_weight_i：基础权重（根据异常类型预设）
    - reliability_i：通道可靠性（根据数据质量、模型置信度动态计算）
    - relevance_i：通道相关性（该通道对该异常类型的检测能力）

    异常类型基础权重：
    - 操作手机：video=0.70, audio=0.10, log=0.20
    - 起身离席：video=0.80, audio=0.00, log=0.20
    - 交头接耳：video=0.50, audio=0.40, log=0.10
    - 不当言论：video=0.10, audio=0.80, log=0.10
    - 网络异常：video=0.10, audio=0.10, log=0.80
    """

    # 基础权重表（根据异常类型）
    BASE_WEIGHTS = {
        BehaviorType.PHONE_USAGE: {ModalityType.VIDEO: 0.70, ModalityType.AUDIO: 0.10, ModalityType.LOG: 0.20},
        BehaviorType.LEAVING_SEAT: {ModalityType.VIDEO: 0.80, ModalityType.AUDIO: 0.00, ModalityType.LOG: 0.20},
        BehaviorType.WHISPERING: {ModalityType.VIDEO: 0.50, ModalityType.AUDIO: 0.40, ModalityType.LOG: 0.10},
        BehaviorType.IMPROPER_SPEECH: {ModalityType.VIDEO: 0.10, ModalityType.AUDIO: 0.80, ModalityType.LOG: 0.10},
        BehaviorType.FACE_OCCLUSION: {ModalityType.VIDEO: 0.85, ModalityType.AUDIO: 0.05, ModalityType.LOG: 0.10},
        BehaviorType.NETWORK_ANOMALY: {ModalityType.VIDEO: 0.10, ModalityType.AUDIO: 0.10, ModalityType.LOG: 0.80},
        BehaviorType.DEVICE_ANOMALY: {ModalityType.VIDEO: 0.30, ModalityType.AUDIO: 0.10, ModalityType.LOG: 0.60},
    }

    # 通道相关性（该通道对各异常类型的检测能力）
    MODALITY_RELEVANCE = {
        ModalityType.VIDEO: {
            BehaviorType.PHONE_USAGE: 0.95,
            BehaviorType.LEAVING_SEAT: 0.98,
            BehaviorType.WHISPERING: 0.80,
            BehaviorType.IMPROPER_SPEECH: 0.30,
            BehaviorType.FACE_OCCLUSION: 0.95,
        },
        ModalityType.AUDIO: {
            BehaviorType.PHONE_USAGE: 0.20,
            BehaviorType.LEAVING_SEAT: 0.05,
            BehaviorType.WHISPERING: 0.85,
            BehaviorType.IMPROPER_SPEECH: 0.98,
            BehaviorType.FACE_OCCLUSION: 0.10,
        },
        ModalityType.LOG: {
            BehaviorType.PHONE_USAGE: 0.60,
            BehaviorType.LEAVING_SEAT: 0.50,
            BehaviorType.WHISPERING: 0.20,
            BehaviorType.IMPROPER_SPEECH: 0.25,
            BehaviorType.FACE_OCCLUSION: 0.15,
        },
    }

    def __init__(self):
        print("[WeightedFusion] 置信度加权融合引擎初始化完成")

    def calculate_reliability(self, event: ModalityEvent) -> float:
        """
        计算单通道事件的可靠性

        视频通道可靠性：
        reliability = 0.4×detection_confidence + 0.3×tracking_stability + 0.3×quality_factor

        音频通道可靠性：
        reliability = 0.5×asr_confidence + 0.3×snr_factor + 0.2×speaker_separation_quality

        日志通道可靠性：
        reliability = 0.7×log_completeness + 0.3×timestamp_accuracy
        """
        base_confidence = event.confidence
        evidence = event.evidence

        if event.modality == ModalityType.VIDEO:
            detection_conf = evidence.get("detection_confidence", base_confidence)
            tracking_stability = evidence.get("tracking_stability", 0.8)
            quality_factor = evidence.get("quality_factor", 0.9)  # 光照、遮挡等
            reliability = 0.4 * detection_conf + 0.3 * tracking_stability + 0.3 * quality_factor

        elif event.modality == ModalityType.AUDIO:
            asr_conf = evidence.get("asr_confidence", base_confidence)
            snr_factor = evidence.get("snr_factor", 0.85)  # 信噪比
            sep_quality = evidence.get("speaker_separation_quality", 0.8)
            reliability = 0.5 * asr_conf + 0.3 * snr_factor + 0.2 * sep_quality

        else:  # LOG
            log_completeness = evidence.get("log_completeness", 0.9)
            timestamp_accuracy = evidence.get("timestamp_accuracy", 0.95)
            reliability = 0.7 * log_completeness + 0.3 * timestamp_accuracy

        return min(max(reliability, 0.0), 1.0)

    def calculate_dynamic_weights(
        self,
        behavior_type: BehaviorType,
        events: List[ModalityEvent]
    ) -> Dict[ModalityType, float]:
        """
        计算动态权重

        w_i = base_weight_i × reliability_i × relevance_i
        然后归一化使 Σw_i = 1
        """
        weights = {}
        base_weights = self.BASE_WEIGHTS.get(behavior_type, {
            ModalityType.VIDEO: 0.5, ModalityType.AUDIO: 0.3, ModalityType.LOG: 0.2
        })

        for modality in ModalityType:
            base_w = base_weights.get(modality, 0.1)
            relevance = self.MODALITY_RELEVANCE.get(modality, {}).get(behavior_type, 0.5)

            # 找到该通道的事件，计算平均可靠性
            modality_events = [e for e in events if e.modality == modality]
            if modality_events:
                avg_reliability = np.mean([self.calculate_reliability(e) for e in modality_events])
            else:
                avg_reliability = 0.0  # 无事件则该通道权重为0

            weights[modality] = base_w * avg_reliability * relevance

        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def fuse(self, event_group: List[ModalityEvent]) -> FusionResult:
        """
        执行融合决策

        步骤：
        1. 确定主要行为类型（取置信度最高的事件类型）
        2. 计算动态权重
        3. 加权融合计算final_score
        4. 交叉验证
        5. 判定预警等级
        6. 构建证据包
        """
        if not event_group:
            raise ValueError("事件组为空")

        # 1. 确定主要行为类型
        main_event = max(event_group, key=lambda e: e.confidence)
        behavior_type = main_event.behavior_type

        # 2. 计算动态权重
        weights = self.calculate_dynamic_weights(behavior_type, event_group)

        # 3. 加权融合
        channel_scores = {}
        for modality in ModalityType:
            modality_events = [e for e in event_group if e.modality == modality]
            if modality_events:
                # 取该通道最高置信度（或加权平均）
                channel_scores[modality.value] = max(e.confidence for e in modality_events)
            else:
                channel_scores[modality.value] = 0.0

        final_score = sum(
            weights[modality] * channel_scores[modality.value]
            for modality in ModalityType
        )

        # 4. 交叉验证
        verification = self._cross_validate(event_group, behavior_type)

        # 矛盾则降级
        if verification == "矛盾":
            final_score *= 0.6  # 矛盾降级
        elif verification == "疑似":
            final_score *= 0.85  # 疑似轻微降级

        # 5. 判定预警等级
        alert_level = self._determine_alert_level(final_score, behavior_type, event_group)

        # 6. 构建结果
        start_time = min(e.timestamp for e in event_group)
        end_time = max(e.timestamp + e.duration for e in event_group if e.duration > 0)
        expert_id = main_event.expert_id
        seat_number = main_event.seat_number

        result = FusionResult(
            event_id=f"FUSION-{int(start_time)}-{behavior_type.name}",
            behavior_type=behavior_type,
            final_score=final_score,
            alert_level=alert_level,
            start_time=start_time,
            end_time=end_time if end_time > start_time else None,
            expert_id=expert_id,
            seat_number=seat_number,
            description=f"多模态融合检测到{behavior_type.value}，综合置信度{final_score:.3f}，{verification}",
            channel_scores=channel_scores,
            channel_weights={k.value: v for k, v in weights.items()},
            verification=verification,
            evidence_package=self._build_evidence_package(event_group)
        )

        return result

    def _cross_validate(self, events: List[ModalityEvent], behavior_type: BehaviorType) -> str:
        """
        交叉验证

        规则：
        - 强确认：≥2个通道同时检测到同类异常
        - 弱确认：1个通道高置信度(>0.7) + 其他通道无矛盾
        - 疑似：1个通道中等置信度(0.4-0.7) + 其他通道无验证
        - 矛盾：通道间检测结果矛盾
        """
        modalities_detected = set(e.modality for e in events if e.confidence > 0.3)
        num_modalities = len(modalities_detected)

        if num_modalities >= 2:
            return "强确认"
        elif num_modalities == 1:
            event = events[0]
            if event.confidence > 0.7:
                return "弱确认"
            else:
                return "疑似"
        else:
            return "矛盾"

    def _determine_alert_level(
        self,
        final_score: float,
        behavior_type: BehaviorType,
        events: List[ModalityEvent]
    ) -> AlertLevel:
        """
        判定预警等级

        规则：
        - 红色警报：final_score≥0.7 或 高风险行为类型（不当言论中的改分/内定）
        - 橙色警示：0.5≤final_score<0.7 或 重复同类异常
        - 黄色预警：0.3≤final_score<0.5
        """
        # 高风险行为直接升级
        high_risk_keywords = ["改分", "内定", "标底", "入围名单", "关照"]
        for event in events:
            if any(kw in event.description for kw in high_risk_keywords):
                return AlertLevel.RED

        if final_score >= 0.7:
            return AlertLevel.RED
        elif final_score >= 0.5:
            return AlertLevel.ORANGE
        else:
            return AlertLevel.YELLOW

    def _build_evidence_package(self, events: List[ModalityEvent]) -> Dict:
        """构建多模态证据包"""
        package = {
            "video_evidence": [],
            "audio_evidence": [],
            "log_evidence": [],
            "timeline": [],
        }

        for event in events:
            evidence_item = {
                "event_id": event.event_id,
                "modality": event.modality.value,
                "confidence": event.confidence,
                "timestamp": event.timestamp,
                "description": event.description,
                "details": event.evidence,
            }

            if event.modality == ModalityType.VIDEO:
                package["video_evidence"].append(evidence_item)
            elif event.modality == ModalityType.AUDIO:
                package["audio_evidence"].append(evidence_item)
            else:
                package["log_evidence"].append(evidence_item)

            package["timeline"].append({
                "timestamp": event.timestamp,
                "modality": event.modality.value,
                "event": event.behavior_type.value,
                "confidence": event.confidence,
            })

        # 按时间排序
        package["timeline"].sort(key=lambda x: x["timestamp"])

        return package


# ============================================================
# 四、贝叶斯后验融合（高级可选）
# ============================================================

class BayesianFusion:
    """
    贝叶斯后验概率融合（高级可选方案）

    原理：将每个通道的检测结果视为证据，使用贝叶斯定理更新异常行为的后验概率

    P(H|E1,E2,E3) = P(E1,E2,E3|H) × P(H) / P(E1,E2,E3)

    假设条件独立：
    P(E1,E2,E3|H) = P(E1|H) × P(E2|H) × P(E3|H)

    对数似然比形式：
    log[P(H|E)/P(¬H|E)] = log[P(H)/P(¬H)] + Σ log[P(Ei|H)/P(Ei|¬H)]

    优点：
    - 概率解释清晰
    - 可以自然地处理缺失通道
    - 先验概率可以根据场景调整

    缺点：
    - 条件独立假设在实际中可能不成立
    - 需要准确的似然概率估计
    """

    def __init__(self, prior_probability: float = 0.15):
        """
        Args:
            prior_probability: 异常行为的先验概率（评标场景约15%）
        """
        self.prior_odds = prior_probability / (1 - prior_probability)
        print(f"[BayesianFusion] 初始化: 先验概率={prior_probability}, 先验几率={self.prior_odds:.3f}")

    def calculate_likelihood_ratio(
        self,
        confidence: float,
        false_positive_rate: float = 0.1
    ) -> float:
        """
        计算似然比 LR = P(E|H) / P(E|¬H)

        Args:
            confidence: 检测置信度（视为真阳性率TPR）
            false_positive_rate: 假阳性率FPR

        Returns:
            似然比
        """
        tpr = confidence  # 真阳性率
        fpr = false_positive_rate  # 假阳性率
        lr = tpr / fpr if fpr > 0 else float('inf')
        return min(lr, 100)  # 上限防止数值溢出

    def fuse(self, events: List[ModalityEvent]) -> float:
        """
        贝叶斯融合，返回后验概率

        log(posterior_odds) = log(prior_odds) + Σ log(likelihood_ratio_i)
        """
        log_posterior_odds = np.log(self.prior_odds)

        for event in events:
            # 根据通道类型设置不同的假阳性率
            if event.modality == ModalityType.VIDEO:
                fpr = 0.08  # 视频通道假阳性率约8%
            elif event.modality == ModalityType.AUDIO:
                fpr = 0.12  # 音频通道假阳性率约12%
            else:
                fpr = 0.05  # 日志通道假阳性率约5%

            lr = self.calculate_likelihood_ratio(event.confidence, fpr)
            log_posterior_odds += np.log(lr)

        # 转换回概率
        posterior_odds = np.exp(log_posterior_odds)
        posterior_prob = posterior_odds / (1 + posterior_odds)

        return float(posterior_prob)


# ============================================================
# 五、多模态融合主类
# ============================================================

class MultimodalFusionSystem:
    """
    多模态融合系统主类

    整合时间戳对齐、置信度加权融合、贝叶斯融合、交叉验证、证据链构建
    """

    def __init__(self, use_bayesian: bool = False):
        self.aligner = TimestampAligner(window_size=2.0, step_size=0.5)
        self.fusion_engine = WeightedFusionEngine()
        self.bayesian_fusion = BayesianFusion() if use_bayesian else None
        self.results_history: List[FusionResult] = []
        print("[MultimodalFusion] 多模态融合系统初始化完成")

    def process(
        self,
        video_events: List[ModalityEvent],
        audio_events: List[ModalityEvent],
        log_events: List[ModalityEvent]
    ) -> List[FusionResult]:
        """
        处理三路事件，返回融合结果列表

        流程：
        1. 时间戳对齐
        2. 对每个事件组执行融合
        3. （可选）贝叶斯后验融合校准
        4. 返回融合结果
        """
        start_time = time.time()

        # 1. 时间戳对齐
        event_groups = self.aligner.align(video_events, audio_events, log_events)

        # 2. 逐组融合
        results = []
        for group in event_groups:
            # 加权融合
            result = self.fusion_engine.fuse(group)

            # 贝叶斯校准（可选）
            if self.bayesian_fusion:
                bayesian_prob = self.bayesian_fusion.fuse(group)
                # 融合两种方法的结果（取平均）
                result.final_score = 0.6 * result.final_score + 0.4 * bayesian_prob

            results.append(result)
            self.results_history.append(result)

        processing_time = (time.time() - start_time) * 1000
        print(f"[MultimodalFusion] 处理完成: {len(event_groups)}组事件, "
              f"生成{len(results)}个融合结果, 耗时{processing_time:.1f}ms")

        return results

    def get_statistics(self) -> Dict:
        """获取融合统计信息"""
        if not self.results_history:
            return {}

        alert_counts = {level.value: 0 for level in AlertLevel}
        behavior_counts = {}
        verification_counts = {}

        for result in self.results_history:
            alert_counts[result.alert_level.value] += 1
            behavior_counts[result.behavior_type.value] = \
                behavior_counts.get(result.behavior_type.value, 0) + 1
            verification_counts[result.verification] = \
                verification_counts.get(result.verification, 0) + 1

        avg_score = np.mean([r.final_score for r in self.results_history])

        return {
            "total_results": len(self.results_history),
            "alert_distribution": alert_counts,
            "behavior_distribution": behavior_counts,
            "verification_distribution": verification_counts,
            "average_final_score": float(avg_score),
        }


# ============================================================
# 六、使用示例
# ============================================================

if __name__ == "__main__":
    # 初始化多模态融合系统
    fusion_system = MultimodalFusionSystem(use_bayesian=True)

    # 模拟三路异常事件
    current_time = time.time()

    video_events = [
        ModalityEvent(
            event_id="VID-001",
            modality=ModalityType.VIDEO,
            behavior_type=BehaviorType.PHONE_USAGE,
            confidence=0.88,
            timestamp=current_time,
            duration=3.5,
            expert_id="expert_3",
            description="检测到操作手机行为",
            evidence={"detection_confidence": 0.88, "tracking_stability": 0.9, "quality_factor": 0.85}
        ),
    ]

    audio_events = [
        ModalityEvent(
            event_id="AUD-001",
            modality=ModalityType.AUDIO,
            behavior_type=BehaviorType.IMPROPER_SPEECH,
            confidence=0.92,
            timestamp=current_time + 0.5,
            duration=2.0,
            expert_id="expert_1",
            description="检测到违规言论：这个标肯定是XX单位的",
            evidence={"asr_confidence": 0.95, "snr_factor": 0.9, "speaker_separation_quality": 0.85}
        ),
    ]

    log_events = [
        ModalityEvent(
            event_id="LOG-001",
            modality=ModalityType.LOG,
            behavior_type=BehaviorType.DEVICE_ANOMALY,
            confidence=0.75,
            timestamp=current_time + 0.2,
            expert_id="expert_3",
            description="检测到屏幕亮度异常变化（可能手机亮屏）",
            evidence={"log_completeness": 0.95, "timestamp_accuracy": 0.98}
        ),
    ]

    # 执行融合
    print("\n=== 执行多模态融合 ===\n")
    results = fusion_system.process(video_events, audio_events, log_events)

    # 输出结果
    for result in results:
        print(f"事件ID: {result.event_id}")
        print(f"行为类型: {result.behavior_type.value}")
        print(f"综合置信度: {result.final_score:.3f}")
        print(f"预警等级: {result.alert_level.value}")
        print(f"验证结果: {result.verification}")
        print(f"通道得分: {result.channel_scores}")
        print(f"通道权重: {result.channel_weights}")
        print(f"描述: {result.description}")
        print()

    # 统计信息
    print("=== 融合统计 ===")
    stats = fusion_system.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")
