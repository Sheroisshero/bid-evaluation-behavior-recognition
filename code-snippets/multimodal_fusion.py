"""
视频/音频/文本多模态置信度融合 - 核心逻辑片段（脱敏版）

展示多模态异常检测的置信度融合机制，包括时空对齐、加权融合、
交叉验证、最终异常判定等核心模块。非完整生产代码。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class Modality(Enum):
    """模态类型"""
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"


class ConfidenceLevel(Enum):
    """置信度等级"""
    LOW = "low"           # 单模态触发
    MEDIUM = "medium"     # 双模态触发
    HIGH = "high"         # 三模态触发


@dataclass
class ModalityEvent:
    """单模态事件"""
    modality: Modality
    timestamp: float
    confidence: float
    event_type: str
    description: str = ""
    expert_id: Optional[int] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class FusedEvent:
    """融合后的异常事件"""
    event_id: str
    timestamp: float
    event_type: str
    final_confidence: float
    confidence_level: ConfidenceLevel
    triggered_modalities: List[Modality]
    expert_id: Optional[int] = None
    description: str = ""
    modality_events: List[ModalityEvent] = field(default_factory=list)
    requires_intervention: bool = False  # 是否需要事中介入


class ModalityFusionEngine:
    """
    多模态置信度融合引擎

    融合策略：
    1. 时空对齐：将各模态事件按时间窗口对齐（±5秒）
    2. 加权融合：视频0.5 + 音频0.3 + 文本0.2
    3. 交叉验证：单模态=低置信，双模态=中置信，三模态=高置信
    4. 最终判定：根据置信度等级决定处理方式
    """

    # 模态权重
    WEIGHTS = {
        Modality.VIDEO: 0.5,
        Modality.AUDIO: 0.3,
        Modality.TEXT: 0.2,
    }

    # 时间对齐窗口（秒）
    ALIGNMENT_WINDOW = 5.0

    # 置信度阈值
    SINGLE_MODALITY_THRESHOLD = 0.7   # 单模态触发阈值
    FINAL_EVENT_THRESHOLD = 0.5        # 最终事件输出阈值

    def __init__(self):
        # 待融合的事件缓冲区（按模态分组）
        self.event_buffer: Dict[Modality, List[ModalityEvent]] = {
            Modality.VIDEO: [],
            Modality.AUDIO: [],
            Modality.TEXT: [],
        }
        self.event_count = 0

    def add_event(self, event: ModalityEvent):
        """
        添加单模态事件到缓冲区
        """
        self.event_buffer[event.modality].append(event)
        logger.debug(f"Event added: {event.modality.value} - {event.event_type} "
                     f"(conf={event.confidence:.2f})")

    def fuse(self, current_time: float) -> List[FusedEvent]:
        """
        执行多模态融合，返回融合后的异常事件列表

        current_time: 当前时间戳（用于清理过期事件）
        """
        fused_events = []

        # 1. 清理过期事件（超过对齐窗口2倍的事件）
        self._cleanup_old_events(current_time)

        # 2. 以视频事件为锚点进行时空对齐（视频是最主要的模态）
        for video_event in self.event_buffer[Modality.VIDEO]:
            if video_event.confidence < self.SINGLE_MODALITY_THRESHOLD:
                continue

            # 查找对齐窗口内的其他模态事件
            aligned_events = self._find_aligned_events(video_event)

            # 3. 加权融合
            fused_event = self._weighted_fusion(video_event, aligned_events)

            if fused_event and fused_event.final_confidence >= self.FINAL_EVENT_THRESHOLD:
                fused_events.append(fused_event)

                # 标记已融合的事件（避免重复融合）
                self._mark_as_fused(video_event, aligned_events)

        # 4. 处理音频单独触发的事件（敏感词是强信号）
        for audio_event in self.event_buffer[Modality.AUDIO]:
            if audio_event.confidence >= 0.8:  # 高置信敏感词
                # 检查是否已经被融合
                if not audio_event.metadata.get("fused", False):
                    fused_event = self._create_audio_only_event(audio_event)
                    fused_events.append(fused_event)
                    audio_event.metadata["fused"] = True

        return fused_events

    def _find_aligned_events(self, anchor_event: ModalityEvent) -> List[ModalityEvent]:
        """
        查找与锚点事件在时间窗口内对齐的其他模态事件
        """
        aligned = []
        t = anchor_event.timestamp

        for modality in [Modality.AUDIO, Modality.TEXT]:
            for event in self.event_buffer[modality]:
                if event.metadata.get("fused", False):
                    continue
                # 时间差在对齐窗口内
                if abs(event.timestamp - t) <= self.ALIGNMENT_WINDOW:
                    # 专家ID匹配（如果有的话）
                    if (anchor_event.expert_id is not None
                            and event.expert_id is not None
                            and anchor_event.expert_id != event.expert_id):
                        continue
                    aligned.append(event)

        return aligned

    def _weighted_fusion(
        self,
        video_event: ModalityEvent,
        aligned_events: List[ModalityEvent],
    ) -> Optional[FusedEvent]:
        """
        加权置信度融合
        final_confidence = sum(weight * confidence) for triggered modalities
        """
        all_events = [video_event] + aligned_events
        triggered_modalities = set(e.modality for e in all_events)

        # 计算加权置信度
        weighted_sum = 0.0
        total_weight = 0.0
        for event in all_events:
            weight = self.WEIGHTS[event.modality]
            weighted_sum += weight * event.confidence
            total_weight += weight

        if total_weight == 0:
            return None

        final_confidence = weighted_sum / total_weight

        # 确定置信度等级
        num_modalities = len(triggered_modalities)
        if num_modalities >= 3:
            conf_level = ConfidenceLevel.HIGH
        elif num_modalities == 2:
            conf_level = ConfidenceLevel.MEDIUM
        else:
            conf_level = ConfidenceLevel.LOW

        # 交叉验证提升：多模态同时触发时，置信度额外提升
        if num_modalities >= 2:
            final_confidence = min(final_confidence + 0.05 * (num_modalities - 1), 1.0)

        # 确定事件类型（以置信度最高的模态为准）
        primary_event = max(all_events, key=lambda e: e.confidence)

        # 是否需要事中介入
        requires_intervention = (
            conf_level == ConfidenceLevel.HIGH
            or (conf_level == ConfidenceLevel.MEDIUM and final_confidence >= 0.8)
            or primary_event.event_type in ["敏感词", "传递纸条"]
        )

        self.event_count += 1
        return FusedEvent(
            event_id=f"FUSED-{int(time.time())}-{self.event_count:04d}",
            timestamp=primary_event.timestamp,
            event_type=primary_event.event_type,
            final_confidence=round(final_confidence, 3),
            confidence_level=conf_level,
            triggered_modalities=list(triggered_modalities),
            expert_id=primary_event.expert_id,
            description=f"[{conf_level.value}] 检测到{primary_event.event_type}，"
                        f"置信度{final_confidence:.2f}，"
                        f"触发模态: {[m.value for m in triggered_modalities]}",
            modality_events=all_events,
            requires_intervention=requires_intervention,
        )

    def _create_audio_only_event(self, audio_event: ModalityEvent) -> FusedEvent:
        """
        创建音频单独触发的事件（高置信敏感词）
        """
        self.event_count += 1
        return FusedEvent(
            event_id=f"FUSED-{int(time.time())}-{self.event_count:04d}",
            timestamp=audio_event.timestamp,
            event_type=audio_event.event_type,
            final_confidence=audio_event.confidence,
            confidence_level=ConfidenceLevel.MEDIUM,  # 音频单独触发但高置信，定为中等级
            triggered_modalities=[Modality.AUDIO],
            expert_id=audio_event.expert_id,
            description=f"[medium] 音频检测到{audio_event.event_type}，"
                        f"置信度{audio_event.confidence:.2f}（敏感词强信号）",
            modality_events=[audio_event],
            requires_intervention=True,  # 敏感词需要事中介入
        )

    def _mark_as_fused(self, video_event: ModalityEvent, aligned_events: List[ModalityEvent]):
        """标记事件已融合"""
        video_event.metadata["fused"] = True
        for event in aligned_events:
            event.metadata["fused"] = True

    def _cleanup_old_events(self, current_time: float):
        """清理过期事件"""
        max_age = self.ALIGNMENT_WINDOW * 3
        for modality in self.event_buffer:
            self.event_buffer[modality] = [
                e for e in self.event_buffer[modality]
                if current_time - e.timestamp <= max_age
            ]


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    engine = ModalityFusionEngine()

    # 模拟：视频检测到掏手机（置信度0.75）
    engine.add_event(ModalityEvent(
        modality=Modality.VIDEO,
        timestamp=100.0,
        confidence=0.75,
        event_type="掏手机",
        expert_id=3,
    ))

    # 模拟：2秒后音频检测到敏感词（置信度0.85）
    engine.add_event(ModalityEvent(
        modality=Modality.AUDIO,
        timestamp=102.0,
        confidence=0.85,
        event_type="敏感词",
        expert_id=3,
    ))

    # 执行融合
    events = engine.fuse(current_time=105.0)

    for event in events:
        print(f"事件ID: {event.event_id}")
        print(f"  类型: {event.event_type}")
        print(f"  最终置信度: {event.final_confidence}")
        print(f"  置信等级: {event.confidence_level.value}")
        print(f"  触发模态: {[m.value for m in event.triggered_modalities]}")
        print(f"  事中介入: {event.requires_intervention}")
        print(f"  描述: {event.description}")
