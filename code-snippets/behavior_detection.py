"""
目标检测+时序行为识别双路架构 - 核心逻辑片段（脱敏版）

展示CV行为识别系统的双路架构设计，包括目标检测、人体跟踪、
时序行为识别、异常事件输出等核心模块。非完整生产代码。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BehaviorType(Enum):
    """异常行为类型"""
    PHONE_USE = "掏手机"
    LEAVE_SEAT = "离座"
    WHISPERING = "交头接耳"
    PASSING_NOTE = "传递纸条"
    ABNORMAL_GESTURE = "异常手势"
    NORMAL = "正常"


@dataclass
class Detection:
    """检测结果"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_name: str
    track_id: Optional[int] = None


@dataclass
class BehaviorEvent:
    """行为事件"""
    event_id: str
    timestamp: float
    expert_id: int
    behavior_type: BehaviorType
    confidence: float
    modalities: List[str] = field(default_factory=list)
    key_frames: List[str] = field(default_factory=list)
    video_clip_ref: str = ""
    description: str = ""


class ObjectDetector:
    """
    目标检测器（YOLOv8）
    检测人体、手机、纸条等关键物体
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        # self.model = YOLO(model_path)  # 脱敏：实际加载模型

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        单帧目标检测
        返回：人体、手机、纸条等检测结果
        """
        # results = self.model(frame, conf=self.conf_threshold)
        # detections = self._parse_results(results)
        # return detections
        return []  # 脱敏展示

    def _parse_results(self, results) -> List[Detection]:
        """解析YOLO输出为Detection格式"""
        detections = []
        # for result in results:
        #     for box in result.boxes:
        #         cls_name = result.names[int(box.cls)]
        #         detections.append(Detection(
        #             bbox=box.xyxy[0].tolist(),
        #             confidence=float(box.conf),
        #             class_name=cls_name,
        #         ))
        return detections


class PersonTracker:
    """
    人体跟踪器（DeepSORT）
    为每个检测到的人体分配稳定的跟踪ID
    """

    def __init__(self):
        # self.tracker = DeepSORT(...)  # 脱敏
        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Detection]:
        """
        更新跟踪状态，为人体检测分配track_id
        """
        # tracked = self.tracker.update(detections, frame)
        # return tracked
        return detections  # 脱敏展示


class BehaviorRecognizer:
    """
    时序行为识别器（SlowFast）
    基于视频片段识别具体行为类别
    """

    def __init__(self, model_path: str, num_classes: int = 6):
        self.model_path = model_path
        self.num_classes = num_classes
        # self.model = SlowFast.load(model_path)  # 脱敏

    def recognize(self, video_clip: np.ndarray, person_bbox: Optional[Tuple] = None) -> Tuple[BehaviorType, float]:
        """
        识别视频片段中的行为
        video_clip: 连续帧（如32帧）
        person_bbox: 目标人体的bbox（用于裁剪关注区域）
        返回：(行为类型, 置信度)
        """
        # 1. 如果有人体bbox，裁剪关注区域
        # if person_bbox:
        #     clip = self._crop_person(video_clip, person_bbox)
        # else:
        #     clip = video_clip

        # 2. 模型推理
        # logits = self.model(clip)
        # probs = softmax(logits)
        # pred_class = argmax(probs)
        # confidence = probs[pred_class]

        # 3. 映射为BehaviorType
        # behavior = self._class_to_behavior(pred_class)

        # return behavior, confidence
        return BehaviorType.NORMAL, 0.0  # 脱敏展示

    def _crop_person(self, clip: np.ndarray, bbox: Tuple) -> np.ndarray:
        """裁剪视频片段中的人体区域"""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # 扩展bbox以包含上下文
        h, w = clip.shape[1:3]
        x1 = max(0, x1 - 20)
        y1 = max(0, y1 - 20)
        x2 = min(w, x2 + 20)
        y2 = min(h, y2 + 20)
        return clip[:, y1:y2, x1:x2, :]


class DualPathBehaviorDetectionSystem:
    """
    目标检测+时序行为识别双路架构系统

    路1（目标检测）：YOLOv8实时检测人体、手机、纸条，提供空间位置
    路2（时序行为识别）：SlowFast识别具体行为类别
    融合：目标检测的跟踪ID和物体位置与行为识别结果关联
    """

    def __init__(
        self,
        yolo_model_path: str,
        slowfast_model_path: str,
        frame_rate: int = 5,          # 抽帧率（fps）
        clip_length: int = 32,         # 行为识别视频片段长度（帧）
        behavior_conf_threshold: float = 0.6,
    ):
        self.detector = ObjectDetector(yolo_model_path)
        self.tracker = PersonTracker()
        self.recognizer = BehaviorRecognizer(slowfast_model_path)
        self.frame_rate = frame_rate
        self.clip_length = clip_length
        self.behavior_conf_threshold = behavior_conf_threshold

        # 帧缓冲区（用于行为识别）
        self.frame_buffer: List[np.ndarray] = []
        # 每个人体的最近检测结果
        self.person_detections: Dict[int, List[Detection]] = {}
        # 事件计数器
        self.event_count = 0

    def process_frame(self, frame: np.ndarray, timestamp: float) -> List[BehaviorEvent]:
        """
        处理单帧，返回当前帧检测到的异常行为事件
        """
        events = []

        # ===== 路1：目标检测 =====
        detections = self.detector.detect(frame)

        # 人体跟踪
        person_dets = [d for d in detections if d.class_name == "person"]
        tracked_persons = self.tracker.update(person_dets, frame)

        # 物体检测（手机、纸条等）
        object_dets = [d for d in detections if d.class_name in ["cell phone", "paper"]]

        # 更新帧缓冲区
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.clip_length:
            self.frame_buffer.pop(0)

        # ===== 路2：时序行为识别（每clip_length帧执行一次）=====
        if len(self.frame_buffer) >= self.clip_length and len(self.frame_buffer) % self.clip_length == 0:
            video_clip = np.stack(self.frame_buffer[-self.clip_length:])

            for person in tracked_persons:
                if person.track_id is None:
                    continue

                # 行为识别
                behavior, confidence = self.recognizer.recognize(
                    video_clip, person.bbox
                )

                # 物体检测辅助判断（如检测到手机+行为=掏手机）
                behavior = self._fuse_with_object_detection(
                    behavior, confidence, person, object_dets
                )

                # 异常行为判定
                if behavior != BehaviorType.NORMAL and confidence >= self.behavior_conf_threshold:
                    event = self._create_event(
                        timestamp, person.track_id, behavior, confidence, frame
                    )
                    events.append(event)

        return events

    def _fuse_with_object_detection(
        self,
        behavior: BehaviorType,
        confidence: float,
        person: Detection,
        objects: List[Detection],
    ) -> Tuple[BehaviorType, float]:
        """
        目标检测结果与行为识别融合
        例如：检测到手机在人手附近 + 行为识别为低头 = 掏手机
        """
        for obj in objects:
            # 计算物体与人体的距离
            if self._is_close(person.bbox, obj.bbox, threshold=50):
                if obj.class_name == "cell phone" and behavior == BehaviorType.NORMAL:
                    # 检测到手机但行为识别为正常，提升为掏手机（低置信）
                    return BehaviorType.PHONE_USE, max(confidence, 0.55)
                elif obj.class_name == "cell phone" and behavior == BehaviorType.PHONE_USE:
                    # 双路一致，提升置信度
                    return behavior, min(confidence + 0.15, 1.0)
                elif obj.class_name == "paper" and behavior == BehaviorType.PASSING_NOTE:
                    return behavior, min(confidence + 0.15, 1.0)

        return behavior, confidence

    def _is_close(self, bbox1: Tuple, bbox2: Tuple, threshold: float = 50) -> bool:
        """判断两个bbox是否接近（中心距离）"""
        c1x = (bbox1[0] + bbox1[2]) / 2
        c1y = (bbox1[1] + bbox1[3]) / 2
        c2x = (bbox2[0] + bbox2[2]) / 2
        c2y = (bbox2[1] + bbox2[3]) / 2
        distance = ((c1x - c2x) ** 2 + (c1y - c2y) ** 2) ** 0.5
        return distance < threshold

    def _create_event(
        self,
        timestamp: float,
        expert_id: int,
        behavior: BehaviorType,
        confidence: float,
        frame: np.ndarray,
    ) -> BehaviorEvent:
        """创建行为事件"""
        self.event_count += 1
        event = BehaviorEvent(
            event_id=f"BEH-{timestamp:.0f}-{self.event_count:04d}",
            timestamp=timestamp,
            expert_id=expert_id,
            behavior_type=behavior,
            confidence=confidence,
            modalities=["video"],
            description=f"专家席位{expert_id}检测到{behavior.value}行为，置信度{confidence:.2f}",
        )
        logger.info(f"Behavior event detected: {event.description}")
        return event


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    # 初始化双路检测系统
    system = DualPathBehaviorDetectionSystem(
        yolo_model_path="models/yolov8n.pt",
        slowfast_model_path="models/slowfast_r50.pth",
        frame_rate=5,
        clip_length=32,
        behavior_conf_threshold=0.6,
    )

    # 模拟视频流处理
    # cap = cv2.VideoCapture("rtsp://camera-stream")
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    #     events = system.process_frame(frame, timestamp)
    #     for event in events:
    #         print(f"[{event.timestamp:.1f}s] {event.description}")

    print("Dual-path behavior detection system initialized")
    print("  - Object detection: YOLOv8")
    print("  - Behavior recognition: SlowFast")
    print("  - Tracking: DeepSORT")
    print("  - Frame rate: 5fps")
    print("  - Clip length: 32 frames")
