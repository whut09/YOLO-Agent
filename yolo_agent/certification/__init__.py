"""Opt-in local GPU and full-COCO certification support."""

from yolo_agent.certification.schemas import CertificationReport
from yolo_agent.certification.runner import RealGpuAcceptanceSuite

__all__ = ["CertificationReport", "RealGpuAcceptanceSuite"]
