"""Protocol models for Purple Agent communication"""
from pydantic import BaseModel
from typing import Literal, Optional


class InitPayload(BaseModel):
    """Initial task received from green agent"""
    type: Literal["init"] = "init"
    text: str


class ObservationPayload(BaseModel):
    """Observation at each step"""
    type: Literal["obs"] = "obs"
    step: int
    obs: str  # base64 encoded image


class ActionPayload(BaseModel):
    """Action returned by purple agent"""
    type: Literal["action"] = "action"
    action_type: Literal["agent"] = "agent"
    buttons: list[int]  # [button_combination_idx]
    camera: list[int]   # [camera_idx]


class AckPayload(BaseModel):
    """Acknowledgment response"""
    type: Literal["ack"] = "ack"
    success: bool
    message: str = ""
