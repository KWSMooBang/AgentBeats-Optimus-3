"""Optimus-3 Purple Agent Wrapper

This module wraps the Optimus-3 model for AgentBeats Competition (Purple Agent protocol).
"""
import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import json
import numpy as np
import torch
from PIL import Image

from minecraftoptimus.model.agent.optimus3 import Optimus3Agent
from minecraftoptimus.model.steve1.agent import Optimus3ActionAgent
from src.action.action_space import noop_action

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Agent session state"""
    task_text: Optional[str] = None
    plan: Optional[list[str]] = None
    step_index: int = 0
    current_goal: Optional[str] = None
    idle_count: int = 0


class Optimus3PurpleAgent:
    """
    Optimus-3 Purple Agent Wrapper
    
    This class wraps the Optimus-3 model to conform to the Purple Agent protocol.
    
    Main responsibilities:
    1. Task planning (initial_state)
    2. Action generation (act)
    3. Convert to joint space action format
    
    Args:
        policy_ckpt_path: Optimus-3 Action Head checkpoint path
        mllm_model_path: Optimus-3 MLLM model path
        task_router_ckpt_path: Task Router checkpoint path
        device: 'cuda' or 'cpu'
        use_planning: Whether to use planning phase
    """
    
    def __init__(
        self,
        policy_ckpt_path: str,
        mllm_model_path: str,
        task_router_ckpt_path: str,
        device: str = "cuda",
        use_planning: bool = True,
    ):
        self.device = device
        self.use_planning = use_planning
        
        logger.info("Loading Optimus-3 models...")
        
        # Load MLLM + Task Router
        self.optimus_agent = Optimus3Agent(
            policy_ckpt_path=policy_ckpt_path,
            mllm_model_path=mllm_model_path,
            task_router_ckpt_path=task_router_ckpt_path,
            device=device,
        )
        
        # Action Agent is already loaded inside Optimus3Agent
        self.action_agent = self.optimus_agent.mine_policy
        
        # Current state
        self.current_task = None
        
        logger.info("Optimus-3 models loaded successfully")
    
    def reset(self):
        """Reset agent state"""
        self.current_task = None
    
    def initial_state(self, task_text: str) -> AgentState:
        """
        Create initial state (including planning)
        
        Args:
            task_text: Task received from green agent (e.g., "obtain 10 logs")
        
        Returns:
            AgentState: Initial state (including plan)
        """
        logger.info("Initializing agent state for task: %s", task_text)
        
        state = AgentState(
            task_text=task_text,
            step_index=0,
            idle_count=0,
        )
        
        if self.use_planning:
            # Task planning
            try:
                plan = self.optimus_agent.plan(task_text)
                if plan and len(plan) > 0:
                    state.plan = plan
                    state.current_goal = plan[0] if plan else task_text
                    logger.info("Generated plan:\n%s", json.dumps(plan,  indent=2))
                else:
                    state.plan = [task_text]
                    state.current_goal = task_text
                    logger.warning("Planning failed, using task directly")
            except Exception as e:
                logger.exception("Planning error, using task directly: %s", e)
                state.plan = [task_text]
                state.current_goal = task_text
        else:
            state.plan = [task_text]
            state.current_goal = task_text
        
        # Reset agent
        self.optimus_agent.reset(state.current_goal)
        self.current_task = state.current_goal
        
        return state
    
    def act(
        self,
        obs: Dict[str, np.ndarray],
        state: AgentState
    ) -> Tuple[Optional[Dict[str, Any]], AgentState]:
        """
        Generate action from observation
        
        Args:
            obs: {"image": np.ndarray[H, W, 3]} - RGB image
            state: Current agent state
        
        Returns:
            (action, new_state):
                - action: {"buttons": [int], "camera": [int]} or None
                - new_state: Updated state
        """
        try:
            # Get action from model
            action = self._get_action(obs, state.current_goal)
            
            if action is None or (action["buttons"] == [0] and action["camera"] == [60]):
                # Only track idle count and switch subtasks if using planning with multiple subtasks
                if self.use_planning and state.plan and len(state.plan) > 1:
                    state.idle_count += 1
                    logger.info("Action is idle, idle_count: %d", state.idle_count)
                    
                    # Too many idle frames, try next subtask
                    if state.idle_count > 10:
                        state.step_index += 1
                        if state.step_index < len(state.plan):
                            state.current_goal = state.plan[state.step_index]
                            state.idle_count = 0
                            logger.info("Moving to next subtask: %s", state.current_goal)
                            # Reset with new goal to get new MLLM embed
                            self.optimus_agent.reset(state.current_goal)
                            self.current_task = state.current_goal
                        else:
                            logger.info("All subtasks completed")
                else:
                    logger.debug("Action is idle (no planning or single task)")
                
                return None, state
            
            state.idle_count = 0
            return action, state
            
        except Exception as e:
            logger.exception("Failed to act: %s", e)
            return None, state
    
    def _get_action(self, obs: Dict[str, np.ndarray], task: str) -> Optional[Dict[str, Any]]:
        """
        Get action from Optimus-3 model
        
        Args:
            obs: {"image": np.ndarray[H, W, 3]}
            task: Current task/goal
        
        Returns:
            action: {"buttons": [int], "camera": [int]} in joint space format
        """
        try:
            # Get action from Optimus3Agent
            # This uses the cached MLLM embedding from reset()
            agent_action, _ = self.optimus_agent.get_action(
                input=obs,
                task=task,
                deterministic=False,
            )
            
            # Convert to purple format (joint space)
            purple_action = self._convert_to_purple_format(agent_action)
            
            return purple_action
            
        except Exception as e:
            logger.exception("Failed to get action: %s", e)
            return None
    
    def _convert_to_purple_format(self, agent_action: Dict) -> Dict[str, Any]:
        """
        Convert agent_action (joint space) to Purple Agent format
        
        Args:
            agent_action: {"buttons": np.array([idx]), "camera": np.array([idx])}
                         from MineRLConditionalAgent.get_action()
        
        Returns:
            Purple format: {"buttons": [int], "camera": [int]}
        """
        return {
            "buttons": [int(agent_action["buttons"][0])],
            "camera": [int(agent_action["camera"][0])]
        }


def decode_obs(obs_str: str) -> np.ndarray:
    """
    Decode base64 encoded image to numpy array
    
    Args:
        obs_str: base64 encoded image string
    
    Returns:
        np.ndarray: RGB image [H, W, 3]
    """
    img_bytes = base64.b64decode(obs_str)
    img = Image.open(io.BytesIO(img_bytes))
    img_array = np.array(img)
    return img_array
