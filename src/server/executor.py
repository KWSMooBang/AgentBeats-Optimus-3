"""Executor for Optimus-3 Purple Agent

Handles A2A protocol message processing.
"""
import json
import logging
from typing import Any, Dict, List, Optional

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, Role, TextPart
from a2a.utils import new_agent_text_message

from src.agent.agent import Optimus3PurpleAgent, decode_obs, AgentState
from src.protocol.models import InitPayload, ObservationPayload, ActionPayload, AckPayload
from src.server.session_manager import SessionManager
from src.action.action_space import noop_action

logger = logging.getLogger(__name__)


class Optimus3Executor(AgentExecutor):
    """
    A2A AgentExecutor implementation
    
    Main responsibilities:
    1. Parse green agent messages
    2. Call Optimus3PurpleAgent
    3. Generate and return responses
    """
    
    def __init__(
        self,
        sessions: SessionManager,
        policy_ckpt_path: str,
        mllm_model_path: str,
        task_router_ckpt_path: str,
        device: str = "cuda",
        use_planning: bool = True,
    ):
        """
        Args:
            sessions: Session manager
            policy_ckpt_path: Optimus-3 Action Head checkpoint path
            mllm_model_path: Optimus-3 MLLM model path
            task_router_ckpt_path: Task Router checkpoint path
            device: 'cuda' or 'cpu'
            use_planning: Whether to use planning phase
        """
        self.sessions = sessions
        self.use_planning = use_planning
        
        # Create single shared agent
        logger.info("Loading Optimus-3 Agent...")
        self.agent = Optimus3PurpleAgent(
            policy_ckpt_path=policy_ckpt_path,
            mllm_model_path=mllm_model_path,
            task_router_ckpt_path=task_router_ckpt_path,
            device=device,
            use_planning=use_planning,
        )
        
        # Context-specific states
        self.agent_states: Dict[str, AgentState] = {}
        
        logger.info("Optimus3Executor initialized")
    
    async def execute(
        self,
        context: RequestContext,
        event_queue=None
    ) -> Message:
        """
        Process A2A messages
        
        Flow:
        1. Extract JSON from message
        2. Branch by type (init/obs)
        3. Call agent
        4. Generate response
        
        Args:
            context: Request context with message
            event_queue: Event queue for async operations
        
        Returns:
            Message: Response message
        """
        try:
            # Get message from context
            msg = getattr(context, "message", None)
            text = self._extract_text(msg)
            
            # Get context IDs
            context_id = (
                getattr(msg, "context_id", None) 
                or getattr(context, "context_id", None)
                or context.task_id 
                or context.session_id
                or "default"
            )
            task_id = (
                getattr(msg, "task_id", None) 
                or getattr(context, "task_id", None) 
                or context_id
            )
            
            # Create task updater
            task_updater = TaskUpdater(
                event_queue=event_queue,
                task_id=task_id,
                context_id=context_id,
            )
            
            if not text:
                return await self._error_response("No text in message", task_updater)
            
            payload = json.loads(text)
            msg_type = payload.get("type")
            
            logger.info("Processing message: type=%s, context_id=%s", msg_type, context_id)
            
            if msg_type == "init":
                return await self._handle_init(payload, context_id, task_updater)
            elif msg_type == "obs":
                return await self._handle_obs(payload, context_id, task_updater)
            else:
                return await self._error_response(f"Unknown type: {msg_type}", task_updater)
                
        except Exception as e:
            logger.exception("Execute failed: %s", e)
            return await self._error_response(str(e), task_updater)
    
    async def _handle_init(
        self,
        payload: Dict,
        context_id: str,
        task_updater: TaskUpdater
    ) -> Message:
        """Handle init message"""
        init = InitPayload(**payload)
        
        logger.info("Handling init: task=%s", init.text)
        
        # Create initial state (including planning)
        state = self.agent.initial_state(init.text)
        self.agent_states[context_id] = state
        
        # Ack response
        response = AckPayload(
            success=True,
            message=f"Task initialized: {init.text}"
        )
        
        logger.info("Init successful: plan=%s", state.plan)
        
        # Complete task
        response_msg = new_agent_text_message(response.model_dump_json())
        await task_updater.complete(message=response_msg)
        
        return response_msg
    
    async def _handle_obs(
        self,
        payload: Dict,
        context_id: str,
        task_updater: TaskUpdater
    ) -> Message:
        """Handle observation message"""
        obs_payload = ObservationPayload(**payload)
        
        # Get state
        state = self.agent_states.get(context_id)
        
        if not state:
            logger.error("No agent initialized for context_id=%s", context_id)
            return await self._error_response("No agent initialized", task_updater)
        
        logger.debug("Handling obs: step=%d", obs_payload.step)
        
        # Decode image
        image = decode_obs(obs_payload.obs)
        
        # Generate action
        action, new_state = self.agent.act(
            obs={"image": image},
            state=state
        )
        
        if action is None:
            logger.warning("Action is None, returning noop")
            action = noop_action()
        
        # Save state
        self.agent_states[context_id] = new_state
        
        # Generate response
        response = ActionPayload(**action)
        
        logger.debug("Action generated: buttons=%s, camera=%s", action["buttons"], action["camera"])
        
        # Complete task
        response_msg = new_agent_text_message(response.model_dump_json())
        await task_updater.complete(message=response_msg)
        
        return response_msg
    
    async def cancel(self, context: RequestContext, event_queue=None) -> None:
        """Cancel handler (no-op for Optimus-3)."""
        logger.warning("Cancel called (no-op)")
        return
    

    def _extract_text(self, msg: Message) -> Optional[str]:
        """Extract text from A2A Message."""
        parts = getattr(msg, "parts", None)
        if not isinstance(parts, list):
            return None

        for part in parts:
            # Case 1) Part(root=TextPart(...))
            root = getattr(part, "root", None)
            if isinstance(root, TextPart) and isinstance(root.text, str):
                return root.text

            # Case 2) part itself is TextPart
            if isinstance(part, TextPart) and isinstance(part.text, str):
                return part.text

            # Case 3) dict-like
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    return text

            # Case 4) generic attribute 'text'
            text = getattr(part, "text", None)
            if isinstance(text, str):
                return text

        return None
    
    async def _error_response(self, error_msg: str, task_updater: TaskUpdater = None) -> Message:
        """Generate error response"""
        response = AckPayload(
            success=False,
            message=error_msg
        )
        response_msg = new_agent_text_message(response.model_dump_json())
        if task_updater:
            await task_updater.complete(message=response_msg)
        return response_msg
