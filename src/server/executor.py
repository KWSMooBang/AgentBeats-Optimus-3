"""Executor for Optimus-3 Purple Agent

Handles A2A protocol message processing.
"""
import json
import logging
from typing import Any, Dict, List, Optional

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, Role, TextPart, Task, TaskState, InvalidRequestError
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from src.agent.agent import Optimus3PurpleAgent, decode_obs, AgentState
from src.protocol.models import InitPayload, ObservationPayload, ActionPayload, AckPayload
from src.server.session_manager import SessionManager
from src.action.action_space import noop_action

logger = logging.getLogger(__name__)


TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected
}


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
        self.agent_states: Dict[Any, AgentState] = {}
        
        logger.info("Optimus3Executor initialized")
    
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        """
        Process A2A messages
        
        Flow:
        1. Check/create task
        2. Extract JSON from message
        3. Branch by type (init/obs)
        4. Call agent
        5. Generate response
        
        Args:
            context: Request context with message
            event_queue: Event queue for async operations
        """
        # Get message from context
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message in request"))
        
        # Check if task is already in terminal state
        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(error=InvalidRequestError(
                message=f"Task {task.id} already processed (state: {task.status.state})"
            ))
        
        # Create task if not exists
        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)
        
        # Get IDs from task
        context_id = task.context_id
        task_id = task.id
        
        # Create task updater
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task_id,
            context_id=context_id,
        )
        
        # Start working
        await updater.start_work()
        
        try:
            # Extract text from message
            text = self._extract_text(msg)
            if not text:
                await updater.failed(
                    new_agent_text_message("No text in message", context_id=context_id, task_id=task_id)
                )
                return
            
            payload = json.loads(text)
            msg_type = payload.get("type")
            
            logger.info("Processing message: context_id=%s, task_id=%s, type=%s", context_id, task_id, msg_type)
            
            if msg_type == "init":
                await self._handle_init(payload, context_id, updater)
            elif msg_type == "obs":
                await self._handle_obs(payload, context_id, updater)
            else:
                await updater.failed(
                    new_agent_text_message(f"Unknown type: {msg_type}", context_id=context_id, task_id=task_id)
                )
                
        except Exception as e:
            logger.exception("Execute failed: %s", e)
            await updater.failed(
                new_agent_text_message(f"Agent error: {str(e)}", context_id=context_id, task_id=task_id)
            )
    
    async def _handle_init(
        self,
        payload: Dict,
        context_id: str,
        task_updater: TaskUpdater
    ) -> None:
        """Handle init message"""
        init = InitPayload(**payload)
        
        logger.info("Initialize Task: %s", init.text)
        
        # Create initial state (including planning)
        state = self.agent.initial_state(init.text)
        self.agent_states[context_id] = state
        
        logger.info("Create Plan Successfully")
        
        # Ack response
        response = AckPayload(
            success=True,
            message=f"Task initialized: {init.text}"
        )
        
        # Complete task
        response_msg = new_agent_text_message(response.model_dump_json())
        await task_updater.complete(message=response_msg)
    
    async def _handle_obs(
        self,
        payload: Dict,
        context_id: str,
        task_updater: TaskUpdater
    ) -> None:
        """Handle observation message"""
        obs_payload = ObservationPayload(**payload)
        
        # Get state
        state = self.agent_states.get(context_id)
        
        if not state:
            logger.error("No agent initialized for context_id=%s", context_id)
            await task_updater.failed(new_agent_text_message("No agent initialized"))
            return
        
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
        else: 
            logger.debug("Action generated: buttons=%s, camera=%s", action["buttons"], action["camera"])
        
        # Save state
        self.agent_states[context_id] = new_state
        
        # Generate response
        response = ActionPayload(**action)
        
        # Complete task
        response_msg = new_agent_text_message(response.model_dump_json())
        await task_updater.complete(message=response_msg)
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
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
