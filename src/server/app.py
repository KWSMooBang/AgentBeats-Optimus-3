"""Optimus-3 Purple Agent FastAPI Server

A2A protocol server for AgentBeats Competition.
"""
import argparse
import logging
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

from src.server.executor import Optimus3Executor
from src.server.session_manager import SessionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Optimus-3 Purple Agent Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=9019, help="Server port")
    parser.add_argument("--card-url", type=str, help="Agent card URL")
    
    # Optimus-3 model paths
    parser.add_argument("--policy-ckpt", required=True, help="Policy checkpoint path")
    parser.add_argument("--mllm-model", required=True, help="MLLM model path")
    parser.add_argument("--task-router", required=True, help="Task router checkpoint path")
    
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--use-planning", action="store_true", default=False, help="Use planning")
    parser.add_argument("--state-ttl", type=int, default=3600, help="Session TTL in seconds")
    args = parser.parse_args()
    
    logger.info("Starting Optimus-3 Purple Agent Server")
    logger.info("Host: %s, Port: %d", args.host, args.port)
    logger.info("Policy: %s", args.policy_ckpt)
    logger.info("MLLM: %s", args.mllm_model)
    logger.info("Router: %s", args.task_router)
    logger.info("Device: %s", args.device)
    logger.info("Planning: %s", args.use_planning)
    
    # Agent Card
    skill = AgentSkill(
        id="optimus3-purple-policy",
        name="Optimus-3 Purple Policy",
        description="Optimus-3 agent for MCU AgentBeats Competition",
        tags=["optimus3", "minecraft", "moe", "purple"],
        examples=[],
    )
    
    agent_card = AgentCard(
        name="Optimus-3 Purple Agent",
        description="Generalist multimodal Minecraft agent with task-specific experts",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="0.1.0",
        default_input_modes=["text", "application/json"],
        default_output_modes=["text", "application/json"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )
    
    logger.info("Agent Card: %s", agent_card.name)
    
    # Session + Executor
    sessions = SessionManager(ttl_seconds=args.state_ttl)
    
    executor = Optimus3Executor(
        sessions=sessions,
        policy_ckpt_path=args.policy_ckpt,
        mllm_model_path=args.mllm_model,
        task_router_ckpt_path=args.task_router,
        device=args.device,
        use_planning=args.use_planning,
    )
    
    # Request handler
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    
    # A2A application
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    asgi_app = app.build()
    
    logger.info("Server ready to accept requests")
    
    # Run server
    uvicorn.run(asgi_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
