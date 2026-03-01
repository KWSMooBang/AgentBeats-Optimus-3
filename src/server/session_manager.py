"""Session Manager for Purple Agent

Manages agent sessions and lifecycle.
"""
import time
import logging
from typing import Dict
from dataclasses import dataclass, field

from src.agent.agent import AgentState

logger = logging.getLogger(__name__)


class SessionManager:
    """Manage session lifecycle"""
    
    def __init__(self, ttl_seconds: int = 3600):
        """
        Args:
            ttl_seconds: Session expiration time (seconds)
        """
        self.ttl_seconds = ttl_seconds
        self.sessions: Dict[str, AgentState] = {}
        self.touched_times: Dict[str, float] = {}
        
        logger.info("SessionManager initialized with TTL=%d seconds", ttl_seconds)
    
    def get_or_create(self, session_id: str) -> AgentState:
        """
        Get or create session
        
        Args:
            session_id: Session ID
        
        Returns:
            AgentState: Session state
        """
        if session_id not in self.sessions:
            logger.info("Creating new session: %s", session_id)
            self.sessions[session_id] = AgentState()
            self.touched_times[session_id] = time.time()
        else:
            self.touched_times[session_id] = time.time()
        
        return self.sessions[session_id]
    
    def update(self, session_id: str, state: AgentState):
        """
        Update session state
        
        Args:
            session_id: Session ID
            state: State to update
        """
        self.sessions[session_id] = state
        self.touched_times[session_id] = time.time()
    
    def delete(self, session_id: str):
        """
        Delete session
        
        Args:
            session_id: Session ID
        """
        if session_id in self.sessions:
            logger.info("Deleting session: %s", session_id)
            del self.sessions[session_id]
            del self.touched_times[session_id]
    
    def gc(self):
        """Garbage collect expired sessions"""
        now = time.time()
        dead_sessions = [
            sid for sid, touched_time in self.touched_times.items()
            if (now - touched_time) > self.ttl_seconds
        ]
        
        for sid in dead_sessions:
            logger.info("Garbage collecting expired session: %s", sid)
            del self.sessions[sid]
            del self.touched_times[sid]
        
        if dead_sessions:
            logger.info("Garbage collected %d expired sessions", len(dead_sessions))
