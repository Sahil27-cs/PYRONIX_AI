"""
Base Agent Abstract Definition and Messaging Protocol
Author: Lead AI/ML Engineer
System: Satellite Wildfire AI System
"""

from abc import ABC, abstractmethod
import time
from typing import Dict, Any, Optional

class AgentMessage:
    """Represents an immutable message passed between agents."""
    def __init__(self, sender: str, recipient: str, message_type: str, payload: Dict[str, Any]):
        self.sender = sender
        self.recipient = recipient
        self.message_type = message_type
        self.payload = payload
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type,
            "timestamp": self.timestamp,
            "payload": self.payload
        }

class AgentResponse:
    """Represents the standardized output of an agent's execution."""
    def __init__(self, agent_name: str, status: str, data: Dict[str, Any], message: str = ""):
        self.agent_name = agent_name
        self.status = status  # "SUCCESS", "WARNING", "ERROR"
        self.data = data
        self.message = message
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "status": self.status,
            "timestamp": self.timestamp,
            "message": self.message,
            "data": self.data
        }

class BaseAgent(ABC):
    """Abstract base class for all specialized wildfire domain agents."""
    def __init__(self, name: str, role: str, description: str):
        self.name = name
        self.role = role
        self.description = description
        self.logs = []

    def log(self, message: str, level: str = "INFO"):
        entry = f"[{time.strftime('%H:%M:%S')}] [{self.name}] [{level}] {message}"
        self.logs.append(entry)
        print(entry)

    @abstractmethod
    def process(self, context: Dict[str, Any]) -> AgentResponse:
        """Processes the shared operational context and returns structured findings."""
        pass

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "total_logs": len(self.logs)
        }
