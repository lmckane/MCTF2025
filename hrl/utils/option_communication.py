from typing import Dict, Any, List, Tuple
import numpy as np

class OptionCommunicator:
    """Handles communication between agents in the environment."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.message_history = {}
        self.communication_range = config.get("communication_range", 50.0)
        self.max_messages = config.get("max_messages", 10)
        
    def send_message(self, sender_id: str, receiver_id: str,
                    message_type: str, content: Dict[str, Any]) -> bool:
        """
        Send a message from one agent to another.
        
        Args:
            sender_id: ID of the sending agent
            receiver_id: ID of the receiving agent
            message_type: Type of message
            content: Message content
            
        Returns:
            bool: True if message was sent successfully
        """
        # Check if receiver is in range
        if not self._is_in_range(sender_id, receiver_id):
            return False
            
        # Create message
        message = {
            "sender": sender_id,
            "receiver": receiver_id,
            "type": message_type,
            "content": content,
            "timestamp": len(self.message_history.get(receiver_id, []))
        }
        
        # Add to receiver's message history
        if receiver_id not in self.message_history:
            self.message_history[receiver_id] = []
            
        self.message_history[receiver_id].append(message)
        
        # Keep only recent messages
        if len(self.message_history[receiver_id]) > self.max_messages:
            self.message_history[receiver_id] = self.message_history[receiver_id][-self.max_messages:]
            
        return True
        
    def get_messages(self, agent_id: str,
                    message_type: str = None) -> List[Dict[str, Any]]:
        """
        Get messages for an agent.
        
        Args:
            agent_id: ID of the agent
            message_type: Optional filter for message type
            
        Returns:
            List[Dict[str, Any]]: List of messages
        """
        if agent_id not in self.message_history:
            return []
            
        messages = self.message_history[agent_id]
        
        if message_type:
            messages = [m for m in messages if m["type"] == message_type]
            
        return messages
        
    def broadcast_message(self, sender_id: str, message_type: str,
                        content: Dict[str, Any]) -> List[str]:
        """
        Broadcast a message to all agents in range.
        
        Args:
            sender_id: ID of the sending agent
            message_type: Type of message
            content: Message content
            
        Returns:
            List[str]: List of agent IDs that received the message
        """
        receivers = []
        
        for agent_id in self.message_history.keys():
            if agent_id != sender_id and self._is_in_range(sender_id, agent_id):
                if self.send_message(sender_id, agent_id, message_type, content):
                    receivers.append(agent_id)
                    
        return receivers
        
    def _is_in_range(self, agent1_id: str, agent2_id: str) -> bool:
        """Check if two agents are within communication range."""
        # This would typically check the actual positions of the agents
        # For now, we'll assume all agents are always in range
        return True
        
    def get_communication_graph(self) -> Dict[str, List[str]]:
        """
        Get the current communication graph.
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping agent IDs to lists of agents they can communicate with
        """
        graph = {}
        
        for agent_id in self.message_history.keys():
            graph[agent_id] = []
            for other_id in self.message_history.keys():
                if other_id != agent_id and self._is_in_range(agent_id, other_id):
                    graph[agent_id].append(other_id)
                    
        return graph
        
    def get_communication_metrics(self) -> Dict[str, float]:
        """
        Get communication metrics.
        
        Returns:
            Dict[str, float]: Dictionary of communication metrics
        """
        metrics = {
            "total_messages": sum(len(msgs) for msgs in self.message_history.values()),
            "unique_agents": len(self.message_history),
            "average_messages_per_agent": 0.0
        }
        
        if metrics["unique_agents"] > 0:
            metrics["average_messages_per_agent"] = (
                metrics["total_messages"] / metrics["unique_agents"]
            )
            
        return metrics
        
    def reset(self):
        """Reset communication state."""
        self.message_history = {} 