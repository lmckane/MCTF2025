from typing import Dict, Any, List, Tuple
import numpy as np
from collections import defaultdict
import logging
import json
from datetime import datetime

class OptionLogger:
    """Handles logging of option execution and performance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_level = config.get("log_level", "INFO")
        self.log_file = config.get("log_file", "option_logs.json")
        self.execution_logs = defaultdict(list)
        self.performance_logs = defaultdict(list)
        self.error_logs = defaultdict(list)
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("OptionLogger")
        
    def log_execution(self, option_name: str,
                     state: Dict[str, Any],
                     action: Dict[str, Any],
                     result: Dict[str, Any]):
        """
        Log option execution.
        
        Args:
            option_name: Name of the option
            state: Current state
            action: Action taken
            result: Execution result
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "state": state,
            "action": action,
            "result": result
        }
        
        self.execution_logs[option_name].append(log_entry)
        self.logger.info(f"Logged execution for option {option_name}")
        
    def log_performance(self, option_name: str,
                       metrics: Dict[str, float]):
        """
        Log performance metrics for an option.
        
        Args:
            option_name: Name of the option
            metrics: Dictionary of performance metrics
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        self.performance_logs[option_name].append(log_entry)
        self.logger.info(f"Logged performance for option {option_name}")
        
    def log_error(self, option_name: str,
                 error: Dict[str, Any]):
        """
        Log error for an option.
        
        Args:
            option_name: Name of the option
            error: Dictionary containing error information
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "error": error
        }
        
        self.error_logs[option_name].append(log_entry)
        self.logger.error(f"Logged error for option {option_name}: {error}")
        
    def get_execution_logs(self, option_name: str,
                         start_time: str = None,
                         end_time: str = None) -> List[Dict[str, Any]]:
        """
        Get execution logs for an option.
        
        Args:
            option_name: Name of the option
            start_time: Start time for filtering (ISO format)
            end_time: End time for filtering (ISO format)
            
        Returns:
            List[Dict[str, Any]]: List of execution logs
        """
        logs = self.execution_logs[option_name]
        
        if start_time:
            logs = [log for log in logs if log["timestamp"] >= start_time]
        if end_time:
            logs = [log for log in logs if log["timestamp"] <= end_time]
            
        return logs
        
    def get_performance_logs(self, option_name: str,
                           start_time: str = None,
                           end_time: str = None) -> List[Dict[str, Any]]:
        """
        Get performance logs for an option.
        
        Args:
            option_name: Name of the option
            start_time: Start time for filtering (ISO format)
            end_time: End time for filtering (ISO format)
            
        Returns:
            List[Dict[str, Any]]: List of performance logs
        """
        logs = self.performance_logs[option_name]
        
        if start_time:
            logs = [log for log in logs if log["timestamp"] >= start_time]
        if end_time:
            logs = [log for log in logs if log["timestamp"] <= end_time]
            
        return logs
        
    def get_error_logs(self, option_name: str,
                      start_time: str = None,
                      end_time: str = None) -> List[Dict[str, Any]]:
        """
        Get error logs for an option.
        
        Args:
            option_name: Name of the option
            start_time: Start time for filtering (ISO format)
            end_time: End time for filtering (ISO format)
            
        Returns:
            List[Dict[str, Any]]: List of error logs
        """
        logs = self.error_logs[option_name]
        
        if start_time:
            logs = [log for log in logs if log["timestamp"] >= start_time]
        if end_time:
            logs = [log for log in logs if log["timestamp"] <= end_time]
            
        return logs
        
    def save_logs(self, file_path: str = None):
        """
        Save all logs to a file.
        
        Args:
            file_path: Path to save logs (defaults to config value)
        """
        if file_path is None:
            file_path = self.log_file
            
        logs = {
            "execution_logs": self.execution_logs,
            "performance_logs": self.performance_logs,
            "error_logs": self.error_logs
        }
        
        with open(file_path, 'w') as f:
            json.dump(logs, f, indent=2)
            
        self.logger.info(f"Saved logs to {file_path}")
        
    def load_logs(self, file_path: str = None):
        """
        Load logs from a file.
        
        Args:
            file_path: Path to load logs from (defaults to config value)
        """
        if file_path is None:
            file_path = self.log_file
            
        try:
            with open(file_path, 'r') as f:
                logs = json.load(f)
                
            self.execution_logs = defaultdict(list, logs.get("execution_logs", {}))
            self.performance_logs = defaultdict(list, logs.get("performance_logs", {}))
            self.error_logs = defaultdict(list, logs.get("error_logs", {}))
            
            self.logger.info(f"Loaded logs from {file_path}")
        except FileNotFoundError:
            self.logger.warning(f"Log file {file_path} not found")
            
    def get_log_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the logs.
        
        Returns:
            Dict[str, Any]: Dictionary of log statistics
        """
        stats = {
            "num_options": len(self.execution_logs),
            "total_executions": sum(
                len(logs) for logs in self.execution_logs.values()
            ),
            "total_performance_entries": sum(
                len(logs) for logs in self.performance_logs.values()
            ),
            "total_errors": sum(
                len(logs) for logs in self.error_logs.values()
            )
        }
        
        return stats
        
    def reset(self):
        """Reset all logs."""
        self.execution_logs.clear()
        self.performance_logs.clear()
        self.error_logs.clear()
        self.logger.info("Reset all logs") 