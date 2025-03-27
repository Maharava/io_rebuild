"""
Actions Handler - Executes actions when wake words are detected
"""
import logging
import platform
import queue
import subprocess
import threading
import time
import os
from typing import Any, Callable, Dict, Optional, Union

logger = logging.getLogger("io_wake_word.utils")

class ActionStrategy:
    """Base class for action strategies"""
    
    def execute(self, params: Dict[str, Any]) -> bool:
        """Execute the action with given parameters
        
        Args:
            params: Action parameters
            
        Returns:
            True if executed successfully, False otherwise
        """
        raise NotImplementedError("Subclasses must implement execute method")


class NotificationAction(ActionStrategy):
    """Show a desktop notification"""
    
    def execute(self, params: Dict[str, Any]) -> bool:
        """Show a desktop notification
        
        Args:
            params: Action parameters, should include 'message'
            
        Returns:
            True if notification was shown, False otherwise
        """
        message = params.get("message", "Wake word detected!")
        system = platform.system()
        
        try:
            if system == "Windows":
                # Try PowerShell notification
                ps_script = (
                    f'[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, '
                    f'ContentType = WindowsRuntime] > $null\n'
                    f'$template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent('
                    f'[Windows.UI.Notifications.ToastTemplateType]::ToastText01)\n'
                    f'$toastXml = [xml] $template\n'
                    f'$toastXml.GetElementsByTagName("text")[0].AppendChild($toastXml.CreateTextNode("'
                    f'{message}")) > $null\n'
                    f'$toast = [Windows.UI.Notifications.ToastNotification]::new($toastXml)\n'
                    f'$notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('
                    f'"Io Wake Word")\n'
                    f'$notifier.Show($toast);'
                )
                subprocess.run(['powershell', '-Command', ps_script], 
                               capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                return True
            
            elif system == "Darwin":  # macOS
                cmd = f'osascript -e \'display notification "{message}" with title "Io Wake Word"\''
                subprocess.run(cmd, shell=True, timeout=3)
                return True
            
            elif system == "Linux":
                # Try notify-send first
                subprocess.run(['notify-send', 'Io Wake Word', message], timeout=3)
                return True
                
            else:
                logger.warning(f"Notifications not implemented for {system}")
                # Always return True to avoid retries
                return True
                
        except Exception as e:
            logger.error(f"Error showing notification: {e}")
            return False


class CommandAction(ActionStrategy):
    """Run a shell command"""
    
    def execute(self, params: Dict[str, Any]) -> bool:
        """Run a shell command
        
        Args:
            params: Action parameters, should include 'command'
            
        Returns:
            True if command was executed, False otherwise
        """
        command = params.get("command", "")
        if not command:
            logger.warning("Empty command provided")
            return False
            
        try:
            # Use subprocess.Popen with proper arguments
            if platform.system() == "Windows":
                # Use CREATE_NO_WINDOW flag on Windows
                subprocess.Popen(command, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                # Use shell on Unix systems
                subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            return True
                
        except Exception as e:
            logger.error(f"Error running command: {e}")
            return False


class CustomScriptAction(ActionStrategy):
    """Run a custom script"""
    
    def execute(self, params: Dict[str, Any]) -> bool:
        """Run a custom script
        
        Args:
            params: Action parameters, should include 'script_path'
            
        Returns:
            True if script was executed, False otherwise
        """
        script_path = params.get("script_path", "")
        if not script_path or not os.path.exists(script_path):
            logger.warning(f"Script not found: {script_path}")
            return False
            
        try:
            if platform.system() == "Windows":
                subprocess.Popen(script_path, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                subprocess.Popen(script_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            return True
                
        except Exception as e:
            logger.error(f"Error running script: {e}")
            return False


class ActionHandler:
    """Handle actions when wake word is detected"""
    
    def __init__(self, action_config: Dict[str, Any], debounce_time: float = 3.0):
        """Initialize action handler
        
        Args:
            action_config: Action configuration dictionary
            debounce_time: Minimum time between consecutive actions
        """
        self.action_config = action_config
        self.debounce_time = debounce_time
        self.last_trigger_time = 0
        self.lock = threading.Lock()
        
        # Set up action strategies
        self.strategies = {
            "notification": NotificationAction(),
            "command": CommandAction(),
            "custom_script": CustomScriptAction(),
        }
        
        # Queue for actions to be executed
        self.action_queue = queue.Queue()
        
        # Start the action worker thread
        self.running = True
        self.worker_thread = threading.Thread(target=self._action_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def trigger(self) -> bool:
        """Execute configured action if debounce period has passed
        
        Returns:
            True if action was triggered, False otherwise
        """
        current_time = time.time()
        
        with self.lock:
            # Check if debounce period has passed
            if current_time - self.last_trigger_time < self.debounce_time:
                logger.debug("Ignoring trigger due to debounce period")
                return False
            
            # Update last trigger time
            self.last_trigger_time = current_time
        
        # Queue the action for execution
        action_config = self.action_config.copy()  # Copy to avoid race conditions
        self.action_queue.put(action_config)
        
        return True
    
    def _action_worker(self) -> None:
        """Worker thread to execute actions from the queue"""
        while self.running:
            try:
                # Get action with timeout to allow thread to exit
                action_config = self.action_queue.get(timeout=1.0)
                
                # Execute the action
                self._execute_action(action_config)
                
                # Mark task as done
                self.action_queue.task_done()
            except queue.Empty:
                # Timeout waiting for action, just continue
                pass
            except Exception as e:
                logger.error(f"Error in action worker: {e}")
                # Don't crash the thread on error
    
    def _execute_action(self, action_config: Dict[str, Any]) -> None:
        """Execute the configured action using strategy pattern
        
        Args:
            action_config: Action configuration dictionary
        """
        try:
            action_type = action_config.get("type", "notification")
            params = action_config.get("params", {})
            
            logger.info(f"Executing action: {action_type}")
            
            # Get the appropriate strategy
            strategy = self.strategies.get(action_type)
            
            if strategy:
                strategy.execute(params)
            else:
                logger.warning(f"Unknown action type: {action_type}")
                
        except Exception as e:
            logger.error(f"Error executing action: {e}")
    
    def update_config(self, action_config: Dict[str, Any], debounce_time: Optional[float] = None) -> None:
        """Update action configuration thread-safely
        
        Args:
            action_config: New action configuration
            debounce_time: New debounce time (if None, keep current)
        """
        with self.lock:
            self.action_config = action_config
            
            if debounce_time is not None:
                self.debounce_time = debounce_time
                
            logger.info(f"Updated action configuration: {action_config}")
    
    def shutdown(self) -> None:
        """Shutdown the action worker cleanly"""
        self.running = False
        
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)