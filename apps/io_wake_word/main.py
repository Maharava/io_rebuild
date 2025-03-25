"""
Standalone application for Io wake word detection
"""
import logging
import sys

from io_wake_word.utils.paths import ensure_app_directories

# Application entry point
def main():
    """Main application entry point"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Ensure app directories exist
    ensure_app_directories()
    
    # Import UI modules here to avoid loading UI dependencies
    # when they're not needed (e.g., in command-line usage)
    try:
        from io_app.ui.app import IoApp
    except ImportError:
        logging.error("UI dependencies not installed. Run: pip install io-wake-word[ui]")
        return 1
    
    # Create and run the application
    app = IoApp()
    app.run()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())