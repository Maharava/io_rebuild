"""
Main entry point for Io wake word CLI
"""
import argparse
import logging
import sys
from typing import List, Optional

from io_wake_word.cli.commands import (add_analyze_parser, add_detect_parser,
                                     add_init_parser, add_recover_parser,
                                     add_record_parser, add_train_parser,
                                     setup_logging)
from io_wake_word.utils.paths import ensure_app_directories

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments
    
    Args:
        args: Command-line arguments
        
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        prog="io-wake-word",
        description="Io Wake Word Detection Engine",
    )
    
    parser.add_argument(
        "--verbose",
        help="Enable verbose logging",
        action="store_true",
    )
    
    # Add subparsers
    subparsers = parser.add_subparsers(
        dest="command",
        help="Command to run",
        required=True,
    )
    
    # Add subcommand parsers
    add_init_parser(subparsers)
    add_train_parser(subparsers)
    add_detect_parser(subparsers)
    add_analyze_parser(subparsers)
    add_recover_parser(subparsers)
    add_record_parser(subparsers)
    
    return parser.parse_args(args)

def main() -> int:
    """Main entry point
    
    Returns:
        Exit code
    """
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Ensure app directories exist
    ensure_app_directories()
    
    # Run command
    try:
        return args.func(args)
    except Exception as e:
        logging.error(f"Error running command: {e}", exc_info=args.verbose)
        return 1

if __name__ == "__main__":
    sys.exit(main())