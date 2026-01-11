#!/usr/bin/env python3
"""
Progress Tracker

Track and display progress during generation and evaluation.
"""

from datetime import datetime
from typing import Optional


class ProgressTracker:
    """
    Track and display progress
    """

    def __init__(self, total: int, task_name: str = "Processing"):
        """
        Initialize progress tracker

        Args:
            total: Total number of items
            task_name: Name of the task
        """
        self.total = total
        self.current = 0
        self.task_name = task_name
        self.start_time = datetime.now()
        self.errors = 0
        self.skipped = 0

    def log(self, message: str, level: str = "INFO"):
        """
        Log a message with timestamp

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def update(self, current: int, message: Optional[str] = None):
        """
        Update progress

        Args:
            current: Current progress count
            message: Optional message to display
        """
        self.current = current

        # Calculate progress
        percentage = (current / self.total * 100) if self.total > 0 else 0

        # Calculate ETA
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if current > 0:
            avg_time_per_item = elapsed / current
            remaining_items = self.total - current
            eta_seconds = avg_time_per_item * remaining_items
            eta_str = self._format_seconds(eta_seconds)
        else:
            eta_str = "calculating..."

        # Build progress message
        progress_msg = f"[{current}/{self.total}] {percentage:.1f}% | ETA: {eta_str}"

        if message:
            progress_msg += f" | {message}"

        self.log(progress_msg)

    def increment_error(self):
        """Increment error count"""
        self.errors += 1

    def increment_skipped(self):
        """Increment skipped count"""
        self.skipped += 1

    def complete(self):
        """Mark as complete and display summary"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        elapsed_str = self._format_seconds(elapsed)

        summary = (
            f"\n{'='*60}\n"
            f"{self.task_name} Complete!\n"
            f"Total: {self.total} | Processed: {self.current} | "
            f"Errors: {self.errors} | Skipped: {self.skipped}\n"
            f"Time elapsed: {elapsed_str}\n"
            f"{'='*60}"
        )

        print(summary)

    def _format_seconds(self, seconds: float) -> str:
        """
        Format seconds as human-readable string

        Args:
            seconds: Number of seconds

        Returns:
            Formatted string (e.g., "1h 23m 45s")
        """
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
