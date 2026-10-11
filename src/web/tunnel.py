"""Zero-dependency secure cloud tunneling for RepoLens.

Enables instant public HTTPS access to RepoLens from any phone or browser
worldwide over mobile data (4G/5G) or external networks without port forwarding.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from typing import Optional


class CloudTunnel:
    """Manages an instant secure SSH reverse proxy tunnel."""

    def __init__(self, local_port: int):
        self.local_port = local_port
        self.public_url: Optional[str] = None
        self.process: Optional[subprocess.Popen] = None
        self._output_thread: Optional[threading.Thread] = None

    def start(self, timeout: int = 15) -> Optional[str]:
        """Start the tunnel and return the public HTTPS URL, or None if failed."""
        cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-R", f"80:localhost:{self.local_port}",
            "nokey@localhost.run",
        ]

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            start_time = time.time()
            while time.time() - start_time < timeout:
                if not self.process.stdout:
                    break
                line = self.process.stdout.readline()
                if not line and self.process.poll() is not None:
                    break

                # Extract https://*.lhr.life URL
                match = re.search(r"https://[a-zA-Z0-9_\-\.]+\.lhr\.life", line)
                if match:
                    self.public_url = match.group(0)
                    # Start background reader so pipe does not block
                    self._output_thread = threading.Thread(
                        target=self._drain_output, daemon=True
                    )
                    self._output_thread.start()
                    return self.public_url

                # Secondary fallback: any https://... URL from proxy
                match_fallback = re.search(r"https://[a-zA-Z0-9_\-\.]+\.[a-zA-Z]{2,}", line)
                if match_fallback and "localhost.run" not in match_fallback.group(0):
                    self.public_url = match_fallback.group(0)
                    self._output_thread = threading.Thread(
                        target=self._drain_output, daemon=True
                    )
                    self._output_thread.start()
                    return self.public_url

        except Exception:
            return None

        return None

    def _drain_output(self) -> None:
        """Keep consuming process output in background to prevent pipe deadlock."""
        try:
            if self.process and self.process.stdout:
                for _ in self.process.stdout:
                    pass
        except Exception:
            pass

    def stop(self) -> None:
        """Terminate the tunnel process."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
