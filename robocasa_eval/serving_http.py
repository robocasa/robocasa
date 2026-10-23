"""Single-port model server for RoboCasa G1 evaluation clients.

The model process owns the listening port. Simulation workers connect to it and
keep episode state isolated through explicit reset/predict/end calls. A shared
PolicyPool supplies batching and model replicas behind this HTTP boundary.
"""

from __future__ import annotations

import copy
from dataclasses import fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import math
import os
import signal
import socket
import sys
import threading
import traceback

from robocasa_eval.adapters.http import (
    FINGERPRINT_HEADER, MAX_MESSAGE_BYTES, PROTOCOL, _unique_object,
    decode_value, encode_value, validate_metadata,
)
from robocasa_eval.policy import EpisodeContext, validate_action_batch


_CONTEXT_FIELDS = {field.name for field in fields(EpisodeContext)}


def _context(value):
    if not isinstance(value, dict) or set(value) != _CONTEXT_FIELDS:
        raise ValueError("context must contain exactly the EpisodeContext fields.")
    for key in ("episode_id", "task"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"context {key} must be a nonempty string.")
    for key in ("episode", "seed", "rank", "slot"):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError(f"context {key} must be a nonnegative integer.")
    return EpisodeContext(**value)


class PolicyHTTPServer(ThreadingHTTPServer):
    """A ready pool and fixed model identity served on one TCP port."""

    daemon_threads = True
    block_on_close = False

    def service_actions(self):
        monitor = getattr(self, "quota_monitor", None)
        if monitor is not None:
            monitor.poll()

    def server_close(self):
        try:
            super().server_close()
        finally:
            try:
                if not getattr(self, "_pool_closed", True):
                    self._pool_closed = True
                    self.pool.close()
            finally:
                monitor = getattr(self, "quota_monitor", None)
                if monitor is not None:
                    monitor.close()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "RoboCasaG1"
    sys_version = ""

    def setup(self):
        self.request.settimeout(self.server.read_timeout)
        super().setup()

    def log_message(self, format, *args):
        # Never log authorization headers, observation payloads, or URLs. The
        # launch command redirects model diagnostics into the chosen workspace.
        pass

    def _log_model_error(self, error):
        # Preserve the exception chain for operators without logging request
        # headers, payloads, or frame locals (which can contain image arrays).
        diagnostic = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        authorization = self.server.authorization
        if authorization:
            # A third-party model might include its environment in an error.
            # Redact this service's credential even from such diagnostics.
            diagnostic = diagnostic.replace(authorization, "[redacted]")
            diagnostic = diagnostic.replace(authorization.removeprefix("Bearer "), "[redacted]")
        try:
            sys.stderr.write(f"Model operation {self.path} failed:\n{diagnostic}")
            sys.stderr.flush()
        except (OSError, ValueError):
            # A closed or full log destination must not replace the model
            # error response with a second transport failure.
            pass

    def _reply(self, status, payload):
        body = json.dumps(
            {"protocol": PROTOCOL, "fingerprint": self.server.metadata["fingerprint"], **payload},
            allow_nan=False, separators=(",", ":"),
        ).encode("utf-8")
        if len(body) > MAX_MESSAGE_BYTES:
            status = 500
            body = json.dumps({"protocol": PROTOCOL, "fingerprint": self.server.metadata["fingerprint"],
                               "error": "Model response exceeds the message size limit."}).encode()
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        expected = self.server.authorization
        received = self.headers.get_all("Authorization", [])
        if expected is not None and (
            len(received) != 1 or not hmac.compare_digest(received[0].encode(), expected.encode())
        ):
            self._reply(401, {"error": "Authentication required."})
            return False
        return True

    def _identity_matches(self):
        expected = self.headers.get_all(FINGERPRINT_HEADER, [])
        if expected and (len(expected) != 1 or expected[0] != self.server.metadata["fingerprint"]):
            self._reply(409, {"error": "Serving model fingerprint differs from the requested model."})
            return False
        return True

    def do_GET(self):
        try:
            if not self._authorized() or not self._identity_matches():
                return
            if self.path == "/metadata":
                self._reply(200, self.server.metadata)
            elif self.path == "/health":
                ready = bool(getattr(self.server.pool, "ready", True))
                self._reply(200 if ready else 503, {**self.server.metadata, "ready": ready})
            else:
                self._reply(404, {"error": "Unknown endpoint."})
        except (BrokenPipeError, ConnectionError, TimeoutError):
            self.close_connection = True

    def _read_request(self):
        if self.headers.get("Transfer-Encoding") is not None:
            raise ValueError("Transfer-Encoding is unsupported; send Content-Length.")
        sizes = self.headers.get_all("Content-Length", [])
        if len(sizes) != 1 or not sizes[0].isascii() or not sizes[0].isdigit():
            raise ValueError("A single nonnegative Content-Length is required.")
        size = int(sizes[0])
        if size > self.server.max_request_bytes:
            return None, 413, "Request exceeds the message size limit."
        content_types = self.headers.get_all("Content-Type", [])
        if len(content_types) != 1 or content_types[0].split(";", 1)[0].strip().lower() != "application/json":
            raise ValueError("Content-Type must be application/json.")
        raw = self.rfile.read(size)
        if len(raw) != size:
            raise ValueError("Incomplete request body.")
        try:
            payload = decode_value(json.loads(raw, object_pairs_hook=_unique_object))
        except (UnicodeError, ValueError, TypeError, RecursionError):
            raise ValueError("Malformed JSON or encoded array.") from None
        if not isinstance(payload, dict) or payload.get("protocol") != PROTOCOL:
            raise ValueError("Request has an incompatible protocol.")
        return payload, None, None

    def _parse_operation(self, payload):
        if self.path in {"/reset", "/end_episode"}:
            if set(payload) != {"protocol", "context"}:
                raise ValueError("Lifecycle request must contain protocol and context.")
            return _context(payload["context"])
        if set(payload) != {"protocol", "observations", "contexts"}:
            raise ValueError("Predict request must contain protocol, observations, and contexts.")
        observations, contexts = payload["observations"], payload["contexts"]
        if not isinstance(observations, list) or not isinstance(contexts, list) or len(observations) != len(contexts):
            raise ValueError("observations and contexts must be lists with equal lengths.")
        if not observations or any(not isinstance(obs, dict) for obs in observations):
            raise ValueError("observations must contain one or more objects.")
        contexts = [_context(value) for value in contexts]
        if len({value.episode_id for value in contexts}) != len(contexts):
            raise ValueError("A prediction batch must contain distinct episodes.")
        return observations, contexts

    def do_POST(self):
        try:
            if not self._authorized() or not self._identity_matches():
                return
            if self.path not in {"/reset", "/predict", "/end_episode"}:
                self._reply(404, {"error": "Unknown endpoint."})
                return
            try:
                payload, status, message = self._read_request()
                if status is not None:
                    self._reply(status, {"error": message})
                    return
                operation = self._parse_operation(payload)
            except (socket.timeout, TimeoutError):
                self._reply(408, {"error": "Request body read timed out."})
                return
            except ValueError as error:
                self._reply(400, {"error": str(error)})
                return
            try:
                if self.path == "/reset":
                    self.server.pool.reset(operation)
                    result = {"ok": True}
                elif self.path == "/end_episode":
                    self.server.pool.end_episode(operation)
                    result = {"ok": True}
                else:
                    observations, contexts = operation
                    actions = self.server.pool.predict_batch(observations, contexts)
                    # Model output errors are server errors, not malformed
                    # requests. Validate before sending a partial response.
                    actions = validate_action_batch(actions, len(contexts), 2**63 - 1)
                    result = {"actions": encode_value(actions)}
            except ValueError as error:
                from robocasa_eval.serving_pool import PolicyContextError
                # Only the dedicated context exception belongs to HTTP 409.
                # Array/output/model ValueErrors are inference failures.
                if isinstance(error, PolicyContextError):
                    self._reply(409, {"error": "Episode context conflicts with active model state."})
                else:
                    self._log_model_error(error)
                    self._reply(500, {"error": "Model operation failed."})
                return
            except TimeoutError as error:
                self._log_model_error(error)
                self._reply(504, {"error": "Model operation timed out."})
                return
            except Exception as error:
                self._log_model_error(error)
                self._reply(500, {"error": "Model operation failed."})
                return
            self._reply(200, result)
        except (BrokenPipeError, ConnectionError, TimeoutError):
            self.close_connection = True


def make_server(pool, metadata, *, host="127.0.0.1", port=8000, auth_token_env=None,
                max_request_bytes=MAX_MESSAGE_BYTES, read_timeout=30):
    """Bind a ready policy pool; port=0 selects an available local test port.

    The returned server owns the pool and closes it in ``server_close``. On bind
    failure, ownership remains with the caller. No thread starts until the
    caller invokes ``serve_forever``.
    """
    metadata = copy.deepcopy(validate_metadata(metadata))
    json.dumps(metadata, allow_nan=False)
    if not isinstance(host, str) or not host:
        raise ValueError("HTTP host must be a nonempty string.")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("HTTP port must be an integer from 0 to 65535.")
    if type(max_request_bytes) is not int or not 1 <= max_request_bytes <= MAX_MESSAGE_BYTES:
        raise ValueError("HTTP max_request_bytes must be a positive integer <= 64 MiB.")
    if isinstance(read_timeout, bool) or not isinstance(read_timeout, (int, float)) or not math.isfinite(read_timeout) or read_timeout <= 0:
        raise ValueError("HTTP read_timeout must be a positive finite number.")
    authorization = None
    if auth_token_env is not None:
        if not isinstance(auth_token_env, str) or not auth_token_env:
            raise ValueError("auth_token_env must name an environment variable.")
        token = os.environ.get(auth_token_env)
        if not token or any(character in token for character in "\r\n"):
            raise ValueError("HTTP authentication environment variable is missing or invalid.")
        authorization = "Bearer " + token
    server = PolicyHTTPServer((host, port), _Handler)
    server.pool = pool
    server._pool_closed = False
    server.metadata = metadata
    server.authorization = authorization
    server.max_request_bytes = max_request_bytes
    server.read_timeout = float(read_timeout)
    return server


def serve(config, metadata):
    """Load model replicas and serve clients until interrupted."""
    from robocasa_eval.evaluation import disk_preflight
    from robocasa_eval.gpu_worker import DiskQuotaMonitor
    from robocasa_eval.serving_pool import PolicyPool

    pool = PolicyPool(config)
    server = None
    try:
        disk_preflight(config)
        server = make_server(
            pool, metadata, host=config.get("host", "127.0.0.1"), port=config.get("port", 8000),
            auth_token_env=config.get("auth_token_env"),
            max_request_bytes=config.get("max_request_bytes", MAX_MESSAGE_BYTES),
            read_timeout=config.get("read_timeout_seconds", 30),
        )
        server.quota_monitor = DiskQuotaMonitor(config)
    except BaseException:
        if server is not None:
            server.server_close()
        else:
            pool.close()
        raise
    try:
        previous_handler = None
        if threading.current_thread() is threading.main_thread():
            previous_handler = signal.getsignal(signal.SIGTERM)

            def stop(signum, frame):
                # shutdown() waits for serve_forever() and would deadlock if
                # called from its own thread. Interrupt the loop instead.
                raise KeyboardInterrupt

            signal.signal(signal.SIGTERM, stop)
        try:
            print(json.dumps({"event": "ready", "host": server.server_address[0],
                              "port": server.server_address[1], **server.metadata}), flush=True)
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass
        finally:
            if previous_handler is not None:
                signal.signal(signal.SIGTERM, previous_handler)
    finally:
        server.server_close()
