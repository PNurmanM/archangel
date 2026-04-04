#!/usr/bin/env python3
"""Small binary protocol for frame streaming and result messages."""

from __future__ import annotations

import json
import socket
import struct


HEADER_STRUCT = struct.Struct("!II")


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("Socket closed while receiving data.")
        chunks.extend(chunk)
    return bytes(chunks)


def send_message(sock: socket.socket, message: dict, payload: bytes = b"") -> None:
    message_bytes = json.dumps(message, separators=(",", ":")).encode("utf-8")
    sock.sendall(HEADER_STRUCT.pack(len(message_bytes), len(payload)))
    sock.sendall(message_bytes)
    if payload:
        sock.sendall(payload)


def recv_message(sock: socket.socket) -> tuple[dict, bytes]:
    header = _recv_exact(sock, HEADER_STRUCT.size)
    message_size, payload_size = HEADER_STRUCT.unpack(header)
    message = json.loads(_recv_exact(sock, message_size).decode("utf-8"))
    payload = _recv_exact(sock, payload_size) if payload_size else b""
    return message, payload
