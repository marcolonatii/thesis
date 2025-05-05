# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Union


class MultipartResponseBuilder:
    """
    Builder class for generating multipart response messages suitable
    for streaming. Each part will contain the specified headers and body,
    and parts are separated by a boundary line.

    IMPORTANT: We prepend '\\r\\n' to each boundary so that the client,
    which looks for '\\r\\n--boundary', can correctly find and split parts.
    """

    message: bytes

    def __init__(self, boundary: str) -> None:
        # CHANGED: We add a preceding "\r\n" to each boundary
        self.message = b"\r\n--" + boundary.encode("utf-8") + b"\r\n"

    @classmethod
    def build(
        cls, boundary: str, headers: Dict[str, str], body: Union[str, bytes]
    ) -> "MultipartResponseBuilder":
        """
        Build a single multipart chunk using the given boundary, headers, and body.
        """
        builder = cls(boundary=boundary)
        for k, v in headers.items():
            builder.__append_header(key=k, value=v)
        if isinstance(body, bytes):
            builder.__append_body(body)
        elif isinstance(body, str):
            builder.__append_body(body.encode("utf-8"))
        else:
            raise ValueError(
                f"body needs to be of type bytes or str but got {type(body)}"
            )
        return builder

    def get_message(self) -> bytes:
        """
        Return the complete multipart part as bytes.
        """
        return self.message

    def __append_header(self, key: str, value: str) -> "MultipartResponseBuilder":
        """
        Append a single header to the current part, like:
            Content-Type: image/jpeg
        """
        self.message += key.encode("utf-8") + b": " + value.encode("utf-8") + b"\r\n"
        return self

    def __close_header(self) -> "MultipartResponseBuilder":
        """
        Close the headers with an additional CRLF (empty line).
        """
        self.message += b"\r\n"
        return self

    def __append_body(self, body: bytes) -> "MultipartResponseBuilder":
        """
        Append the body (payload) to the part, preceded by Content-Length.
        """
        self.__append_header(key="Content-Length", value=str(len(body)))
        self.__close_header()
        self.message += body
        return self