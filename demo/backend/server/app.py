# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from typing import Any, Generator

from app_conf import (
    GALLERY_PATH,
    GALLERY_PREFIX,
    POSTERS_PATH,
    POSTERS_PREFIX,
    TEMP_PATH,
    TEMP_PREFIX,
    UPLOADS_PATH,
    UPLOADS_PREFIX,
)
from data.loader import preload_data
from data.schema import schema
from data.store import set_videos
from flask import (
    Flask,
    make_response,
    Request,
    jsonify,
    request,
    Response,
    send_from_directory,
)
from flask_cors import CORS
from inference.data_types import PropagateDataResponse, PropagateInVideoRequest
from inference.multipart import MultipartResponseBuilder
from inference.predictor import InferenceAPI
from strawberry.flask.views import GraphQLView
import json

logger = logging.getLogger(__name__)

app = Flask(__name__)
cors = CORS(app, supports_credentials=True)

videos = preload_data()
set_videos(videos)

inference_api = InferenceAPI()


@app.route("/healthy")
def healthy() -> Response:
    return make_response("OK", 200)


@app.route(f"/{GALLERY_PREFIX}/<path:path>", methods=["GET"])
def send_gallery_video(path: str) -> Response:
    try:
        return send_from_directory(
            GALLERY_PATH,
            path,
        )
    except:
        raise ValueError("resource not found")


@app.route(f"/{POSTERS_PREFIX}/<path:path>", methods=["GET"])
def send_poster_image(path: str) -> Response:
    try:
        return send_from_directory(
            POSTERS_PATH,
            path,
        )
    except:
        raise ValueError("resource not found")


@app.route(f"/{UPLOADS_PREFIX}/<path:path>", methods=["GET"])
def send_uploaded_video(path: str):
    try:
        return send_from_directory(
            UPLOADS_PATH,
            path,
        )
    except:
        raise ValueError("resource not found")
    
@app.route(f"/mask", methods=["POST"])
def predict_image() -> Response:
    data = request.json
    start_time = time.time()
    res = inference_api.predict_image(data["url"],data["points"],data["labels"],None,True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"mask生成时间: {elapsed_time:.6f} 秒")
    return Response(
        res,
        mimetype="image/jpeg",
        headers={
            "Content-Disposition": "attachment; filename=mask.jpg" 
        }
    )

@app.route(f"/masks", methods=["POST"])
def generate_masks() -> Response:
    data = request.json
    start_time = time.time()
    res = inference_api.generate_masks(data["url"])
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"masks生成时间: {elapsed_time:.6f} 秒")
    return Response(
        res,
        mimetype="image/png",
        headers={
            "Content-Disposition": "attachment; filename=mask.png" 
        }
    )


@app.route("/image_masks_save", methods=["POST"])
def image_masks_save() -> Response:
    data = request.get_json(silent=True) or {}
    image_input = data.get("url") or data.get("path")
    filename_prefix = data.get("name") or "mask"

    if not image_input:
        return jsonify({"error": "url or path is required"}), 400

    timestamp = str(int(time.time() * 1000))
    output_dir = TEMP_PATH / timestamp

    try:
        saved_files = inference_api.save_masks_to_dir(
            image_input=image_input,
            output_dir=output_dir,
            filename_prefix=filename_prefix,
        )
    except Exception as exc:
        logger.exception("failed to save image masks")
        return jsonify({"error": f"failed to save masks: {exc}"}), 500

    rel_dir = f"{TEMP_PREFIX}/{timestamp}"
    masks_payload = [{"path": f"{rel_dir}/{name}"} for name in saved_files]
    return jsonify(
        {
            "count": len(saved_files),
            "saved_dir": rel_dir,
            "masks": masks_payload,
        }
    )

@app.route("/image_masks", methods=["POST"])
def image_masks() -> Response:
    """
    输入图片（url/path/data uri），直接返回全局 mask（base64 PNG 列表），不落盘。
    """
    data = request.get_json(silent=True) or {}
    image_input = data.get("url") or data.get("path")
    if not image_input:
        return jsonify({"error": "url or path is required"}), 400

    try:
        masks_payload = inference_api.generate_masks_base64(image_input=image_input)
    except Exception as exc:
        logger.exception("failed to generate masks in memory")
        return jsonify({"error": f"failed to generate masks: {exc}"}), 500

    return jsonify({
        "count": len(masks_payload),
        "masks": masks_payload,
    })


# TOOD: Protect route with ToS permission check
@app.route("/propagate_in_video", methods=["POST"])
def propagate_in_video() -> Response:
    data = request.json
    args = {
        "session_id": data["session_id"],
        "start_frame_index": data.get("start_frame_index", 0),
    }

    boundary = "frame"
    frame = gen_track_with_mask_stream(boundary, **args)
    return Response(frame, mimetype="multipart/x-savi-stream; boundary=" + boundary)


def gen_track_with_mask_stream(
    boundary: str,
    session_id: str,
    start_frame_index: int,
) -> Generator[bytes, None, None]:
    with inference_api.autocast_context():
        request = PropagateInVideoRequest(
            type="propagate_in_video",
            session_id=session_id,
            start_frame_index=start_frame_index,
        )

        for chunk in inference_api.propagate_in_video(request=request):
            yield MultipartResponseBuilder.build(
                boundary=boundary,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Frame-Current": "-1",
                    # Total frames minus the reference frame
                    "Frame-Total": "-1",
                    "Mask-Type": "RLE[]",
                },
                body=chunk.to_json().encode("UTF-8"),
            ).get_message()


class MyGraphQLView(GraphQLView):
    def get_context(self, request: Request, response: Response) -> Any:
        return {"inference_api": inference_api}


# Add GraphQL route to Flask app.
app.add_url_rule(
    "/graphql",
    view_func=MyGraphQLView.as_view(
        "graphql_view",
        schema=schema,
        # Disable GET queries
        # https://strawberry.rocks/docs/operations/deployment
        # https://strawberry.rocks/docs/integrations/flask
        allow_queries_via_get=False,
        # Strawberry recently changed multipart request handling, which now
        # requires enabling support explicitly for views.
        # https://github.com/strawberry-graphql/strawberry/issues/3655
        multipart_uploads_enabled=True,
    ),
)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7263)
