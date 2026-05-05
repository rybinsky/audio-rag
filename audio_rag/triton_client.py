import json
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import request

from .settings import AppSettings


class TritonHttpClient:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

    def ingest_podcast(
        self,
        *,
        source_id: str,
        audio_file: Path,
        transcript_file: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "inputs": [
                self._string_input("INPUT_SOURCE_ID", source_id),
                self._string_input("INPUT_AUDIO_PATH", self._map_path(audio_file)),
                self._string_input("INPUT_TRANSCRIPT_PATH", self._map_path(transcript_file) if transcript_file else ""),
                self._string_input("INPUT_METADATA_JSON", json.dumps(metadata or {}, ensure_ascii=False)),
            ]
        }
        response_payload = self._infer(base_url=base_url, model_name=self._settings.triton_http.model_ingest_name, payload=payload)
        return self._decode_single_string_output(response_payload)

    def ask(self, *, query_text: str, top_k: Optional[int] = None, base_url: Optional[str] = None) -> Dict[str, Any]:
        payload = {
            "inputs": [
                self._string_input("INPUT_QUERY_TEXT", query_text),
                self._string_input("INPUT_QUESTION_AUDIO_PATH", ""),
                self._string_input("INPUT_QUESTION_TRANSCRIPT_PATH", ""),
                self._uint32_input("INPUT_TOP_K", top_k or self._settings.retrieval.default_top_k),
            ]
        }
        response_payload = self._infer(base_url=base_url, model_name=self._settings.triton_http.model_query_name, payload=payload)
        return self._decode_single_string_output(response_payload)

    def ask_audio(
        self,
        *,
        question_audio_file: Path,
        question_transcript_file: Optional[Path] = None,
        top_k: Optional[int] = None,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "inputs": [
                self._string_input("INPUT_QUERY_TEXT", ""),
                self._string_input("INPUT_QUESTION_AUDIO_PATH", self._map_path(question_audio_file)),
                self._string_input(
                    "INPUT_QUESTION_TRANSCRIPT_PATH",
                    self._map_path(question_transcript_file) if question_transcript_file else "",
                ),
                self._uint32_input("INPUT_TOP_K", top_k or self._settings.retrieval.default_top_k),
            ]
        }
        response_payload = self._infer(base_url=base_url, model_name=self._settings.triton_http.model_query_name, payload=payload)
        return self._decode_single_string_output(response_payload)

    def _infer(self, *, base_url: Optional[str], model_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        resolved_base_url = (base_url or self._settings.triton_http.base_url).rstrip("/")
        endpoint = resolved_base_url + self._settings.triton_http.infer_endpoint_template.format(model_name=model_name)
        body = json.dumps(payload, ensure_ascii=False).encode(self._settings.transcript.encoding)
        http_request = request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": self._settings.triton_http.json_content_type},
            method="POST",
        )
        with request.urlopen(http_request) as response:
            return json.loads(response.read().decode(self._settings.transcript.encoding))

    @staticmethod
    def _decode_single_string_output(response_payload: Dict[str, Any]) -> Dict[str, Any]:
        outputs = response_payload["outputs"]
        raw_value = outputs[0]["data"][0]
        return json.loads(raw_value)

    @staticmethod
    def _string_input(name: str, value: str) -> Dict[str, Any]:
        return {"name": name, "shape": [1], "datatype": "BYTES", "data": [value]}

    @staticmethod
    def _uint32_input(name: str, value: int) -> Dict[str, Any]:
        return {"name": name, "shape": [1], "datatype": "UINT32", "data": [value]}

    def _map_path(self, path: Optional[Path]) -> str:
        if path is None:
            return ""
        resolved_path = path.expanduser().resolve()
        host_root = self._settings.triton_http.host_project_root
        container_root = self._settings.triton_http.container_project_root
        if host_root:
            resolved_host_root = Path(host_root).expanduser().resolve()
            try:
                relative_path = resolved_path.relative_to(resolved_host_root)
                return str(Path(container_root) / relative_path)
            except ValueError:
                return str(resolved_path)
        return str(resolved_path)
