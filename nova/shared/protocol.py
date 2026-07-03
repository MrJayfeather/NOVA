from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter

PROTOCOL_VERSION = 1


class Hello(BaseModel):
    type: Literal["hello"] = "hello"
    protocol: int = PROTOCOL_VERSION
    profile: str
    persona: str
    token: str = ""


class Frame(BaseModel):
    type: Literal["frame"] = "frame"
    ts: float
    jpeg_b64: str
    kind: Literal["periodic", "burst"] = "periodic"
    burst_id: Optional[str] = None
    seq: int = 0
    cursor_x: Optional[int] = None
    cursor_y: Optional[int] = None


class DetectorEvent(BaseModel):
    type: Literal["event"] = "event"
    ts: float
    event: Literal["scene_change", "motion_burst"]


class AudioSegment(BaseModel):
    type: Literal["audio_segment"] = "audio_segment"
    ts: float
    pcm_b64: str
    sample_rate: int = 16000
    source: str = "local_mic"


class Hotkey(BaseModel):
    type: Literal["hotkey"] = "hotkey"
    action: Literal["comment_now", "toggle_pause", "feedback_up", "feedback_down"]


class HelloAck(BaseModel):
    type: Literal["hello_ack"] = "hello_ack"
    protocol: int = PROTOCOL_VERSION
    mock: bool


class SpeakStart(BaseModel):
    type: Literal["speak_start"] = "speak_start"
    utterance_id: str
    text: str
    reason: str
    sample_rate: int
    heard: str = ""  # что распознал ASR (для отладки «она меня не так поняла»)


class AudioChunk(BaseModel):
    type: Literal["audio_chunk"] = "audio_chunk"
    utterance_id: str
    seq: int
    pcm_b64: str


class SpeakEnd(BaseModel):
    type: Literal["speak_end"] = "speak_end"
    utterance_id: str


ClientMessage = Annotated[
    Union[Hello, Frame, DetectorEvent, AudioSegment, Hotkey],
    Field(discriminator="type"),
]
ServerMessage = Annotated[
    Union[HelloAck, SpeakStart, AudioChunk, SpeakEnd],
    Field(discriminator="type"),
]

_client_adapter: TypeAdapter = TypeAdapter(ClientMessage)
_server_adapter: TypeAdapter = TypeAdapter(ServerMessage)


def parse_client_message(data: str | bytes):
    return _client_adapter.validate_json(data)


def parse_server_message(data: str | bytes):
    return _server_adapter.validate_json(data)


def dump_message(msg: BaseModel) -> str:
    return msg.model_dump_json()
