"""Собственный Streamlit-компонент видеоплеера для просмотра видео с обратной связью по временной метке"""

from pathlib import Path

import streamlit.components.v1 as components

_component = components.declare_component(
    "video_player",
    path=str(Path(__file__).parent),
)


def video_player(url: str, height: int = 400, key: str | None = None) -> dict | None:
    """
    Видеоплеер с обратной связью по временной метке.

    Args:
        url: URL видеофайла (.mp4, .webm, .ogg)
        height: Высота плеера в пикселях
        key: Уникальный ключ Streamlit-компонента

    Returns:
        dict с полями currentTime, playing, duration — или None до первого взаимодействия
    """
    return _component(url=url, height=height, key=key, default=None)
