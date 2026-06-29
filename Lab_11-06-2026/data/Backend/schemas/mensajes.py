from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MensajeAgente:
    agente: str
    tipo: str
    estado: str
    contenido: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "agente": self.agente,
            "tipo": self.tipo,
            "estado": self.estado,
            "contenido": self.contenido,
            "timestamp": self.timestamp,
        }
