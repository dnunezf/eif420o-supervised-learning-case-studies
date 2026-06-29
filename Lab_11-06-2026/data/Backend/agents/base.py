from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from Backend.schemas.contexto import ContextoMAS


class BaseAgent(ABC):
    nombre = "Agente"

    @abstractmethod
    def ejecutar(self, contexto: ContextoMAS) -> dict[str, Any]:
        """Ejecuta el agente usando el contexto MAS compartido."""
