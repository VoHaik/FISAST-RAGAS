import sys
from unittest.mock import MagicMock

# Mock VertexAI modules to prevent Ragas import errors when langchain-community>=0.4.0 is used.
mock_vertex = MagicMock()
sys.modules['langchain_community.chat_models.vertexai'] = mock_vertex
sys.modules['langchain_community.llms.vertexai'] = mock_vertex
