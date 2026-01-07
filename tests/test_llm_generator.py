"""Unit tests for llm_generator module."""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestLLMGeneratorInit(unittest.TestCase):
    """Test LLMGenerator initialization."""

    @patch('llm_generator.Groq')
    def test_init_with_api_key(self, mock_groq):
        """Verify LLMGenerator initializes with valid API key."""
        from llm_generator import LLMGenerator
        
        generator = LLMGenerator(api_key="test_api_key")
        
        mock_groq.assert_called_with(api_key="test_api_key")
        self.assertIsNotNone(generator.client)

    def test_init_without_api_key_raises_error(self):
        """Verify LLMGenerator raises error without API key."""
        from llm_generator import LLMGenerator
        
        with self.assertRaises(ValueError) as context:
            LLMGenerator(api_key="")
        
        self.assertIn("GROQ_API_KEY", str(context.exception))


class TestLLMGeneratorFormatHistory(unittest.TestCase):
    """Test LLMGenerator chat history formatting."""

    @patch('llm_generator.Groq')
    def test_format_empty_history(self, mock_groq):
        """Verify empty history returns appropriate message."""
        from llm_generator import LLMGenerator
        
        generator = LLMGenerator(api_key="test_key")
        result = generator._format_chat_history([])
        
        self.assertIn("No conversation history", result)

    @patch('llm_generator.Groq')
    def test_format_history_with_messages(self, mock_groq):
        """Verify history with messages is formatted correctly."""
        from llm_generator import LLMGenerator
        
        generator = LLMGenerator(api_key="test_key")
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        
        result = generator._format_chat_history(history)
        
        self.assertIn("Hello", result)
        self.assertIn("Hi there", result)

    @patch('llm_generator.Groq')
    def test_format_history_truncates_long_content(self, mock_groq):
        """Verify long content is truncated in history."""
        from llm_generator import LLMGenerator
        
        generator = LLMGenerator(api_key="test_key")
        long_content = "A" * 1000
        history = [{"role": "user", "content": long_content}]
        
        result = generator._format_chat_history(history)
        
        # Content should be truncated to 500 chars
        self.assertLess(len(result), 600)


class TestLLMGeneratorGenerate(unittest.TestCase):
    """Test LLMGenerator generate method."""

    @patch('llm_generator.Groq')
    def test_generate_success(self, mock_groq):
        """Verify generate returns response content."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq.return_value = mock_client
        
        from llm_generator import LLMGenerator
        generator = LLMGenerator(api_key="test_key")
        
        result = generator.generate(
            question="Test question",
            context="Test context"
        )
        
        self.assertEqual(result, "Test response")

    @patch('llm_generator.Groq')
    def test_generate_with_custom_params(self, mock_groq):
        """Verify generate uses custom parameters."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq.return_value = mock_client
        
        from llm_generator import LLMGenerator
        generator = LLMGenerator(api_key="test_key")
        
        generator.generate(
            question="Test",
            context="Context",
            model_id="custom-model",
            temperature=0.5,
            max_tokens=512
        )
        
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "custom-model")
        self.assertEqual(call_kwargs["temperature"], 0.5)
        self.assertEqual(call_kwargs["max_tokens"], 512)

    @patch('llm_generator.Groq')
    def test_generate_handles_exception(self, mock_groq):
        """Verify generate handles exceptions gracefully."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_groq.return_value = mock_client
        
        from llm_generator import LLMGenerator
        generator = LLMGenerator(api_key="test_key")
        
        result = generator.generate(question="Test", context="Context")
        
        # Should return error message in English
        self.assertIn("error", result.lower())


class TestLLMGeneratorStream(unittest.TestCase):
    """Test LLMGenerator streaming methods."""

    @patch('llm_generator.Groq')
    def test_generate_stream_yields_chunks(self, mock_groq):
        """Verify generate_stream yields content chunks."""
        mock_client = MagicMock()
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock(delta=MagicMock(content="Hello "))]
        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock(delta=MagicMock(content="World"))]
        
        mock_client.chat.completions.create.return_value = iter([mock_chunk1, mock_chunk2])
        mock_groq.return_value = mock_client
        
        from llm_generator import LLMGenerator
        generator = LLMGenerator(api_key="test_key")
        
        result = list(generator.generate_stream(
            question="Test",
            context="Context"
        ))
        
        self.assertEqual(result, ["Hello ", "World"])

    @patch('llm_generator.Groq')
    def test_generate_stream_handles_exception(self, mock_groq):
        """Verify generate_stream handles exceptions."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Stream Error")
        mock_groq.return_value = mock_client
        
        from llm_generator import LLMGenerator
        generator = LLMGenerator(api_key="test_key")
        
        result = list(generator.generate_stream(question="Test", context="Context"))
        
        # Should yield error message
        self.assertTrue(len(result) > 0)


class TestLLMGeneratorStaticMethods(unittest.TestCase):
    """Test LLMGenerator static methods."""

    def test_get_model_display_name_found(self):
        """Verify get_model_display_name returns display name when found."""
        from llm_generator import LLMGenerator
        
        result = LLMGenerator.get_model_display_name("llama-3.3-70b-versatile")
        
        self.assertEqual(result, "Llama 3.3 70B")

    def test_get_model_display_name_not_found(self):
        """Verify get_model_display_name returns model_id when not found."""
        from llm_generator import LLMGenerator
        
        result = LLMGenerator.get_model_display_name("unknown-model")
        
        self.assertEqual(result, "unknown-model")

    def test_get_model_id_found(self):
        """Verify get_model_id returns model_id when found."""
        from llm_generator import LLMGenerator
        
        result = LLMGenerator.get_model_id("Llama 3.3 70B")
        
        self.assertEqual(result, "llama-3.3-70b-versatile")

    def test_get_model_id_not_found(self):
        """Verify get_model_id returns default when not found."""
        from llm_generator import LLMGenerator
        from config import DEFAULT_LLM_MODEL
        
        result = LLMGenerator.get_model_id("Unknown Model")
        
        self.assertEqual(result, DEFAULT_LLM_MODEL)


class TestGetLLMGenerator(unittest.TestCase):
    """Test get_llm_generator singleton function."""

    @patch('llm_generator._llm_generator_instance', None)
    @patch('llm_generator.LLMGenerator')
    def test_get_llm_generator_creates_singleton(self, mock_class):
        """Verify get_llm_generator creates singleton instance."""
        import llm_generator
        llm_generator._llm_generator_instance = None
        
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        
        result1 = llm_generator.get_llm_generator()
        result2 = llm_generator.get_llm_generator()
        
        self.assertEqual(result1, result2)


if __name__ == "__main__":
    unittest.main()
