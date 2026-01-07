from typing import Optional, Generator, List, Dict
from groq import Groq

from config import (
    GROQ_API_KEY,
    LLM_MODELS,
    DEFAULT_LLM_MODEL,
    SYSTEM_PROMPT_TR
)


class LLMGenerator:
    def __init__(self, api_key: str = GROQ_API_KEY):
        if not api_key:
            raise ValueError("GROQ_API_KEY required. Please check your .env file.")
        
        self.client = Groq(api_key=api_key)
        self.available_models = LLM_MODELS
        print("Groq LLM connection established")
    
    def _format_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        if not chat_history:
            return "No conversation history yet."
        
        formatted_parts = []
        for msg in chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:500]
            formatted_parts.append(f"{role}: {content}")
        
        return "\n".join(formatted_parts)
    
    def generate(
        self,
        question: str,
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        model_id: str = DEFAULT_LLM_MODEL,
        temperature: float = 0.1,
        max_tokens: int = 1024
    ) -> str:

        history_str = self._format_chat_history(chat_history or [])
        
        formatted_prompt = SYSTEM_PROMPT_TR.format(
            context=context,
            question=question,
            chat_history=history_str
        )
        
        try:
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "system",
                        "content": "Sen Türkçe yanıt veren bir finans uzmanısın."
                    },
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = f"LLM error: {str(e)}"
            print(f"{error_msg}")
            return "Connection error, please try again."
    
    def generate_stream(
        self,
        question: str,
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        model_id: str = DEFAULT_LLM_MODEL,
        temperature: float = 0.1,
        max_tokens: int = 1024
    ) -> Generator[str, None, None]:
        
        history_str = self._format_chat_history(chat_history or [])
        
        formatted_prompt = SYSTEM_PROMPT_TR.format(
            context=context,
            question=question,
            chat_history=history_str
        )
        
        try:
            stream = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "system",
                        "content": "Sen Türkçe yanıt veren bir finans uzmanısın."
                    },
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:

            yield "Connection error, please try again."
    
    @staticmethod
    def get_model_display_name(model_id: str) -> str:
        for display_name, mid in LLM_MODELS.items():
            if mid == model_id:
                return display_name
        return model_id
    
    @staticmethod
    def get_model_id(display_name: str) -> str:
        return LLM_MODELS.get(display_name, DEFAULT_LLM_MODEL)


_llm_generator_instance = None


def get_llm_generator() -> LLMGenerator:
    global _llm_generator_instance
    if _llm_generator_instance is None:
        _llm_generator_instance = LLMGenerator()
    return _llm_generator_instance


if __name__ == "__main__":
    print("LLM Generator Test")
    
    try:
        generator = LLMGenerator()
        
        test_context = """
        [Kaynak 1: kdv_mevzuati.pdf]
        Katma Değer Vergisi (KDV) genel oranı %18'dir. 
        Temel gıda maddeleri için indirimli oran %8 uygulanır.
        Bazı tarım ürünleri için %1 oranı geçerlidir.
        """
        
        test_question = "KDV oranları nelerdir?"
        
        print(f"\nSoru: {test_question}")
        print("Yanıt oluşturuluyor...")
        
        response = generator.generate(
            question=test_question,
            context=test_context,
            model_id="llama-3.3-70b-versatile"
        )
        
        print(f"\nYanıt:\n{response}")
        
    except ValueError as e:
        print(f"{e}")
        print(".env dosyasına GROQ_API_KEY ekleyin")
