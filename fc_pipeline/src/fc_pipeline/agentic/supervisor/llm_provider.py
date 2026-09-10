"""Provider-agnostic LLM loader reading configuration from environment variables."""

import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI


def get_supervisor_llm() -> BaseChatModel:
    """Loads a configurable, provider-agnostic BaseChatModel instance from environment variables.
    
    Required Environment Variables:
      - SUPERVISOR_LLM_ENDPOINT: The base URL of the OpenAI-compatible REST API endpoint
        (e.g., 'https://api.groq.com/openai/v1' or local inference server)
      - SUPERVISOR_LLM_MODEL: The identifier of the model to instantiate
        (e.g., 'llama-3.3-70b-versatile' on Groq, or local model)
      
    Optional / Authentication Environment Variables:
      - SUPERVISOR_LLM_API_KEY: Primary authentication token
      - GROQ_API_KEY: Fallback authentication token when connecting to Groq
      
    Raises:
      EnvironmentError: If SUPERVISOR_LLM_ENDPOINT or SUPERVISOR_LLM_MODEL is missing,
                        or if an authenticated endpoint (like Groq) is used without an API key.
    """
    endpoint = os.getenv("SUPERVISOR_LLM_ENDPOINT")
    api_key = os.getenv("SUPERVISOR_LLM_API_KEY") or os.getenv("GROQ_API_KEY")
    model_name = os.getenv("SUPERVISOR_LLM_MODEL")

    missing_vars = []
    if not endpoint or not endpoint.strip():
        missing_vars.append("SUPERVISOR_LLM_ENDPOINT")
    if not model_name or not model_name.strip():
        missing_vars.append("SUPERVISOR_LLM_MODEL")

    if missing_vars:
        raise EnvironmentError(
            f"Missing required LLM environment configuration: {', '.join(missing_vars)}. "
            "Please ensure these variables are defined in your environment or .env file."
        )

    endpoint_clean = endpoint.strip().strip('"').strip("'")
    model_clean = model_name.strip().strip('"').strip("'")
    api_key_clean = api_key.strip().strip('"').strip("'") if api_key else None

    # Hosted providers like Groq require an authentic API key.
    is_groq = "groq.com" in endpoint_clean.lower()
    if is_groq and (not api_key_clean or api_key_clean == "not-required"):
        raise EnvironmentError(
            "Groq endpoint detected but no valid API key was found. "
            "Please set SUPERVISOR_LLM_API_KEY or GROQ_API_KEY in your .env file."
        )

    # ChatOpenAI client library validation requires a non-empty string for api_key.
    # For no-auth local endpoints (e.g. CDAC / vLLM / Ollama), we pass 'not-required' as a dummy string.
    effective_api_key = api_key_clean if (api_key_clean and api_key_clean.strip()) else "not-required"

    # Instantiate via standard OpenAI-compatible REST interface
    return ChatOpenAI(
        base_url=endpoint_clean,
        api_key=effective_api_key,
        model=model_clean,
        temperature=0.0,
    )


# Backwards compatibility alias
load_supervisor_llm = get_supervisor_llm


