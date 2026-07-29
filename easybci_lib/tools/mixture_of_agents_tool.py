#!/usr/bin/env python3
"""
Mixture-of-Agents Tool Module

This module implements the Mixture-of-Agents (MoA) methodology that leverages
the collective strengths of multiple LLMs through a layered architecture to
achieve state-of-the-art performance on complex reasoning tasks.

Based on the research paper: "Mixture-of-Agents Enhances Large Language Model Capabilities"
by Junlin Wang et al. (arXiv:2406.04692v1)

Key Features:
- Multi-layer LLM collaboration for enhanced reasoning
- Parallel processing of reference models for efficiency
- Intelligent aggregation and synthesis of diverse responses
- Specialized for extremely difficult problems requiring intense reasoning
- Optimized for coding, mathematics, and complex analytical tasks

Available Tool:
- mixture_of_agents_tool: Process complex queries using multiple frontier models

Architecture:
1. Reference models generate diverse initial responses in parallel
2. Aggregator model synthesizes responses into a high-quality output
3. Multiple layers can be used for iterative refinement (future enhancement)

Default Models:
- Reference Models: claude-opus-4.6, gemini-2.5-pro, gpt-5.4-pro, deepseek-v3.2
- Aggregator Model: claude-opus-4.6 (highest capability for synthesis)

Configuration:
    Provider/endpoint is configured via cli-config.yaml under auxiliary.moa:
      auxiliary:
        moa:
          provider: "openrouter"   # or "custom", "anthropic", etc.
          base_url: "https://..."  # custom OpenAI-compatible endpoint
          api_key: "sk-..."        # API key for the endpoint
          reference_models:        # override default reference model list
            - "model-a"
            - "model-b"
          aggregator_model: "model-c"

    Fallback constants at the top of this file are used when no config is set.

Usage:
    from mixture_of_agents_tool import mixture_of_agents_tool
    import asyncio
    
    # Process a complex query
    result = await mixture_of_agents_tool(
        user_prompt="Solve this complex mathematical proof..."
    )
"""

import json
import logging
import os
import asyncio
import datetime
from typing import Dict, Any, List, Optional
from easybci_agent.auxiliary_client import async_call_llm, extract_content_or_reasoning
from easybci_lib.tools.debug_helpers import DebugSession
import sys

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration — overridable via cli-config.yaml auxiliary.moa section
# ---------------------------------------------------------------------------

DEFAULT_REFERENCE_MODELS = [
    "anthropic/claude-opus-4.6",
    "google/gemini-3.5-pro",
    "openai/gpt-5.5-pro",
    "deepseek/deepseek-v4-pro",
]

DEFAULT_AGGREGATOR_MODEL = "anthropic/claude-opus-4.6"

# Temperature settings optimized for MoA performance
REFERENCE_TEMPERATURE = 0.6
AGGREGATOR_TEMPERATURE = 0.4

# Failure handling configuration
MIN_SUCCESSFUL_REFERENCES = 1


def _load_moa_config() -> Dict[str, Any]:
    """Load MoA configuration from auxiliary.moa in cli-config.yaml."""
    try:
        from easybci_cli.config import load_config
        config = load_config()
    except (ImportError, Exception):
        return {}
    aux = config.get("auxiliary", {}) if isinstance(config, dict) else {}
    moa_cfg = aux.get("moa", {}) if isinstance(aux, dict) else {}
    return moa_cfg if isinstance(moa_cfg, dict) else {}


def _get_reference_models() -> List[str]:
    """Get reference models from config or fall back to defaults."""
    cfg = _load_moa_config()
    models = cfg.get("reference_models")
    if models and isinstance(models, list) and all(isinstance(m, str) for m in models):
        return models
    return DEFAULT_REFERENCE_MODELS


def _get_aggregator_model() -> str:
    """Get aggregator model from config or fall back to default."""
    cfg = _load_moa_config()
    model = cfg.get("aggregator_model")
    if model and isinstance(model, str):
        return model
    return DEFAULT_AGGREGATOR_MODEL

# System prompt for the aggregator model (from the research paper)
AGGREGATOR_SYSTEM_PROMPT = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

_debug = DebugSession("moa_tools", env_var="MOA_TOOLS_DEBUG")


def _construct_aggregator_prompt(system_prompt: str, responses: List[str]) -> str:
    """
    Construct the final system prompt for the aggregator including all model responses.
    
    Args:
        system_prompt (str): Base system prompt for aggregation
        responses (List[str]): List of responses from reference models
        
    Returns:
        str: Complete system prompt with enumerated responses
    """
    response_text = "\n".join([f"{i+1}. {response}" for i, response in enumerate(responses)])
    return f"{system_prompt}\n\n{response_text}"


async def _run_reference_model_safe(
    model: str,
    user_prompt: str,
    temperature: float = REFERENCE_TEMPERATURE,
    max_tokens: int = 32000,
    max_retries: int = 6
) -> tuple[str, str, bool]:
    """
    Run a single reference model with retry logic and graceful failure handling.

    Args:
        model (str): Model identifier to use
        user_prompt (str): The user's query
        temperature (float): Sampling temperature for response generation
        max_tokens (int): Maximum tokens in response
        max_retries (int): Maximum number of retry attempts

    Returns:
        tuple[str, str, bool]: (model_name, response_content_or_error, success_flag)
    """
    for attempt in range(max_retries):
        try:
            logger.info("Querying %s (attempt %s/%s)", model, attempt + 1, max_retries)

            call_kwargs = {
                "task": "moa",
                "model": model,
                "messages": [{"role": "user", "content": user_prompt}],
                "max_tokens": max_tokens,
                "extra_body": {"reasoning": {"enabled": True, "effort": "xhigh"}},
            }
            if not model.lower().startswith("gpt-"):
                call_kwargs["temperature"] = temperature

            response = await async_call_llm(**call_kwargs)

            content = extract_content_or_reasoning(response)
            if not content:
                logger.warning("%s returned empty content (attempt %s/%s), retrying", model, attempt + 1, max_retries)
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(2 ** (attempt + 1), 60))
                    continue
            logger.info("%s responded (%s characters)", model, len(content))
            return model, content, True
            
        except Exception as e:
            error_str = str(e)
            # Keep retry-path logging concise; full tracebacks are reserved for
            # terminal failure paths so long-running MoA retries don't flood logs.
            if "invalid" in error_str.lower():
                logger.warning("%s invalid request error (attempt %s): %s", model, attempt + 1, error_str)
            elif "rate" in error_str.lower() or "limit" in error_str.lower():
                logger.warning("%s rate limit error (attempt %s): %s", model, attempt + 1, error_str)
            else:
                logger.warning("%s unknown error (attempt %s): %s", model, attempt + 1, error_str)

            if attempt < max_retries - 1:
                # Exponential backoff for rate limiting: 2s, 4s, 8s, 16s, 32s, 60s
                sleep_time = min(2 ** (attempt + 1), 60)
                logger.info("Retrying in %ss...", sleep_time)
                await asyncio.sleep(sleep_time)
            else:
                error_msg = f"{model} failed after {max_retries} attempts: {error_str}"
                logger.error("%s", error_msg, exc_info=True)
                return model, error_msg, False


async def _run_aggregator_model(
    agg_model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = AGGREGATOR_TEMPERATURE,
    max_tokens: int = None
) -> str:
    """
    Run the aggregator model to synthesize the final response.

    Args:
        agg_model (str): Aggregator model identifier
        system_prompt (str): System prompt with all reference responses
        user_prompt (str): Original user query
        temperature (float): Focused temperature for consistent aggregation
        max_tokens (int): Maximum tokens in final response

    Returns:
        str: Synthesized final response
    """
    logger.info("Running aggregator model: %s", agg_model)

    call_kwargs = {
        "task": "moa",
        "model": agg_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "extra_body": {"reasoning": {"enabled": True, "effort": "xhigh"}},
    }
    if not agg_model.lower().startswith("gpt-"):
        call_kwargs["temperature"] = temperature

    response = await async_call_llm(**call_kwargs)
    content = extract_content_or_reasoning(response)

    if not content:
        logger.warning("Aggregator returned empty content, retrying once")
        response = await async_call_llm(**call_kwargs)
        content = extract_content_or_reasoning(response)

    logger.info("Aggregation complete (%s characters)", len(content))
    return content


async def mixture_of_agents_tool(
    user_prompt: str,
    reference_models: Optional[List[str]] = None,
    aggregator_model: Optional[str] = None
) -> str:
    """
    Process a complex query using the Mixture-of-Agents methodology.
    
    This tool leverages multiple frontier language models to collaboratively solve
    extremely difficult problems requiring intense reasoning. It's particularly
    effective for:
    - Complex mathematical proofs and calculations
    - Advanced coding problems and algorithm design
    - Multi-step analytical reasoning tasks
    - Problems requiring diverse domain expertise
    - Tasks where single models show limitations
    
    The MoA approach uses a fixed 2-layer architecture:
    1. Layer 1: Multiple reference models generate diverse responses in parallel (temp=0.6)
    2. Layer 2: Aggregator model synthesizes the best elements into final response (temp=0.4)
    
    Args:
        user_prompt (str): The complex query or problem to solve
        reference_models (Optional[List[str]]): Custom reference models to use
        aggregator_model (Optional[str]): Custom aggregator model to use
    
    Returns:
        str: JSON string containing the MoA results with the following structure:
             {
                 "success": bool,
                 "response": str,
                 "models_used": {
                     "reference_models": List[str],
                     "aggregator_model": str
                 },
                 "processing_time": float
             }
    
    Raises:
        Exception: If MoA processing fails or API key is not set
    """
    start_time = datetime.datetime.now()

    # Resolve models from config
    ref_models = reference_models or _get_reference_models()
    agg_model = aggregator_model or _get_aggregator_model()

    debug_call_data = {
        "parameters": {
            "user_prompt": user_prompt[:200] + "..." if len(user_prompt) > 200 else user_prompt,
            "reference_models": ref_models,
            "aggregator_model": agg_model,
            "reference_temperature": REFERENCE_TEMPERATURE,
            "aggregator_temperature": AGGREGATOR_TEMPERATURE,
            "min_successful_references": MIN_SUCCESSFUL_REFERENCES
        },
        "error": None,
        "success": False,
        "reference_responses_count": 0,
        "failed_models_count": 0,
        "failed_models": [],
        "final_response_length": 0,
        "processing_time_seconds": 0,
        "models_used": {}
    }

    try:
        logger.info("Starting Mixture-of-Agents processing...")
        logger.info("Query: %s", user_prompt[:100])

        logger.info("Using %s reference models in 2-layer MoA architecture", len(ref_models))
        
        # Layer 1: Generate diverse responses from reference models (with failure handling)
        logger.info("Layer 1: Generating reference responses...")
        model_results = await asyncio.gather(*[
            _run_reference_model_safe(model, user_prompt, REFERENCE_TEMPERATURE)
            for model in ref_models
        ])
        
        # Separate successful and failed responses
        successful_responses = []
        failed_models = []
        
        for model_name, content, success in model_results:
            if success:
                successful_responses.append(content)
            else:
                failed_models.append(model_name)
        
        successful_count = len(successful_responses)
        failed_count = len(failed_models)
        
        logger.info("Reference model results: %s successful, %s failed", successful_count, failed_count)
        
        if failed_models:
            logger.warning("Failed models: %s", ', '.join(failed_models))
        
        # Check if we have enough successful responses to proceed
        if successful_count < MIN_SUCCESSFUL_REFERENCES:
            raise ValueError(f"Insufficient successful reference models ({successful_count}/{len(ref_models)}). Need at least {MIN_SUCCESSFUL_REFERENCES} successful responses.")
        
        debug_call_data["reference_responses_count"] = successful_count
        debug_call_data["failed_models_count"] = failed_count
        debug_call_data["failed_models"] = failed_models
        
        # Layer 2: Aggregate responses using the aggregator model
        logger.info("Layer 2: Synthesizing final response...")
        aggregator_system_prompt = _construct_aggregator_prompt(
            AGGREGATOR_SYSTEM_PROMPT, 
            successful_responses
        )
        
        final_response = await _run_aggregator_model(
            agg_model,
            aggregator_system_prompt,
            user_prompt,
            AGGREGATOR_TEMPERATURE
        )
        
        # Calculate processing time
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        logger.info("MoA processing completed in %.2f seconds", processing_time)
        
        # Prepare successful response (only final aggregated result, minimal fields)
        result = {
            "success": True,
            "response": final_response,
            "models_used": {
                "reference_models": ref_models,
                "aggregator_model": agg_model
            }
        }
        
        debug_call_data["success"] = True
        debug_call_data["final_response_length"] = len(final_response)
        debug_call_data["processing_time_seconds"] = processing_time
        debug_call_data["models_used"] = result["models_used"]
        
        # Log debug information
        _debug.log_call("mixture_of_agents_tool", debug_call_data)
        _debug.save()
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Error in MoA processing: {str(e)}"
        logger.error("%s", error_msg, exc_info=True)
        
        # Calculate processing time even for errors
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Prepare error response (minimal fields)
        result = {
            "success": False,
            "response": "MoA processing failed. Please try again or use a single model for this query.",
            "models_used": {
                "reference_models": ref_models,
                "aggregator_model": agg_model
            },
            "error": error_msg
        }
        
        debug_call_data["error"] = error_msg
        debug_call_data["processing_time_seconds"] = processing_time
        _debug.log_call("mixture_of_agents_tool", debug_call_data)
        _debug.save()
        
        return json.dumps(result, indent=2, ensure_ascii=False)


def check_moa_requirements() -> bool:
    """Check if a provider is available for MoA calls.

    Returns True if any LLM provider is configured (OpenRouter, custom endpoint,
    Anthropic, etc.) — not limited to OpenRouter.
    """
    if os.getenv("OPENROUTER_API_KEY"):
        return True
    if os.getenv("ANTHROPIC_API_KEY"):
        return True
    if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL"):
        return True
    cfg = _load_moa_config()
    if cfg.get("base_url") and cfg.get("api_key"):
        return True
    if cfg.get("provider") and cfg.get("provider") != "auto":
        return True
    return False



def get_moa_configuration() -> Dict[str, Any]:
    """
    Get the current MoA configuration settings.

    Returns:
        Dict[str, Any]: Dictionary containing all configuration parameters
    """
    ref_models = _get_reference_models()
    agg_model = _get_aggregator_model()
    return {
        "reference_models": ref_models,
        "aggregator_model": agg_model,
        "reference_temperature": REFERENCE_TEMPERATURE,
        "aggregator_temperature": AGGREGATOR_TEMPERATURE,
        "min_successful_references": MIN_SUCCESSFUL_REFERENCES,
        "total_reference_models": len(ref_models),
        "failure_tolerance": f"{len(ref_models) - MIN_SUCCESSFUL_REFERENCES}/{len(ref_models)} models can fail"
    }


if __name__ == "__main__":
    print("Mixture-of-Agents Tool Module")
    print("=" * 50)

    api_available = check_moa_requirements()

    if not api_available:
        print("No LLM provider configured for MoA.")
        print("Configure via cli-config.yaml auxiliary.moa section,")
        print("or set OPENROUTER_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY.")
        sys.exit(1)
    else:
        print("Provider available")

    config = get_moa_configuration()
    print(f"\nReference models ({len(config['reference_models'])}): {', '.join(config['reference_models'])}")
    print(f"Aggregator model: {config['aggregator_model']}")
    print(f"Reference temperature: {config['reference_temperature']}")
    print(f"Aggregator temperature: {config['aggregator_temperature']}")
    print(f"Failure tolerance: {config['failure_tolerance']}")
    print(f"Min successful models: {config['min_successful_references']}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from easybci_lib.tools.registry import registry

MOA_SCHEMA = {
    "name": "mixture_of_agents",
    "description": "Route a hard problem through multiple frontier LLMs collaboratively. Makes 5 API calls (4 reference models + 1 aggregator) with maximum reasoning effort — use sparingly for genuinely difficult problems. Best for: complex math, advanced algorithms, multi-step analytical reasoning, problems benefiting from diverse perspectives.",
    "parameters": {
        "type": "object",
        "properties": {
            "user_prompt": {
                "type": "string",
                "description": "The complex query or problem to solve using multiple AI models. Should be a challenging problem that benefits from diverse perspectives and collaborative reasoning."
            }
        },
        "required": ["user_prompt"]
    }
}

registry.register(
    name="mixture_of_agents",
    toolset="moa",
    schema=MOA_SCHEMA,
    handler=lambda args, **kw: mixture_of_agents_tool(user_prompt=args.get("user_prompt", "")),
    check_fn=check_moa_requirements,
    is_async=True,
    emoji="🧠",
)
