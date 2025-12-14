# Initialize Jinja2 environment
import dataclasses
from datetime import datetime
import os

from jinja2 import Environment, FileSystemLoader, TemplateNotFound, select_autoescape
from langchain.agents import AgentState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

from config.configuration import Configuration

env = Environment(
    loader=FileSystemLoader(os.path.dirname(__file__)),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


def get_prompt_template(prompt_name: str, locale: str = "en-US") -> str:
    """
    Load and return a prompt template using Jinja2 with locale support.

    Args:
        prompt_name: Name of the prompt template file (without .md extension)
        locale: Language locale (e.g., en-US, zh-CN). Defaults to en-US

    Returns:
        The template string with proper variable substitution syntax
    """
    try:
        # Normalize locale format
        normalized_locale = locale.replace("-", "_") if locale and locale.strip() else "en_US"

        # Try locale-specific template first (e.g., researcher.zh_CN.md)
        try:
            template = env.get_template(f"{prompt_name}.{normalized_locale}.md")
            return template.render()
        except TemplateNotFound:
            # Fallback to English template if locale-specific not found
            template = env.get_template(f"{prompt_name}.md")
            return template.render()
    except Exception as e:
        raise ValueError(f"Error loading template {prompt_name} for locale {locale}: {e}")


def apply_prompt_template(
        prompt_name: str, state: AgentState, configurable: Configuration = None, locale: str = "en-US"
) -> list[BaseMessage]:
    """
    Apply template variables to a prompt template and return formatted LangChain messages.

    Args:
        prompt_name: Name of the prompt template to use
        state: Current agent state containing variables to substitute
        configurable: Configuration object with additional variables
        locale: Language locale for template selection (e.g., en-US, zh-CN)

    Returns:
        List of LangChain message objects with the system prompt as the first message
    """
    # Convert state to dict for template rendering
    state_vars = {
        "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
        "locale": locale,  # Add locale to template variables
        **state,
    }

    # Add configurable variables
    if configurable:
        state_vars.update(dataclasses.asdict(configurable))

    try:
        # Normalize locale format
        normalized_locale = locale.replace("-", "_") if locale and locale.strip() else "en_US"

        # Try locale-specific template first
        try:
            template = env.get_template(f"{prompt_name}.{normalized_locale}.md")
        except TemplateNotFound:
            # Fallback to English template
            template = env.get_template(f"{prompt_name}.md")

        system_prompt = template.render(**state_vars)
        
        # Convert dictionary messages to LangChain message objects
        messages = [SystemMessage(content=system_prompt)]
        
        for msg_dict in state["messages"]:
            role = msg_dict.get("role", "user")
            content = msg_dict.get("content")
            
            # Log content type for debugging
            if isinstance(content, list):
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Converting message with role={role}, content type=list, length={len(content)}")
                for i, item in enumerate(content):
                    if isinstance(item, dict):
                        logger.debug(f"  Content item {i}: keys={list(item.keys())}")
            
            if role == "system":
                messages.append(SystemMessage(content=content))
            elif role == "user":
                # For user messages, content can be a list (text + images) or string
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                # Default to HumanMessage for unknown roles
                messages.append(HumanMessage(content=content))
        
        return messages
    except Exception as e:
        raise ValueError(f"Error applying template {prompt_name} for locale {locale}: {e}")

if __name__ == '__main__':
    from log import get_logger
    logger = get_logger(__name__)
    # logger.info(get_prompt_template("slide_analyzer","en_US"))
    logger.info(apply_prompt_template("slide_analyzer", AgentState(messages=[{"role": "user", "content": "Hello"}]), locale="en_US"))