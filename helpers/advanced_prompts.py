"""
Advanced prompt templates for conversation, argument generation, and prediction.
These prompts use multi-shot examples and better structure from temp.py.
"""

class ConversationPrompts:
    """Improved conversation prompts with better structure."""
    
    AGENT_INSTRUCTIONS = "Carry on the conversation given to you. Speak in 3 sentences or less."
    
    @staticmethod
    def start_conversation_prompt(topic: str) -> str:
        """Prompt for starting a conversation about a topic."""
        return f"""{ConversationPrompts.AGENT_INSTRUCTIONS}

Start a conversation about: {topic}

Your response: """
    
    @staticmethod
    def continue_conversation_prompt(topic: str, conversation_history: str) -> str:
        """Prompt for continuing an ongoing conversation."""
        return f"""{ConversationPrompts.AGENT_INSTRUCTIONS}

Topic: {topic}

Conversation so far:
{conversation_history}

Your response: """


class PerspectivePrompts:
    """Prompts for extracting perspectives from conversations using few-shot examples."""
    
    @staticmethod
    def extract_perspective_prompt(topic: str, conversation_history: str) -> str:
        """Extract a new perspective from conversation using multi-shot prompting."""
        return f"""Topic: Will the United States win the most gold medals at the 2024 Olympics?

Conversation:
Agent A: The US has strong swimming and track programs that typically dominate. Their funding for Olympic sports is also substantial compared to many countries.
Agent B: But China has been investing heavily and their performance in recent Olympics shows they're catching up. They also have advantages in gymnastics and diving.
Agent A: True, but the US has depth across more sports categories. Even if China excels in specific areas, the US can medal in more diverse events.

State a new perspective that you believe in in one sentence from our conversation.
Perspective: The key factor will be which country can better leverage their strengths in high-medal-count sports like swimming and athletics.

Topic: Will Bitcoin reach $100,000 by the end of 2024?

Conversation:
Agent A: Institutional adoption is accelerating with Bitcoin ETFs getting approved. This brings legitimacy and more capital inflow.
Agent B: However, regulatory uncertainty still looms large. Any major crackdown could severely impact the price trajectory.
Agent A: The halving event in 2024 historically drives price increases due to supply constraints. This fundamental factor is predictable.

State a new perspective that you believe in in one sentence from our conversation.
Perspective: Bitcoin's price will depend more on regulatory clarity than technical factors like the halving event.

Topic: {topic}

Conversation:
{conversation_history}

State a new perspective that you believe in in one sentence from our conversation.
Perspective: """


class ArgumentPrompts:
    """Prompts for generating arguments using multi-shot examples."""
    
    @staticmethod
    def generate_argument_prompt(question: str) -> str:
        """Generate an argument for a question using multi-shot prompting."""
        return f"""Question: Will the United States win the most gold medals at the 2024 Olympics?
Argument: The US has historically strong performance across multiple sports and significant investment in Olympic training programs.

Question: Will Bitcoin reach $100,000 by the end of 2024?
Argument: Institutional adoption is increasing and Bitcoin ETFs are gaining approval, creating more mainstream demand.

Question: Will there be a major earthquake (magnitude 7+) in California in 2024?
Argument: California sits on active fault lines and seismic activity has been increasing in recent years.

{question}
Argument: """


class PredictionPrompts:
    """Prompts for making predictions with better yes/no structure."""
    
    @staticmethod
    def predict_with_arguments_prompt(question: str, arguments: list) -> str:
        """Make a prediction using collected arguments with one-shot example."""
        args_text = "\n".join([f"- {arg}" for arg in arguments])
        
        return f"""Will artificial intelligence surpass human performance in chess by 2000?

This question requires a Yes or No answer only.
Answer: Yes

You believe in the following arguments:
{args_text}

{question}

This question requires a Yes or No answer only.
Answer: """
    
    @staticmethod
    def basic_prediction_prompt(question: str) -> str:
        """Basic prediction prompt with one-shot example."""
        return f"""Will artificial intelligence surpass human performance in chess by 2000?

This question requires a Yes or No answer only.
Answer: Yes

{question}

This question requires a Yes or No answer only.
Answer: """


class StrengthCalculationHelpers:
    """Helper functions for calculating opinion strength from token probabilities."""
    
    @staticmethod
    def extract_yes_no_probabilities(probs_dict: dict) -> tuple:
        """Extract yes/no probabilities from token probability dictionary."""
        yes_prob = 0.0
        no_prob = 0.0
        
        for token, prob in probs_dict.items():
            clean_token = token.replace(" ", "").replace("\t", "").replace("\n", "").lower()
            
            if clean_token in ["yes", "y"]:
                yes_prob += prob
            elif clean_token in ["no", "n"]:
                no_prob += prob
        
        return yes_prob, no_prob
    
    @staticmethod
    def calculate_strength_from_probabilities(yes_prob: float, no_prob: float) -> float:
        """Calculate strength as yes_prob / (yes_prob + no_prob), defaulting to 0.5."""
        total_prob = yes_prob + no_prob
        if total_prob > 0:
            return yes_prob / total_prob
        else:
            return 0.5