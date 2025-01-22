class PromptTemplate:
    def __init__(self, system_prompt=None, user_messages=[], model_replies=[], tags={}):
        self.system_prompt = system_prompt
        self.user_messages = user_messages
        self.model_replies = model_replies

        self.system_prefix = tags.get("system_prefix", "")
        self.system_suffix = tags.get("system_suffix", "")
        self.user_prefix = tags.get("user_prefix", "")
        self.user_suffix = tags.get("user_suffix", "")
        self.model_prefix = tags.get("model_prefix", "")
        self.model_suffix = tags.get("model_suffix", "")

    def get_user_messages(self):
        return [x.strip() for x in self.user_messages]

    def get_model_replies(self):
        return [x.strip() for x in self.model_replies]

    def build_prompt(self):
        prompt_parts = []

        # Add system prompt if it exists
        if self.system_prompt:
            SYS = f"{self.system_prefix}\n{self.system_prompt}\n{self.system_suffix}\n"
            prompt_parts.append(SYS)

        # Add the conversation history
        for user_message, model_reply in zip(self.user_messages, self.model_replies):
            prompt_parts.append(f"{self.user_prefix} {user_message} {self.user_suffix}\n")
            prompt_parts.append(f"{self.model_prefix} {model_reply} {self.model_suffix}\n")

        # Add the last user message
        if len(self.user_messages) > len(self.model_replies):
            prompt_parts.append(f"{self.user_prefix} {self.user_messages[-1]} {self.user_suffix}\n")

        # **Append the model prefix to signal the assistant to respond**
        prompt_parts.append(f"{self.model_prefix} ")

        return "".join(prompt_parts)