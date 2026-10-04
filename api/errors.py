RETRYABLE_CATEGORIES = ("rate_limit", "transient", "network", "malformed")
MAX_MESSAGE_LENGTH = 200


class ProviderError(Exception):
    def __init__(self, provider, category, message, status=None, retry_after=None):
        self.provider = provider
        self.category = category
        self.message = str(message)[:MAX_MESSAGE_LENGTH]
        self.status = status
        self.retry_after = retry_after
        super().__init__(self.summary())

    @property
    def retryable(self):
        return self.category in RETRYABLE_CATEGORIES

    def summary(self):
        status = f" (HTTP {self.status})" if self.status else ""
        return f"{self.provider} {self.category} error{status}: {self.message}"
