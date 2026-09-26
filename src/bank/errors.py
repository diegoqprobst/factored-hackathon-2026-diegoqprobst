"""Typed errors raised by the mock bank services. `code` is stable and safe to show the orchestrator."""


class BankError(Exception):
    code = "bank_error"

    def __init__(self, message: str = ""):
        super().__init__(message or self.code)


class AuthError(BankError):
    code = "auth_failed"


class NoVerifiedChannel(AuthError):
    code = "no_verified_channel"


class AccountNotServiceable(AuthError):
    code = "account_not_serviceable"


class OtpInvalid(AuthError):
    code = "otp_invalid"


class OtpExpired(AuthError):
    code = "otp_expired"


class OtpLocked(AuthError):
    code = "otp_locked"


class SessionInvalid(AuthError):
    code = "session_invalid"


class SessionExpired(AuthError):
    code = "session_expired"


class NotFound(BankError):
    code = "not_found"


class InvalidProduct(BankError):
    code = "invalid_product"


class ConfirmationRequired(BankError):
    code = "confirmation_required"


class PolicyViolation(BankError):
    code = "policy_violation"

    def __init__(self, decision):
        super().__init__(decision.reason)
        self.decision = decision


class IdempotencyConflict(BankError):
    code = "idempotency_conflict"
