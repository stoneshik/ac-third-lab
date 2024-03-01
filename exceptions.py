from __future__ import annotations


class IsaError(Exception):
    pass


class IncorrectSpecialCharError(IsaError):
    def __init__(self, char: str) -> None:
        super().__init__(f"Incorrect special char \\{char}")


class MachineError(Exception):
    pass


class RegisterReadingError(MachineError):
    def __init__(self, num_register: str) -> None:
        super().__init__(f"wrong register number {num_register}")


class AluNotSupportedInstrError(MachineError):
    def __init__(self, opcode_hex: str) -> None:
        super().__init__(f"alu not supported instruction {opcode_hex}")


class IncorrectSelectorError(MachineError):
    def __init__(self, sel: str) -> None:
        super().__init__(f"incorrect selector: {sel}")


class InternalError(MachineError):
    def __init__(self) -> None:
        super().__init__("Internal error")


class MnemonicError(Exception):
    pass


class NotFoundInstructionError(MnemonicError):
    def __init__(self, instruction_opcode: str) -> None:
        super().__init__(f"Not found instruction - {instruction_opcode}")


class WrongAddressError(MnemonicError):
    def __init__(self) -> None:
        super().__init__("Wrong address")


class TranslatorError(Exception):
    pass


class IncorrectLiteralError(TranslatorError):
    def __init__(self) -> None:
        super().__init__("Incorrect literal - not number or string")


class NotFoundKeywordError(TranslatorError):
    def __init__(self) -> None:
        super().__init__("Not found keyword")


class IncorrectSecondArgumentError(TranslatorError):
    def __init__(self, arg: str) -> None:
        super().__init__(f"Incorrect second argument - {arg}")


class IncorrectArgumentForPrintError(TranslatorError):
    def __init__(self, arg: str) -> None:
        super().__init__(f"Incorrect argument for print - {arg}")


class ValueNotNumberOrNameVariableError(TranslatorError):
    def __init__(self, arg: str) -> None:
        super().__init__(f"Value is not number or name variable - {arg}")


class IncorrectArgumentInIfExpError(TranslatorError):
    def __init__(self, arg: str | list) -> None:
        super().__init__(f"Incorrect argument in if expression - {arg}")


class StringNotDefinedInMemoryError(TranslatorError):
    def __init__(self, arg: str) -> None:
        super().__init__(f"String not defined in data memory - {arg}")


class VariableNotDefinedInMemoryError(TranslatorError):
    def __init__(self, arg: str) -> None:
        super().__init__(f"Variable not defined in data memory - {arg}")


class IncorrectArgsPassedToFunctionError(TranslatorError):
    def __init__(self, name_function: str) -> None:
        super().__init__(f"Incorrect args passed to function {name_function}")
