from __future__ import annotations

import re

from isa import KeyWord
from pyparsing import nestedExpr


# Парсинг и проверка кода
class LiteralPatterns:
    NAME_VAR: re.Pattern = re.compile(r"[a-zA-Z_]+")
    NAME_FUNCTION: re.Pattern = re.compile(r"[a-zA-Z_]+")
    NUMBER: re.Pattern = re.compile(r"-?\d+")
    STRING: re.Pattern = re.compile(r'".*"')

    @classmethod
    def is_name_var(cls, checking_string: str) -> bool:
        return re.fullmatch(cls.NAME_VAR, checking_string) is not None

    @classmethod
    def is_name_function(cls, checking_string: str) -> bool:
        return re.fullmatch(cls.NAME_FUNCTION, checking_string) is not None

    @classmethod
    def is_number(cls, checking_string: str) -> bool:
        return re.fullmatch(cls.NUMBER, checking_string) is not None

    @classmethod
    def is_string(cls, checking_string: str) -> bool:
        return re.fullmatch(cls.STRING, checking_string) is not None


class Checker:
    def __init__(self, source: str, parsed_source: list[str]) -> None:
        self.__source: str = source
        self.__parsed_source: list[str] = parsed_source
        self.__is_valid: bool = self.__checking_brackets() and self.__checking_nesting() and self.__checking_semantic()

    @property
    def is_valid(self) -> bool:
        return self.__is_valid

    def __checking_brackets(self) -> bool:
        return self.__source.count("(") == self.__source.count(")")

    def __checking_nesting(self) -> bool:
        pattern_for_var: re.Pattern = re.compile(r"\([^()]*\(var")
        pattern_for_aboba: re.Pattern = re.compile(r"\([^()]*\(aboba")
        assert re.search(pattern_for_var, self.__source) is None, f"Nested expression {KeyWord.VAR.value}"
        assert re.search(pattern_for_aboba, self.__source) is None, f"Nested expression {KeyWord.FUNCTION.value}"
        return True

    def __checking_semantic(self) -> bool:
        is_start: bool = True
        exp_var_last_index: int = 0
        for i, exp in enumerate(self.__parsed_source):
            if exp[0] != KeyWord.VAR.value:
                is_start = False
            else:
                assert is_start, f"{KeyWord.VAR.value} not define at the beginning"
                exp_var_last_index = i
        is_start = True
        exp_aboba_first_index: int = exp_var_last_index if exp_var_last_index == 0 else exp_var_last_index + 1
        for i, exp in enumerate(self.__parsed_source[exp_aboba_first_index:]):
            if exp[0] != KeyWord.FUNCTION.value:
                is_start = False
            else:
                assert is_start, f"{KeyWord.FUNCTION.value} not define at the beginning, after {KeyWord.VAR.value}"
        return True


def parse(source: str) -> list[str]:
    return nestedExpr("(", ")").parseString(f"({source})").asList()[0]


def parsed_and_check_source_file(input_file: str) -> list[str]:
    with open(input_file, encoding="utf-8") as source_file:
        source: str = source_file.read()
    parsed_source: list[str] = parse(source)
    checker: Checker = Checker(source, parsed_source)
    assert checker.is_valid, "Error in AbobaLisp code"
    return parsed_source
