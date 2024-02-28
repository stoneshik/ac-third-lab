from __future__ import annotations

from exceptions import NotFoundInstructionError, WrongAddressError
from isa import (
    AddressCode,
    Opcode,
    address_by_hex_dict,
    number_to_hex,
    opcode_by_hex_dict,
)
from memory_config import InstrMemoryConfig


def get_register_string_from_hex_code(hex_code_string: str) -> str:
    assert hex_code_string[:1] in address_by_hex_dict.keys(), "Not register"
    return f"R{int(hex_code_string[2:], 16)}"


class MnemonicCreator:
    def __init__(
        self,
        instruction_words: list[bytes],
        vars_by_address: dict[str, str],
        number_consts_by_address: dict[str, int],
        string_consts_by_address: dict[str, str],
        functions_by_address: dict[str, str],
    ) -> None:
        self.__instruction_words: list[bytes] = instruction_words
        self.__vars_by_address: dict[str, str] = vars_by_address
        self.__number_consts_by_address: dict[str, int] = number_consts_by_address
        self.__string_consts_by_address: dict[str, str] = string_consts_by_address
        self.__functions_by_address: dict[str, str] = functions_by_address

    def save_mnemonic_in_file(self, output_file_name: str) -> None:
        strings: list[str] = self.__create_mnemonics()
        with open(output_file_name, "w", encoding="utf-8") as output_file:
            output_file.writelines(strings)

    def __create_mnemonics(self) -> list[str]:
        strings: list[str] = ["<address> - <hex_code> - <mnemonic>\n"]
        address_string: str = ""
        hex_code_string: str = ""
        current_instruction_opcode: Opcode | None = None
        first_arg_hex_word: str | None = None
        second_arg_hex_word: str | None = None
        counter: int = 0
        for i, instruction_word in enumerate(self.__instruction_words):
            hex_code: str = instruction_word.hex()
            opcode_hex: str = hex_code[:2]
            if opcode_hex in opcode_by_hex_dict.keys():
                if address_string != "" and hex_code_string != "":
                    if second_arg_hex_word is not None:
                        first_arg_hex_word, second_arg_hex_word = second_arg_hex_word, first_arg_hex_word
                    strings.append(
                        self.__create_mnemonic_string(
                            address_string,
                            hex_code_string,
                            current_instruction_opcode,
                            first_arg_hex_word,
                            second_arg_hex_word,
                        )
                    )
                    hex_code_string = ""
                    first_arg_hex_word = None
                    second_arg_hex_word = None
                address_string: str = number_to_hex(InstrMemoryConfig.address_hex_num, i)
                current_instruction_opcode = opcode_by_hex_dict[opcode_hex]
                counter = 0
            if counter == 0:
                hex_code_string += hex_code
            else:
                hex_code_string += f" {hex_code}"
            if counter == 1:
                first_arg_hex_word = hex_code
            elif counter == 2:
                second_arg_hex_word = hex_code
            counter += 1
        strings.append(
            self.__create_mnemonic_string(
                address_string, hex_code_string, current_instruction_opcode, first_arg_hex_word, second_arg_hex_word
            )
        )
        return strings

    def __create_mnemonic_string(
        self,
        address_string: str,
        hex_code_string: str,
        instruction_opcode: Opcode,
        first_arg_hex_word: str | None,
        second_arg_hex_word: str | None,
    ) -> str:
        opcode_string: str = instruction_opcode.name.lower()
        match instruction_opcode.value:
            case Opcode.NOP.value | Opcode.HALT.value | Opcode.RET.value:
                mnemonic_string = opcode_string
            case Opcode.CHAR.value | Opcode.INC.value | Opcode.DEC.value | Opcode.SLB.value | Opcode.SRB.value:
                reg_string: str = get_register_string_from_hex_code(first_arg_hex_word)
                mnemonic_string = f"{reg_string} <- {opcode_string} {reg_string}"
            case Opcode.IES.value:
                reg_string: str = get_register_string_from_hex_code(first_arg_hex_word)
                mnemonic_string = f"{opcode_string} {reg_string}"
            case (
                Opcode.ADD.value
                | Opcode.SUB.value
                | Opcode.MUL.value
                | (Opcode.DIV.value)
                | Opcode.AND.value
                | Opcode.OR.value
                | Opcode.MOD.value
            ):
                mnemonic_string: str = self.__two_args_math_mnemonic(
                    opcode_string, first_arg_hex_word, second_arg_hex_word
                )
            case Opcode.LOAD.value:
                mnemonic_string: str = self.__load_mnemonic(opcode_string, first_arg_hex_word, second_arg_hex_word)
            case Opcode.STORE.value:
                mnemonic_string: str = self.__store_mnemonic(opcode_string, first_arg_hex_word, second_arg_hex_word)
            case Opcode.POP.value | Opcode.READ.value:
                reg_string: str = get_register_string_from_hex_code(first_arg_hex_word)
                mnemonic_string = f"{reg_string} <- {opcode_string}"
            case Opcode.PUSH.value | Opcode.PRINT.value:
                reg_string: str = get_register_string_from_hex_code(first_arg_hex_word)
                mnemonic_string = f"{opcode_string} {reg_string}"
            case Opcode.CMP.value:
                mnemonic_string: str = self.__cmp_mnemonic(opcode_string, first_arg_hex_word, second_arg_hex_word)
            case Opcode.JMP.value | Opcode.JZ.value | Opcode.JNZ.value:
                mnemonic_string: str = self.__jmp_mnemonic(opcode_string, first_arg_hex_word)
            case Opcode.CALL.value:
                function_address: str = first_arg_hex_word[1:]
                assert (
                    function_address in self.__functions_by_address.keys()
                ), f"Not found function address - {function_address}"
                function_name: str = self.__functions_by_address[function_address]
                mnemonic_string = f"{opcode_string} {function_name} | {function_name} -> {function_address}"
            case _:
                raise NotFoundInstructionError(instruction_opcode.value)
        if address_string in self.__functions_by_address.keys():
            name_function: str = self.__functions_by_address[address_string]
            return f"{address_string} - {hex_code_string} - {mnemonic_string} | aboba {name_function}\n"
        return f"{address_string} - {hex_code_string} - {mnemonic_string}\n"

    def __two_args_math_mnemonic(self, opcode_string: str, first_arg_hex_word: str, second_arg_hex_word: str) -> str:
        reg_string: str = get_register_string_from_hex_code(first_arg_hex_word)
        var_data: tuple[str, str | None, str | None] = self.__get_var_string_from_hex_code(second_arg_hex_word)
        var_address: str = var_data[0]
        var_name: str | None = var_data[1]
        var_value: str | None = var_data[2]
        if var_name is None and var_value is None:
            mnemonic_string = f"{reg_string} <- {opcode_string} {reg_string}, {var_address}"
        elif var_name is None and var_value is not None:
            mnemonic_string = (
                f"{reg_string} <- {opcode_string} {reg_string}, {var_address} | {var_address} -> {var_value}"
            )
        else:
            mnemonic_string = f"{reg_string} <- {opcode_string} {reg_string}, {var_name} | {var_name} -> {var_address}"
        return mnemonic_string

    def __load_mnemonic(self, opcode_string: str, first_arg_hex_word: str, second_arg_hex_word: str) -> str:
        reg_string: str = get_register_string_from_hex_code(first_arg_hex_word)
        var_data: tuple[str, str | None, str | None] = self.__get_var_string_from_hex_code(second_arg_hex_word)
        var_address: str = var_data[0]
        var_name: str | None = var_data[1]
        var_value: str | None = var_data[2]
        if var_name is None and var_value is None:
            mnemonic_string = f"{reg_string} <- {opcode_string} {var_address}"
        elif var_name is None and var_value is not None:
            mnemonic_string = f"{reg_string} <- {opcode_string} {var_address} | {var_address} -> {var_value}"
        else:
            mnemonic_string = f"{reg_string} <- {opcode_string} {var_name} | {var_name} -> {var_address}"
        return mnemonic_string

    def __store_mnemonic(self, opcode_string: str, first_arg_hex_word: str, second_arg_hex_word: str) -> str:
        reg_string: str = get_register_string_from_hex_code(first_arg_hex_word)
        var_data: tuple[str, str | None, str | None] = self.__get_var_string_from_hex_code(second_arg_hex_word)
        var_address: str = var_data[0]
        var_name: str | None = var_data[1]
        var_value: str | None = var_data[2]
        if var_name is None and var_value is None:
            mnemonic_string = f"{var_address} <- {opcode_string} {reg_string}"
        elif var_name is None and var_value is not None:
            mnemonic_string = f"{var_address} <- {opcode_string} {reg_string} | {var_address} -> {var_value}"
        else:
            mnemonic_string = f"{var_name} <- {opcode_string} {reg_string} | {var_name} -> {var_address}"
        return mnemonic_string

    def __cmp_mnemonic(self, opcode_string: str, first_arg_hex_word: str, second_arg_hex_word: str) -> str:
        reg_string: str = get_register_string_from_hex_code(first_arg_hex_word)
        var_data: tuple[str, str | None, str | None] = self.__get_var_string_from_hex_code(second_arg_hex_word)
        var_address: str = var_data[0]
        var_name: str | None = var_data[1]
        var_value: str | None = var_data[2]
        if var_name is None and var_value is None:
            mnemonic_string = f"{opcode_string} {reg_string}, {var_address}"
        elif var_name is None and var_value is not None:
            mnemonic_string = f"{opcode_string} {reg_string}, {var_address} | {var_address} -> {var_value}"
        else:
            mnemonic_string = f"{opcode_string} {reg_string}, {var_name} | {var_name} -> {var_address}"
        return mnemonic_string

    def __jmp_mnemonic(self, opcode_string: str, first_arg_hex_word: str) -> str:
        var_data: tuple[str, str | None, str | None] = self.__get_var_string_from_hex_code(first_arg_hex_word)
        var_address: str = var_data[0]
        var_name: str | None = var_data[1]
        if var_name is None:
            mnemonic_string = f"{opcode_string} {var_address}"
        else:
            mnemonic_string = f"{opcode_string} {var_name} | {var_name} -> {var_address}"
        return mnemonic_string

    def __get_var_string_from_hex_code(self, hex_code_string: str) -> tuple[str, str | None, str | None]:
        assert hex_code_string[:1] in address_by_hex_dict.keys(), "Not address"
        address_code: AddressCode = address_by_hex_dict[hex_code_string[:1]]
        address_hex_string: str = hex_code_string[1:]
        is_var: bool = address_hex_string in self.__vars_by_address.keys()
        var_value: str | None = None
        match address_code.value:
            case AddressCode.DIRECT_ABS.value:
                var_address = "$" + address_hex_string
                if is_var:
                    var_name = "$" + self.__vars_by_address[address_hex_string]
                else:
                    if address_hex_string in self.__number_consts_by_address.keys():
                        var_value = str(self.__number_consts_by_address[address_hex_string])
                    elif address_hex_string in self.__string_consts_by_address.keys():
                        var_value = f'"{self.__string_consts_by_address[address_hex_string]}"'
                    var_name = None
            case AddressCode.DIRECT_OFFSET.value:
                var_address = f"$({address_hex_string})"
                if is_var:
                    var_name = f"${(self.__vars_by_address[address_hex_string])}"
                else:
                    var_name = None
            case AddressCode.INDIRECT_SP.value:
                var_address = "&" + address_hex_string
                if is_var:
                    var_name = "&" + self.__vars_by_address[address_hex_string]
                else:
                    var_name = None
            case _:
                raise WrongAddressError()
        return var_address, var_name, var_value
