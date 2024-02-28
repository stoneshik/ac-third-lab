#!/usr/bin/python3
import sys
import math

from memory_config import InstrMemoryConfig, DataMemoryConfig
from checker import LiteralPatterns, parsed_and_check_source_file
from isa import (
    KeyWord,
    Opcode,
    all_keywords,
    number_to_hex,
    string_to_hex_list,
    get_opcode_word,
    get_direct_abs_address,
    get_direct_offset_address,
    get_indirect_sp_address,
    get_direct_reg_address
)
from mnemonic import MnemonicCreator


def traverse(o, tree_types=(list,)):
    """
    Перебор всех значений списка списков
    """
    if isinstance(o, tree_types):
        for value in o:
            for sub_value in traverse(value, tree_types):
                yield sub_value
    else:
        yield o


class Translator:
    LOW_BYTE_FILTER_NUM: int = 255  # число 0000 00FF
    ZERO_ASCII_NUM: int = 48  # номер ascii символа 0
    ZERO_LAST_WORD: int = 808464432  # если из переменной выводится 0, то последнее слово равно 3030 3030

    def __init__(self, parsed_source: list[str]) -> None:
        self.__parsed_source: list[str] = parsed_source
        self.__instruction_words: list[bytes] = [
            bytes.fromhex(get_opcode_word(Opcode.JMP)),
            bytes.fromhex('a000')
        ]
        zero_word_data_memory: str = '0' * DataMemoryConfig.word_hex_num
        self.__data_words: list[bytes] = [bytes.fromhex(zero_word_data_memory)] * DataMemoryConfig.size
        # Индекс ближайшей свободной ячейки в именованной памяти, 3 т.к. 0 и 1 - порты ввода и вывода,
        # 2 значение для Heap Counter
        self.__var_counter: int = 3
        self.__heap_counter: int = DataMemoryConfig.named_memory_size
        self.__vars: dict[str, str] = {}  # второе значение это адрес в памяти
        self.__vars_by_address: dict[str, str] = {}
        self.__number_consts: dict[int, str] = {}  # второе значение это адрес в памяти
        self.__string_consts: dict[str, str] = {}
        self.__number_consts_by_address: dict[str, int] = {}  # обратные словари
        self.__string_consts_by_address: dict[str, str] = {}
        self.__functions: dict[str, str] = {}
        self.__functions_by_address: dict[str, str] = {}
        self.__buffer_address: str = self.__create_buffer(1)
        self.__offset_params: int = 0

    @property
    def instruction_words(self) -> list[bytes]:
        return self.__instruction_words

    @property
    def vars_by_address(self) -> dict[str, str]:
        return self.__vars_by_address

    @property
    def number_consts_by_address(self) -> dict[str, int]:
        return self.__number_consts_by_address

    @property
    def string_consts_by_address(self) -> dict[str, str]:
        return self.__string_consts_by_address

    @property
    def functions_by_address(self) -> dict[str, str]:
        return self.__functions_by_address

    def translate(self) -> None:
        self.__create_vars()
        self.__create_consts()
        self.__create_functions()
        self.__create_expressions()

    def __create_buffer(self, buffer_size: int) -> str:
        buffer_address: str = number_to_hex(InstrMemoryConfig.address_hex_num, self.__heap_counter)
        self.__heap_counter += buffer_size
        assert self.__var_counter < DataMemoryConfig.size, "Space in the heap is over"
        return buffer_address

    def save_instruction_words_in_file(self, output_file_name: str) -> None:
        with open(output_file_name, 'wb') as output_file:
            for word in self.__instruction_words:
                output_file.write(word)

    def save_data_memory_in_file(self, output_file_name: str) -> None:
        with open(output_file_name, 'wb') as output_file:
            for word in self.__data_words:
                output_file.write(word)

    # Создание переменных
    def __create_vars(self):
        for exp in self.__parsed_source:
            if exp[0] != KeyWord.VAR.value:
                return
            assert len(exp) == 3, "Incorrect amount arguments of expression"
            var_name: str = exp[1]
            value: str = exp[2]
            assert LiteralPatterns.is_name_var(var_name), "Variable name is incorrectly specified"
            if LiteralPatterns.is_number(value):
                value_number: int = int(value)
                self.__add_var_num(var_name, value_number)
            elif LiteralPatterns.is_string(value):
                string_value: str = value[1:-1]  # убираем ""
                self.__add_var_string(var_name, string_value)
            else:
                raise Exception("Incorrect literal - not number or string")

    def __add_var_num(self, var_name: str, value: int) -> None:
        assert self.__var_counter < DataMemoryConfig.named_memory_size, \
            f"Number of variables exceeded - {self.__var_counter}"
        # добавление значения переменной
        hex_value: str = number_to_hex(DataMemoryConfig.word_hex_num, value)
        assert len(hex_value) == DataMemoryConfig.word_hex_num, f"Incorrect value length - {hex_value}"
        self.__data_words[self.__var_counter] = bytes.fromhex(hex_value)
        # добавление адреса переменной в словарь
        hex_address: str = number_to_hex(DataMemoryConfig.address_hex_num, self.__var_counter)
        assert len(hex_address) == DataMemoryConfig.address_hex_num, "Incorrect address length"
        self.__vars[var_name] = hex_address
        self.__vars_by_address[hex_address] = var_name
        # обновляем счетчик
        self.__var_counter += 1

    def __add_var_string(self, var_name: str, value: str) -> None:
        assert self.__var_counter < DataMemoryConfig.named_memory_size, "Number of variables exceeded"
        assert (math.floor(len(value) / DataMemoryConfig.word_size) + self.__heap_counter <
                DataMemoryConfig.size), "Space in the heap is over"
        # добавление значения переменной (равно адресу начала строки)
        hex_value: str = number_to_hex(DataMemoryConfig.word_hex_num, self.__heap_counter)
        assert len(hex_value) == DataMemoryConfig.word_hex_num, "Incorrect value length"
        self.__data_words[self.__var_counter] = bytes.fromhex(hex_value)
        # добавление адреса переменной в словарь
        hex_address: str = number_to_hex(DataMemoryConfig.address_hex_num, self.__var_counter)
        assert len(hex_address) == DataMemoryConfig.address_hex_num, "Incorrect address length"
        self.__vars[var_name] = hex_address
        self.__vars_by_address[hex_address] = var_name
        # добавление строки в кучу
        words_storing_string: list[str] = string_to_hex_list(
            DataMemoryConfig.word_hex_num,
            DataMemoryConfig.word_size,
            value
        )
        for word in words_storing_string:
            assert len(word) == DataMemoryConfig.word_hex_num, "Incorrect value length"
            self.__data_words[self.__heap_counter] = bytes.fromhex(word)
            self.__heap_counter += 1
        if words_storing_string[-1][:2] != '00':  # Если в последнем слове последний символ '\0'
            self.__create_buffer(1)  # то добавляем пустое слово
        # обновляем счетчик переменной
        self.__var_counter += 1

    # Создание всех констант, используемых программой
    def __create_consts(self) -> None:
        source_for_consts: list[str | list] = [
            exp for exp in self.__parsed_source if exp[0] != KeyWord.VAR.value
        ]
        self.__create_number_const(0)
        self.__create_number_const(10)
        self.__create_number_const(DataMemoryConfig.word_size)
        self.__create_number_const(self.LOW_BYTE_FILTER_NUM)
        self.__create_number_const(self.ZERO_ASCII_NUM)
        self.__create_number_const(self.ZERO_LAST_WORD)
        for value in traverse(source_for_consts):
            if LiteralPatterns.is_number(value):
                value_number: int = int(value)
                if value_number in self.__number_consts.keys():
                    continue
                self.__create_number_const(value_number)
            elif LiteralPatterns.is_string(value):
                string_value: str = value[1:-1]  # убираем ""
                if string_value in self.__string_consts.keys():
                    continue
                self.__create_string_const(string_value)

    def __create_number_const(self, value: int) -> None:
        assert self.__heap_counter < DataMemoryConfig.size, "Space in the heap is over"
        # добавление значения переменной
        hex_value: str = number_to_hex(DataMemoryConfig.word_hex_num, value)
        assert len(hex_value) == DataMemoryConfig.word_hex_num, "Incorrect value length"
        self.__data_words[self.__heap_counter] = bytes.fromhex(hex_value)
        # добавление адреса переменной в словарь
        hex_address: str = number_to_hex(DataMemoryConfig.address_hex_num, self.__heap_counter)
        assert len(hex_address) == DataMemoryConfig.address_hex_num, "Incorrect address length"
        self.__number_consts[value] = hex_address
        self.__number_consts_by_address[hex_address] = value
        # обновляем счетчик
        self.__heap_counter += 1

    def __create_string_const(self, value: str) -> None:
        assert (math.floor(len(value) / DataMemoryConfig.word_size) + self.__heap_counter <
                DataMemoryConfig.size), "Space in the heap is over"
        # добавление адреса переменной в словарь
        hex_address: str = number_to_hex(DataMemoryConfig.address_hex_num, self.__heap_counter)
        assert len(hex_address) == DataMemoryConfig.address_hex_num, "Incorrect address length"
        self.__string_consts[value] = hex_address
        self.__string_consts_by_address[hex_address] = value
        # добавление строки в кучу
        words_storing_string: list[str] = string_to_hex_list(
            DataMemoryConfig.word_hex_num,
            DataMemoryConfig.word_size,
            value
        )
        for word in words_storing_string:
            assert len(word) == DataMemoryConfig.word_hex_num, "Incorrect value length"
            self.__data_words[self.__heap_counter] = bytes.fromhex(word)
            self.__heap_counter += 1
        if words_storing_string[-1][:2] != '00':  # Если в последнем слове последний символ '\0'
            self.__create_buffer(1)  # то добавляем пустое слово

    # Создание функций
    def __create_functions(self) -> None:
        source_for_functions: list[str | list] = [
            exp for exp in self.__parsed_source if exp[0] == KeyWord.FUNCTION.value
        ]
        for expression in source_for_functions:
            assert len(expression) == 4, f"Incorrect amount arguments of expression - {len(expression)}"
            name_function: str = expression[1]
            assert LiteralPatterns.is_name_function(name_function), f"Incorrect name function - {name_function}"
            assert name_function not in self.__vars.keys(), f"Function named as variable - {name_function}"
            assert isinstance(expression[2], list), f"Incorrect names of function parameters - {name_function}"
            param_names: tuple[str] = tuple(expression[2])
            assert len(param_names) == len(frozenset(param_names)), \
                f"Repeated names of function parameters - {name_function}"
            for param_name in param_names:
                assert LiteralPatterns.is_name_var(param_name), \
                    f"Incorrect names of function parameters - {param_name}"
                assert param_name not in self.__functions.keys() and param_name not in self.__vars.keys(), \
                    f"Parameter named as function or variable {param_name}"
            assert isinstance(expression[3], list), f"Incorrect nested expression in function - {name_function}"
            nested_expression: list[str | list] = expression[3]
            function_address: str = number_to_hex(InstrMemoryConfig.address_hex_num, len(self.__instruction_words))
            self.__functions[name_function] = function_address
            self.__functions_by_address[function_address] = name_function
            self.__offset_params = 1  # 1 т.к. 0-й всегда адрес возврата
            self.__create_expression(nested_expression, param_names)
            self.__add_pop_instruction('0')
            self.__add_zero_args_instruction(Opcode.RET)
            self.__offset_params -= 1
        self.__instruction_words[1] = bytes.fromhex(get_direct_abs_address(
            number_to_hex(InstrMemoryConfig.address_hex_num, len(self.__instruction_words))
        ))

    # Создание остальных выражений
    def __create_expressions(self) -> None:
        source_for_expressions: list[str | list] = [
            exp for exp in self.__parsed_source if exp[0] != KeyWord.VAR.value and exp[0] != KeyWord.FUNCTION.value
        ]
        for expression in source_for_expressions:
            self.__create_expression(expression, None)
            self.__add_pop_instruction('0')
        self.__add_zero_args_instruction(Opcode.HALT)
        self.__data_words[2] = bytes.fromhex(
            number_to_hex(DataMemoryConfig.word_hex_num, self.__heap_counter)
        )

    # Методы для создания выражений
    def __create_expression(self, exp: list[str | list], param_names: tuple[str] | None) -> None:
        key_word: str = exp[0]
        assert key_word != KeyWord.VAR and key_word != KeyWord.FUNCTION, f"Nested expression - {key_word}"
        assert key_word in all_keywords, f"Unknown keyword - {key_word}"
        match key_word:
            case KeyWord.PRINT_NUMBER.value:
                assert len(exp) == 2, f"Incorrect amount arguments of expression - {len(exp)}"
                first_raw_arg: str = exp[1]
                self.__create_machine_code_for_print_number_expression(first_raw_arg, param_names)
            case KeyWord.PRINT.value:
                assert len(exp) == 2, f"Incorrect amount arguments of expression - {len(exp)}"
                raw_arg: str = exp[1]
                self.__create_machine_code_for_print_expression(raw_arg, param_names)
            case KeyWord.READ.value:
                assert len(exp) == 3, f"Incorrect amount arguments of expression - {len(exp)}"
                name_var: str = exp[1]
                assert LiteralPatterns.is_name_var(name_var), \
                    f"Variable name is incorrectly specified - {name_var}"
                second_raw_arg: str = exp[2]
                assert LiteralPatterns.is_number(second_raw_arg), f"Second arg not number - {second_raw_arg}"
                buffer_size: int = int(second_raw_arg)
                self.__create_machine_code_for_read_expression(name_var, buffer_size, param_names)
            case KeyWord.ADD.value | KeyWord.SUB.value | KeyWord.MUL.value | KeyWord.DIV.value:
                assert len(exp) == 3, f"Incorrect amount arguments of expression - {len(exp)}"
                first_raw_arg: str | list[str | list] = exp[1]
                second_raw_arg: str | list[str | list] = exp[2]
                self.__create_machine_code_for_binary_math_expression(
                    key_word,
                    first_raw_arg,
                    second_raw_arg,
                    param_names
                )
            case KeyWord.IF.value:
                assert len(exp) == 4, f"Incorrect amount arguments of expression - {len(exp)}"
                self.__create_machine_code_for_if_expression(exp[1], exp[2], exp[3], param_names)
            case KeyWord.SET.value:
                assert len(exp) == 3, f"Incorrect amount arguments of expression - {len(exp)}"
                name_var: str = exp[1]
                assert LiteralPatterns.is_name_var(name_var), \
                    f"Variable name is incorrectly specified - {name_var}"
                second_raw_arg: str = exp[2]
                self.__create_machine_code_for_set_expression(name_var, second_raw_arg, param_names)
            case KeyWord.CALL.value:
                assert len(exp) == 3, f"Incorrect amount arguments of expression - {len(exp)}"
                name_function: str = exp[1]
                assert LiteralPatterns.is_name_function(name_function), \
                    f"Function name is incorrectly specified - {name_function}"
                raw_args: list[str] = exp[2]
                self.__create_machine_code_for_call_expression(name_function, raw_args)
            case KeyWord.ITER.value:
                assert len(exp) == 4, f"Incorrect amount arguments of expression - {len(exp)}"
                name_iter_var: str = exp[1]
                assert LiteralPatterns.is_name_var(name_iter_var), f"Incorrect iter variable name - {name_iter_var}"
                max_iter_value_raw: str = exp[2]
                assert LiteralPatterns.is_number(max_iter_value_raw), \
                    f"Max iter value is not number - {max_iter_value_raw}"
                max_iter_value: int = int(max_iter_value_raw)
                nested_expression: list[str | list] = exp[3]
                assert isinstance(nested_expression, list), f"Incorrect nested expression"
                self.__create_machine_code_for_iter_expression(
                    name_iter_var,
                    max_iter_value,
                    nested_expression,
                    param_names
                )
            case _:
                raise Exception("Not found keyword")

    def __create_machine_code_for_print_number_expression(
            self,
            first_raw_arg: str,
            param_names: tuple[str] | None) -> None:
        if LiteralPatterns.is_number(first_raw_arg):
            self.__create_string_const(first_raw_arg)
            string_address: str = self.__string_consts[first_raw_arg]
            pointer_value: int = int(string_address, 16)
            direct_pointer_address: str = self.__create_pointer_value(pointer_value)
            self.__create_machine_code_for_print_string_by_pointer_address(direct_pointer_address)
        elif LiteralPatterns.is_name_var(first_raw_arg):
            number_address: str = self.__get_variable_address(param_names, first_raw_arg)
            buffer_size: int = 3  # макс uint32 число состоит из 10 символов
            pointer_value: int = self.__heap_counter + 1
            direct_pointer_address: str = self.__create_pointer_value(pointer_value)
            self.__create_buffer(buffer_size)
            end_string_address_value: int = self.__heap_counter
            direct_end_string_pointer_address: str = self.__create_pointer_value(end_string_address_value)
            self.__create_machine_code_for_convert_number_to_string(number_address, direct_end_string_pointer_address)
            self.__create_machine_code_for_print_converted_number_to_string_by_pointer_address(direct_pointer_address)
        else:
            raise Exception(f"Incorrect argument for print - {first_raw_arg}")

    def __create_machine_code_for_convert_number_to_string(
            self,
            number_address: str,
            direct_end_string_pointer_address: str) -> None:
        # Запись идет в обратном порядке, в начале получившейся строки могут быть нули
        direct_buffer_address: str = get_direct_abs_address(self.__buffer_address)
        offset_buffer_address: str = get_direct_offset_address(self.__buffer_address)
        address_const_0: str = get_direct_abs_address(self.__number_consts[0])
        address_const_10: str = get_direct_abs_address(self.__number_consts[10])
        self.__add_binary_instruction(Opcode.LOAD, '0', direct_end_string_pointer_address)
        self.__add_binary_instruction(Opcode.LOAD, '1', number_address)
        jnz_argument_address: str = number_to_hex(InstrMemoryConfig.address_hex_num, len(self.__instruction_words))
        # Начало цикла
        self.__add_binary_instruction(Opcode.LOAD, '3', address_const_0)
        for _ in range(DataMemoryConfig.word_size):
            self.__add_binary_instruction(Opcode.STORE, '1', direct_buffer_address)
            self.__add_binary_instruction(Opcode.LOAD, '2', direct_buffer_address)
            self.__add_binary_instruction(Opcode.MOD, '2', address_const_10)
            self.__add_unary_instruction(Opcode.CHAR, '2')
            self.__add_binary_instruction(Opcode.STORE, '2', direct_buffer_address)
            self.__add_unary_instruction(Opcode.SLB, '3')
            self.__add_binary_instruction(Opcode.ADD, '3', direct_buffer_address)
            self.__add_binary_instruction(Opcode.DIV, '1', address_const_10)
        self.__add_binary_instruction(Opcode.STORE, '0', direct_buffer_address)
        self.__add_binary_instruction(Opcode.STORE, '3', offset_buffer_address)
        self.__add_unary_instruction(Opcode.DEC, '0')
        self.__add_binary_instruction(Opcode.CMP, '1', address_const_0)
        self.__add_unary_instruction_with_operand_address(Opcode.JNZ, jnz_argument_address)

    def __create_machine_code_for_print_expression(self, raw_arg: str, param_names: tuple[str]) -> None:
        if LiteralPatterns.is_string(raw_arg):
            arg: str = raw_arg[1:-1]  # убираем ""
            string_address: str = self.__string_consts[arg]
            pointer_value: int = int(string_address, 16)
            pointer_address: str = self.__create_pointer_value(pointer_value)
        elif LiteralPatterns.is_name_var(raw_arg):
            pointer_address: str = self.__get_variable_address(param_names, raw_arg)
        else:
            raise Exception(f"Incorrect argument for print - {raw_arg}")
        self.__create_machine_code_for_print_string_by_pointer_address(pointer_address)

    def __create_machine_code_for_print_string_by_pointer_address(self, pointer_address: str) -> None:
        direct_buffer_address: str = get_direct_abs_address(self.__buffer_address)
        offset_buffer_address: str = get_direct_offset_address(self.__buffer_address)
        address_const_0: str = get_direct_abs_address(self.__number_consts[0])
        self.__add_binary_instruction(Opcode.LOAD, '2', pointer_address)
        self.__add_binary_instruction(Opcode.STORE, '2', direct_buffer_address)
        # Начало цикла записи
        jnz_argument_address: str = number_to_hex(InstrMemoryConfig.address_hex_num, len(self.__instruction_words))
        self.__add_binary_instruction(Opcode.LOAD, '3', offset_buffer_address)
        for _ in range(DataMemoryConfig.word_size):
            self.__add_unary_instruction(Opcode.PRINT, '3')
            self.__add_unary_instruction(Opcode.SRB, '3')
        self.__add_binary_instruction(Opcode.LOAD, '3', offset_buffer_address)
        self.__add_unary_instruction(Opcode.IES, '3')
        self.__add_unary_instruction(Opcode.INC, '2')
        self.__add_binary_instruction(Opcode.STORE, '2', direct_buffer_address)
        self.__add_unary_instruction_with_operand_address(Opcode.JNZ, jnz_argument_address)
        # Конец цикла записи
        self.__add_binary_instruction(Opcode.LOAD, '0', address_const_0)
        self.__add_push_instruction('0')

    def __create_machine_code_for_print_converted_number_to_string_by_pointer_address(
            self,
            pointer_address: str) -> None:
        direct_buffer_address: str = get_direct_abs_address(self.__buffer_address)
        offset_buffer_address: str = get_direct_offset_address(self.__buffer_address)
        address_const_0: str = get_direct_abs_address(self.__number_consts[0])
        address_const_zero_last_word: str = get_direct_abs_address(self.__number_consts[self.ZERO_LAST_WORD])
        address_const_data_word_size: str = get_direct_abs_address(self.__number_consts[DataMemoryConfig.word_size])
        address_const_low_byte_filter_num: str = get_direct_abs_address(self.__number_consts[self.LOW_BYTE_FILTER_NUM])
        address_const_low_zero_ascii_num: str = get_direct_abs_address(self.__number_consts[self.ZERO_ASCII_NUM])
        # Пропуск пустых слов
        self.__add_binary_instruction(Opcode.LOAD, '1', address_const_data_word_size)
        self.__add_binary_instruction(Opcode.LOAD, '2', pointer_address)
        self.__add_unary_instruction(Opcode.DEC, '2')
        jz_argument_address: str = number_to_hex(InstrMemoryConfig.address_hex_num, len(self.__instruction_words))
        self.__add_unary_instruction(Opcode.INC, '2')
        self.__add_binary_instruction(Opcode.STORE, '2', direct_buffer_address)
        self.__add_binary_instruction(Opcode.LOAD, '3', offset_buffer_address)
        self.__add_binary_instruction(Opcode.CMP, '3', address_const_0)
        self.__add_unary_instruction(Opcode.DEC, '1')
        self.__add_unary_instruction_with_operand_address(Opcode.JZ, jz_argument_address)
        self.__add_binary_instruction(Opcode.CMP, '1', address_const_0)
        self.__add_unary_instruction_with_operand_address(Opcode.JNZ, '000')  # Если это последнее
        jz_address_start_zero_skip_index: int = len(self.__instruction_words) - 1  # слово то проверяем
        self.__add_binary_instruction(Opcode.CMP, '3', address_const_zero_last_word)  # является ли нулем
        self.__add_unary_instruction_with_operand_address(Opcode.JNZ, '000')
        jnz_address_start_print_index_from_start: int = len(self.__instruction_words) - 1
        self.__add_unary_instruction(Opcode.PRINT, '3')
        self.__add_unary_instruction_with_operand_address(Opcode.JMP, '000')
        jmp_address_end_index: int = len(self.__instruction_words) - 1
        # пропускаем возможные нули перед числом по типу 00121212
        self.__update_arg_for_jmp_instruction(jz_address_start_zero_skip_index)
        self.__add_binary_instruction(Opcode.LOAD, '1', address_const_data_word_size)
        self.__add_binary_instruction(Opcode.STORE, '3', direct_buffer_address)
        jnz_start_zero_skip_address: str = number_to_hex(
            InstrMemoryConfig.address_hex_num, len(self.__instruction_words)
        )
        self.__add_binary_instruction(Opcode.LOAD, '0', direct_buffer_address)
        self.__add_binary_instruction(Opcode.AND, '0', address_const_low_byte_filter_num)
        self.__add_binary_instruction(Opcode.CMP, '0', address_const_low_zero_ascii_num)
        self.__add_unary_instruction_with_operand_address(Opcode.JNZ, '000')
        jnz_address_print_last_chars_index: int = len(self.__instruction_words) - 1
        self.__add_unary_instruction(Opcode.SRB, '3')
        self.__add_binary_instruction(Opcode.STORE, '3', direct_buffer_address)
        self.__add_unary_instruction(Opcode.DEC, '1')
        self.__add_binary_instruction(Opcode.CMP, '1', address_const_0)
        self.__add_unary_instruction_with_operand_address(Opcode.JNZ, jnz_start_zero_skip_address)
        # Проверка является ли выведенное слово последним
        self.__add_binary_instruction(Opcode.STORE, '2', direct_buffer_address)
        self.__add_binary_instruction(Opcode.LOAD, '0', offset_buffer_address)
        self.__add_unary_instruction(Opcode.IES, '0')
        self.__add_unary_instruction_with_operand_address(Opcode.JZ, '000')
        jz_address_end_index: int = len(self.__instruction_words) - 1
        self.__add_unary_instruction_with_operand_address(Opcode.JNZ, '000')
        jnz_address_start_print_index: int = len(self.__instruction_words) - 1
        # Дозапись оставшихся символов в слове
        self.__update_arg_for_jmp_instruction(jnz_address_print_last_chars_index)
        jnz_start_zero_print_last_chars_address: str = number_to_hex(
            InstrMemoryConfig.address_hex_num, len(self.__instruction_words)
        )
        self.__add_unary_instruction(Opcode.PRINT, '3')
        self.__add_unary_instruction(Opcode.SRB, '3')
        self.__add_unary_instruction(Opcode.DEC, '1')
        self.__add_binary_instruction(Opcode.CMP, '1', address_const_0)
        self.__add_unary_instruction_with_operand_address(Opcode.JNZ, jnz_start_zero_print_last_chars_address)
        # Проверка является ли выведенное слово последним
        self.__add_binary_instruction(Opcode.STORE, '2', direct_buffer_address)
        self.__add_binary_instruction(Opcode.LOAD, '0', offset_buffer_address)
        self.__add_unary_instruction(Opcode.IES, '0')
        self.__add_unary_instruction_with_operand_address(Opcode.JZ, '000')
        jz_address_end_index_second: int = len(self.__instruction_words) - 1
        # Начало основного цикла записи
        self.__update_arg_for_jmp_instruction(jnz_address_start_print_index)
        self.__add_unary_instruction(Opcode.INC, '2')
        jnz_argument_address: str = number_to_hex(InstrMemoryConfig.address_hex_num, len(self.__instruction_words))
        self.__update_arg_for_jmp_instruction(jnz_address_start_print_index_from_start)
        self.__add_binary_instruction(Opcode.STORE, '2', direct_buffer_address)
        self.__add_binary_instruction(Opcode.LOAD, '3', offset_buffer_address)
        for _ in range(DataMemoryConfig.word_size):
            self.__add_unary_instruction(Opcode.PRINT, '3')
            self.__add_unary_instruction(Opcode.SRB, '3')
        self.__add_binary_instruction(Opcode.LOAD, '3', offset_buffer_address)
        self.__add_unary_instruction(Opcode.IES, '3')
        self.__add_unary_instruction(Opcode.INC, '2')
        self.__add_binary_instruction(Opcode.STORE, '2', direct_buffer_address)
        self.__add_unary_instruction_with_operand_address(Opcode.JNZ, jnz_argument_address)
        # Конец цикла записи
        self.__update_arg_for_jmp_instruction(jmp_address_end_index)
        self.__update_arg_for_jmp_instruction(jz_address_end_index)
        self.__update_arg_for_jmp_instruction(jz_address_end_index_second)
        self.__add_binary_instruction(Opcode.LOAD, '0', address_const_0)
        self.__add_push_instruction('0')

    def __create_machine_code_for_read_expression(
            self,
            name_var: str,
            buffer_size: int,
            param_names: tuple[str]) -> None:
        direct_buffer_address: str = get_direct_abs_address(self.__buffer_address)
        offset_buffer_address: str = get_direct_offset_address(self.__buffer_address)
        address_const_0: str = get_direct_abs_address(self.__number_consts[0])
        pointer_var_address: str = self.__get_variable_address(param_names, name_var)
        direct_start_string_address: str = self.__create_buffer(buffer_size)
        pointer_value_address: str = self.__create_pointer_value(int(direct_start_string_address, 16))
        self.__add_binary_instruction(Opcode.LOAD, '2', pointer_value_address)
        self.__add_binary_instruction(Opcode.STORE, '2', pointer_var_address)
        self.__add_binary_instruction(Opcode.STORE, '2', direct_buffer_address)
        # Начало цикла чтения
        jnz_argument_address: str = number_to_hex(InstrMemoryConfig.address_hex_num, len(self.__instruction_words))
        self.__add_binary_instruction(Opcode.LOAD, '3', address_const_0)
        for _ in range(DataMemoryConfig.word_size):
            self.__add_unary_instruction(Opcode.SRB, '3')
            self.__add_unary_instruction(Opcode.READ, '3')
        self.__add_binary_instruction(Opcode.STORE, '3', offset_buffer_address)
        self.__add_unary_instruction(Opcode.IES, '3')
        self.__add_unary_instruction(Opcode.INC, '2')
        self.__add_binary_instruction(Opcode.STORE, '2', direct_buffer_address)
        self.__add_unary_instruction_with_operand_address(Opcode.JNZ, jnz_argument_address)
        # Конец цикла чтения
        address_const_0: str = get_direct_abs_address(self.__number_consts[0])
        self.__add_binary_instruction(Opcode.LOAD, '0', address_const_0)
        self.__add_push_instruction('0')

    def __create_pointer_value(self, pointer_value: int) -> str:
        self.__create_number_const(pointer_value)
        return get_direct_abs_address(self.__number_consts[pointer_value])

    def __create_machine_code_for_binary_math_expression(
            self,
            key_word: str,
            first_raw_arg: str | None,
            second_raw_arg: str | None,
            param_names: tuple[str] | None) -> None:
        if isinstance(first_raw_arg, list):
            self.__create_expression(first_raw_arg, param_names)
            first_arg: None = None
        else:
            first_arg: str = first_raw_arg
        if isinstance(second_raw_arg, list):
            self.__create_expression(second_raw_arg, param_names)
            second_arg: None = None
        else:
            second_arg: str = second_raw_arg
        if first_arg is None and second_arg is None:
            first_arg_address: str = get_indirect_sp_address('1')  # &1
            second_arg_address: str = get_indirect_sp_address('0')  # &0
            assert len(first_arg_address) == InstrMemoryConfig.word_hex_num, \
                f"Incorrect word length - {first_arg_address}"
            assert len(second_arg_address) == InstrMemoryConfig.word_hex_num, \
                f"Incorrect word length - {second_arg_address}"
            self.__add_binary_math_instruction(key_word, first_arg_address, second_arg_address, '0')
            self.__add_pop_instruction('1')  # R1 <- pop
            self.__add_pop_instruction('1')  # R1 <- pop
        elif first_arg is not None and second_arg is None:
            first_arg_address: str = self.__get_arg_address_for_math_exp(first_arg, param_names)
            second_arg_address: str = get_indirect_sp_address('0')  # &0
            assert len(first_arg_address) == InstrMemoryConfig.word_hex_num, \
                f"Incorrect word length - {first_arg_address}"
            assert len(second_arg_address) == InstrMemoryConfig.word_hex_num, \
                f"Incorrect word length - {second_arg_address}"
            self.__add_binary_math_instruction(key_word, first_arg_address, second_arg_address, '0')
            self.__add_pop_instruction('1')  # R1 <- pop
        elif first_arg is None and second_arg is not None:
            first_arg_address: str = get_indirect_sp_address('0')  # &0
            second_arg_address: str = self.__get_arg_address_for_math_exp(second_arg, param_names)
            assert len(first_arg_address) == InstrMemoryConfig.word_hex_num, \
                f"Incorrect word length - {first_arg_address}"
            assert len(second_arg_address) == InstrMemoryConfig.word_hex_num, \
                f"Incorrect word length - {second_arg_address}"
            self.__add_binary_math_instruction(key_word, first_arg_address, second_arg_address, '0')
            self.__add_pop_instruction('1')  # R1 <- pop
        else:
            first_arg_address: str = self.__get_arg_address_for_math_exp(first_arg, param_names)
            second_arg_address: str = self.__get_arg_address_for_math_exp(second_arg, param_names)
            assert len(first_arg_address) == InstrMemoryConfig.word_hex_num, \
                f"Incorrect word length - {first_arg_address}"
            assert len(second_arg_address) == InstrMemoryConfig.word_hex_num, \
                f"Incorrect word length - {second_arg_address}"
            self.__add_binary_math_instruction(key_word, first_arg_address, second_arg_address, '0')
        self.__add_push_instruction('0')

    def __get_arg_address_for_math_exp(self, raw_arg_value: str, param_names: tuple[str] | None) -> str:
        if LiteralPatterns.is_number(raw_arg_value):
            arg_value: int = int(raw_arg_value)
            arg_address: str = get_direct_abs_address(self.__number_consts[arg_value])
        elif LiteralPatterns.is_name_var(raw_arg_value):
            arg_address: str = self.__get_variable_address(param_names, raw_arg_value)
        else:
            raise Exception(f"Value is not number or name variable - {raw_arg_value}")
        return arg_address

    def __create_machine_code_for_if_expression(
            self,
            first_raw_arg: str | list,
            second_raw_arg: str | list,
            third_raw_arg: str | list,
            param_names: tuple[str] | None) -> None:
        first_arg_address: str | None = self.__get_argument_for_if_expression(first_raw_arg, param_names)
        if first_arg_address is None:
            self.__add_pop_instruction('0')
        else:
            self.__add_binary_instruction(Opcode.LOAD, '0', first_arg_address)
        address_const_0: str = get_direct_abs_address(self.__number_consts[0])
        self.__add_binary_instruction(Opcode.CMP, '0', address_const_0)
        self.__add_unary_instruction_with_operand_address(Opcode.JNZ, '000')
        jnz_argument_address_index: int = len(self.__instruction_words) - 1
        second_arg_address: str | None = self.__get_argument_for_if_expression(second_raw_arg, param_names)
        if second_arg_address is not None:
            second_arg_address = get_direct_abs_address(second_arg_address)
            self.__add_binary_instruction(Opcode.LOAD, '0', second_arg_address)
            self.__add_push_instruction('0')
        self.__add_unary_instruction_with_operand_address(Opcode.JMP, '000')
        jmp_argument_address_index: int = len(self.__instruction_words) - 1
        self.__update_arg_for_jmp_instruction(jnz_argument_address_index)
        third_arg_address: str | None = self.__get_argument_for_if_expression(third_raw_arg, param_names)
        if third_arg_address is not None:
            third_arg_address = get_direct_abs_address(third_arg_address)
            self.__add_binary_instruction(Opcode.LOAD, '0', third_arg_address)
            self.__add_push_instruction('0')
        self.__update_arg_for_jmp_instruction(jmp_argument_address_index)

    def __get_argument_for_if_expression(self, raw_arg: str | list, param_names: tuple[str] | None) -> str | None:
        if isinstance(raw_arg, str):
            if LiteralPatterns.is_number(raw_arg):
                arg: int = int(raw_arg)
                arg_address: str = self.__number_consts[arg]
            elif LiteralPatterns.is_string(raw_arg):
                arg: str = raw_arg[1:-1]  # убираем ""
                arg_address: str = self.__string_consts[arg]
            elif LiteralPatterns.is_name_var(raw_arg):
                arg_address: str = self.__get_variable_address(param_names, raw_arg)
            else:
                raise Exception(f"Incorrect argument in if expression - {raw_arg}")
        elif isinstance(raw_arg, list):
            self.__create_expression(raw_arg, param_names)
            arg_address: None = None
        else:
            raise Exception(f"Incorrect argument in if expression - {raw_arg}")
        return arg_address

    def __create_machine_code_for_set_expression(
            self,
            name_var: str | None,
            second_arg_raw: str | None,
            param_names: tuple[str] | None) -> None:
        name_var_address: str = self.__get_variable_address(param_names, name_var)
        if isinstance(second_arg_raw, list):
            self.__create_expression(second_arg_raw, param_names)
            second_arg_address: None = None
        else:
            if LiteralPatterns.is_number(second_arg_raw):
                second_arg: int = int(second_arg_raw)
                second_arg_address: str = get_direct_abs_address(self.__number_consts[second_arg])
            elif LiteralPatterns.is_string(second_arg_raw):
                second_arg: str = second_arg_raw[1: -1]
                second_arg_address: str = get_direct_abs_address(self.__string_consts[second_arg])
            elif LiteralPatterns.is_name_var(second_arg_raw):
                second_arg_address: str = self.__get_variable_address(param_names, second_arg_raw)
            else:
                raise Exception(f"Incorrect second argument - {second_arg_raw}")
        if second_arg_address is None:
            self.__add_pop_instruction('0')
        else:
            self.__add_binary_instruction(Opcode.LOAD, '0', second_arg_address)
        self.__add_binary_instruction(Opcode.STORE, '0', name_var_address)
        self.__add_push_instruction('0')

    def __create_machine_code_for_call_expression(self, name_function: str, raw_args: list[str]) -> None:
        assert name_function in self.__functions.keys(), f"Undefined name function {name_function}"
        for raw_arg in reversed(raw_args):
            if LiteralPatterns.is_number(raw_arg):
                arg: int = int(raw_arg)
                arg_address: str = get_direct_abs_address(self.__number_consts[arg])
            elif LiteralPatterns.is_string(raw_arg):
                arg: str = raw_arg[1:-1]  # убираем ""
                if arg in self.__string_consts.keys():
                    arg_address: str = get_direct_abs_address(self.__string_consts[arg])
                else:
                    raise Exception(f"String not defined in data memory - {arg}")
            elif LiteralPatterns.is_name_var(raw_arg):
                if raw_arg in self.__vars.keys():
                    arg_address: str = get_direct_abs_address(self.__vars[raw_arg])
                else:
                    raise Exception(f"Variable not defined in data memory - {raw_arg}")
            else:
                raise Exception(f"Incorrect args passed to function {name_function}")
            self.__add_binary_instruction(Opcode.LOAD, '1', arg_address)
            self.__add_push_instruction('1')
        function_address: str = self.__functions[name_function]
        self.__offset_params += 1
        self.__add_unary_instruction_with_operand_address(Opcode.CALL, function_address)
        for _ in raw_args:
            self.__add_pop_instruction('1')
        self.__add_push_instruction('0')

    def __create_machine_code_for_iter_expression(
            self,
            name_iter_var: str,
            max_iter_value: int,
            nested_expression: list[str | list],
            param_names: tuple[str] | None) -> None:
        assert name_iter_var in self.__vars, f"Not defined iteration variable - {name_iter_var}"
        iter_var_address: str = get_direct_abs_address(self.__vars[name_iter_var])
        max_iter_value_address: str = get_direct_abs_address(self.__number_consts[max_iter_value])
        start_address: str = number_to_hex(InstrMemoryConfig.address_hex_num, len(self.__instruction_words))
        self.__add_binary_instruction(Opcode.LOAD, '1', max_iter_value_address)
        self.__add_binary_instruction(Opcode.CMP, '1', iter_var_address)
        self.__add_unary_instruction_with_operand_address(Opcode.JZ, '000')
        jz_address_end_index: int = len(self.__instruction_words) - 1
        self.__create_expression(nested_expression, param_names)
        self.__add_binary_instruction(Opcode.LOAD, '1', iter_var_address)
        self.__add_unary_instruction(Opcode.INC, '1')
        self.__add_binary_instruction(Opcode.STORE, '1', iter_var_address)
        self.__add_unary_instruction_with_operand_address(Opcode.JMP, start_address)
        self.__update_arg_for_jmp_instruction(jz_address_end_index)

    def __get_variable_address(self, param_names: tuple[str] | None, var_name: str) -> str:
        if param_names is not None and var_name in param_names:
            arg_address: str = get_indirect_sp_address(
                number_to_hex(
                    InstrMemoryConfig.address_hex_num,
                    param_names.index(var_name) + self.__offset_params
                )
            )
        else:
            arg_address: str = get_direct_abs_address(self.__vars[var_name])
        return arg_address

    def __update_arg_for_jmp_instruction(self, jmp_argument_address_index: int) -> None:
        self.__instruction_words[jmp_argument_address_index] = bytes.fromhex(get_direct_abs_address(
            number_to_hex(InstrMemoryConfig.address_hex_num, len(self.__instruction_words))
        ))

    # Добавление инструкций
    def __add_push_instruction(self, reg_num: str) -> None:
        self.__add_unary_instruction(Opcode.PUSH, reg_num)
        self.__offset_params += 1

    def __add_pop_instruction(self, reg_num: str) -> None:
        self.__add_unary_instruction(Opcode.POP, reg_num)
        self.__offset_params -= 1

    def __add_binary_math_instruction(
            self,
            key_word: str,
            first_arg_address: str,
            second_arg_address: str,
            reg_num: str) -> None:
        self.__add_binary_instruction(Opcode.LOAD, reg_num, first_arg_address)
        opcode: Opcode
        match key_word:
            case KeyWord.ADD.value:
                opcode = Opcode.ADD
            case KeyWord.SUB.value:
                opcode = Opcode.SUB
            case KeyWord.MUL.value:
                opcode = Opcode.MUL
            case KeyWord.DIV.value:
                opcode = Opcode.DIV
            case _:
                raise Exception("Not found keyword")
        self.__add_binary_instruction(opcode, reg_num, second_arg_address)

    def __add_zero_args_instruction(self, opcode: Opcode) -> None:
        opcode_word: str = get_opcode_word(opcode)
        assert len(opcode_word) == InstrMemoryConfig.word_hex_num, f"Incorrect word length - {opcode_word}"
        self.__instruction_words.append(bytes.fromhex(opcode_word))

    def __add_unary_instruction(self, opcode: Opcode, reg_num: str) -> None:
        opcode_word: str = get_opcode_word(opcode)
        reg_address_word: str = get_direct_reg_address(reg_num)
        assert len(opcode_word) == InstrMemoryConfig.word_hex_num, f"Incorrect word length - {opcode_word}"
        assert len(reg_address_word) == InstrMemoryConfig.word_hex_num, \
            f"Incorrect word length - {reg_address_word}"
        self.__instruction_words.append(bytes.fromhex(opcode_word))
        self.__instruction_words.append(bytes.fromhex(reg_address_word))

    def __add_unary_instruction_with_operand_address(self, opcode: Opcode, operand_address: str) -> None:
        opcode_word: str = get_opcode_word(opcode)
        operand_address_word: str = get_direct_abs_address(operand_address)
        assert len(opcode_word) == InstrMemoryConfig.word_hex_num, f"Incorrect word length - {opcode_word}"
        assert len(operand_address_word) == InstrMemoryConfig.word_hex_num, \
            f"Incorrect word length - {operand_address_word}"
        self.__instruction_words.append(bytes.fromhex(opcode_word))
        self.__instruction_words.append(bytes.fromhex(operand_address_word))

    def __add_binary_instruction(self, opcode: Opcode, reg_num: str, second_argument_address_word: str) -> None:
        opcode_word: str = get_opcode_word(opcode)
        reg_address_word: str = get_direct_reg_address(reg_num)
        assert len(opcode_word) == InstrMemoryConfig.word_hex_num, f"Incorrect word length - {opcode_word}"
        assert len(reg_address_word) == InstrMemoryConfig.word_hex_num, \
            f"Incorrect word length - {reg_address_word}"
        assert len(second_argument_address_word) == InstrMemoryConfig.word_hex_num, \
            f"Incorrect word length - {second_argument_address_word}"
        self.__instruction_words.append(bytes.fromhex(opcode_word))
        self.__instruction_words.append(bytes.fromhex(second_argument_address_word))
        self.__instruction_words.append(bytes.fromhex(reg_address_word))


def translate_and_save_output_in_files(
        parsed_source: list[str],
        output_instruction_file: str,
        output_data_file: str,
        output_mnemonic_file: str) -> Translator:
    translator: Translator = Translator(parsed_source)
    translator.translate()
    translator.save_instruction_words_in_file(output_instruction_file)
    translator.save_data_memory_in_file(output_data_file)
    mnemonic_creator: MnemonicCreator = MnemonicCreator(
        translator.instruction_words,
        translator.vars_by_address,
        translator.number_consts_by_address,
        translator.string_consts_by_address,
        translator.functions_by_address
    )
    mnemonic_creator.save_mnemonic_in_file(output_mnemonic_file)
    return translator


def main(input_file: str, output_instruction_file: str, output_data_file: str, output_mnemonic_file: str) -> None:
    parsed_source: list = parsed_and_check_source_file(input_file)
    translate_and_save_output_in_files(
        parsed_source,
        output_instruction_file,
        output_data_file,
        output_mnemonic_file
    )


if __name__ == '__main__':
    assert len(sys.argv) == 5, \
        "Wrong args: translator.py <input_file> <output_instruction_file> <output_data_file> <output_mnemonic_file>"
    _, input_file_arg, output_instruction_file_arg, output_data_file_arg, output_mnemonic_file_arg = sys.argv
    main(input_file_arg, output_instruction_file_arg, output_data_file_arg, output_mnemonic_file_arg)
