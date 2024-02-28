from enum import Enum

from memory_config import InstrMemoryConfig


class KeyWord(Enum):
    PRINT_NUMBER: str = 'print_number'
    PRINT: str = 'print'
    READ: str = 'read'
    ADD: str = '+'
    SUB: str = '-'
    MUL: str = '*'
    DIV: str = '/'
    IF: str = 'if'
    FUNCTION: str = 'aboba'
    VAR: str = 'var'
    SET: str = 'set'
    CALL: str = 'call'
    ITER: str = 'iter'


all_keywords: tuple = (
    KeyWord.PRINT_NUMBER.value,
    KeyWord.PRINT.value,
    KeyWord.READ.value,
    KeyWord.ADD.value,
    KeyWord.SUB.value,
    KeyWord.MUL.value,
    KeyWord.DIV.value,
    KeyWord.IF.value,
    KeyWord.FUNCTION.value,
    KeyWord.VAR.value,
    KeyWord.SET.value,
    KeyWord.CALL.value,
    KeyWord.ITER.value
)


class Opcode(Enum):
    NOP: str = '00'
    HALT: str = '10'
    CHAR: str = '20'
    INC: str = '21'
    DEC: str = '22'
    ADD: str = '23'
    SUB: str = '24'
    MUL: str = '25'
    DIV: str = '26'
    SLB: str = '27'
    SRB: str = '28'
    MOD: str = '29'
    AND: str = '30'
    OR: str = '31'
    LOAD: str = '40'
    STORE: str = '41'
    PUSH: str = '50'
    POP: str = '51'
    CMP: str = '60'
    IES: str = '61'
    JMP: str = '70'
    CALL: str = '71'
    RET: str = '72'
    JZ: str = '73'
    JNZ: str = '74'
    READ: str = '80'
    PRINT: str = '81'

    def __str__(self):
        return str(self.value)


opcode_by_hex_dict: dict[str, Opcode] = {
    '00': Opcode.NOP,
    '10': Opcode.HALT,
    '20': Opcode.CHAR,
    '21': Opcode.INC,
    '22': Opcode.DEC,
    '23': Opcode.ADD,
    '24': Opcode.SUB,
    '25': Opcode.MUL,
    '26': Opcode.DIV,
    '27': Opcode.SLB,
    '28': Opcode.SRB,
    '29': Opcode.MOD,
    '30': Opcode.AND,
    '31': Opcode.OR,
    '40': Opcode.LOAD,
    '41': Opcode.STORE,
    '50': Opcode.PUSH,
    '51': Opcode.POP,
    '60': Opcode.CMP,
    '61': Opcode.IES,
    '70': Opcode.JMP,
    '71': Opcode.CALL,
    '72': Opcode.RET,
    '73': Opcode.JZ,
    '74': Opcode.JNZ,
    '80': Opcode.READ,
    '81': Opcode.PRINT
}


class AddressCode(Enum):
    DIRECT_ABS: str = 'a'  # Прямая абсолютная на память
    DIRECT_OFFSET: str = 'b'  # Прямая косвенная на память
    INDIRECT_SP: str = 'c'  # Косвенная относительная со смещением от (SP) на память
    DIRECT_REG: str = 'd'  # Прямая абсолютная на регистр

    def __str__(self):
        return str(self.value)


address_by_hex_dict: dict[str, AddressCode] = {
    'a': AddressCode.DIRECT_ABS,
    'b': AddressCode.DIRECT_OFFSET,
    'c': AddressCode.INDIRECT_SP,
    'd': AddressCode.DIRECT_REG
}


def number_to_hex(word_hex_num: int, value: int) -> str:
    hex_value: str = hex(value)[2:]
    hex_value = '0' * (word_hex_num - len(hex_value)) + hex_value
    return hex_value


def string_to_hex_list(word_hex_num: int, word_size: int, value: str) -> list[str]:
    chars: list[str] = []
    for char in reversed(value):
        if char == '\\':
            if chars[-1] == 'n':
                chars[-1] = '\n'
            elif chars[-1] == 't':
                chars[-1] = '\t'
            elif chars[-1] == 'r':
                chars[-1] = '\r'
            else:
                raise Exception(f"Incorrect special char \\{char[-1]}")
            continue
        chars.append(char)
    chars.reverse()
    grouped_chars: list[list[str]] = [chars[i:i + word_size] for i in range(0, len(chars), word_size)]
    hex_list: list[str] = []
    for group_char in grouped_chars:
        hex_value: str = ''
        for char in group_char:
            symbol_code = ord(char)
            assert -128 <= symbol_code <= 127, f"it's not ascii symbol: code - {symbol_code}"
            hex_value = hex(symbol_code)[2:] + hex_value
        hex_value = '0' * (word_hex_num - len(hex_value)) + hex_value
        hex_list.append(hex_value)
    return hex_list


def get_opcode_word(opcode: Opcode) -> str:
    return opcode.value + '0' * (InstrMemoryConfig.word_hex_num - len(opcode.value))


def get_direct_abs_address(address: str) -> str:
    return AddressCode.DIRECT_ABS.value + address


def get_direct_offset_address(address: str) -> str:
    return AddressCode.DIRECT_OFFSET.value + address


def get_indirect_sp_address(offset: str) -> str:
    return (AddressCode.INDIRECT_SP.value +
            '0' * (InstrMemoryConfig.word_hex_num - len(AddressCode.INDIRECT_SP.value) - len(offset))
            + offset)


def get_direct_reg_address(reg_num: str) -> str:
    return (AddressCode.DIRECT_REG.value +
            '0' * (InstrMemoryConfig.word_hex_num - len(AddressCode.DIRECT_REG.value) - len(reg_num))
            + reg_num)
