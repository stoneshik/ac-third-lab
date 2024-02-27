from dataclasses import dataclass


@dataclass
class InstrMemoryConfig:
    size: int = 4096
    word_size: int = 2  # 2 байта
    word_hex_num: int = 4  # число 16-х чисел для представления машинного слова
    address_hex_num: int = 3  # число 16-х чисел для адресации


@dataclass
class DataMemoryConfig:
    size: int = 4096
    named_memory_size: int = 255
    word_size: int = 4
    word_hex_num: int = 8  # число 16-х чисел для представления машинного слова
    address_hex_num: int = 3  # число 16-х чисел для адресации
    input_port: int = 0
    output_port: int = 1
    heap_counter_value: int = 2
