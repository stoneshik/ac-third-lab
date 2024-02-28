#!/usr/bin/python3
import logging
import sys
from typing import Callable


from memory_config import DataMemoryConfig, InstrMemoryConfig
from isa import Opcode, AddressCode, number_to_hex


class Registers:
    def __init__(self) -> None:
        self.r0: str = '0' * DataMemoryConfig.word_hex_num
        self.r1: str = '0' * DataMemoryConfig.word_hex_num
        self.r2: str = '0' * DataMemoryConfig.word_hex_num
        self.r3: str = '0' * DataMemoryConfig.word_hex_num
        self.__oer0: bool = True
        self.__oer1: bool = False
        self.__oer2: bool = False
        self.__oer3: bool = False

    def r0_setter(self, value: str) -> None:
        self.r0 = value

    def r1_setter(self, value: str) -> None:
        self.r1 = value

    def r2_setter(self, value: str) -> None:
        self.r2 = value

    def r3_setter(self, value: str) -> None:
        self.r3 = value

    def signal_oer0(self) -> None:
        self.__oer0 = True
        self.__oer1 = False
        self.__oer2 = False
        self.__oer3 = False

    def signal_oer1(self) -> None:
        self.__oer0 = False
        self.__oer1 = True
        self.__oer2 = False
        self.__oer3 = False

    def signal_oer2(self) -> None:
        self.__oer0 = False
        self.__oer1 = False
        self.__oer2 = True
        self.__oer3 = False

    def signal_oer3(self) -> None:
        self.__oer0 = False
        self.__oer1 = False
        self.__oer2 = False
        self.__oer3 = True

    def output_data(self) -> str:
        if self.__oer0:
            return self.r0
        elif self.__oer1:
            return self.r1
        elif self.__oer2:
            return self.r2
        elif self.__oer3:
            return self.r3
        else:
            raise Exception("register from which data should be read has not been selected")

    def __repr__(self) -> str:
        state_repr: str = (
            "Registers - R0: {} R1: {} R2: {} R3: {} oer0: {} oer1: {} oer2: {} oer3: {}"
            .format(
                self.r0,
                self.r1,
                self.r2,
                self.r3,
                self.__oer0,
                self.__oer1,
                self.__oer2,
                self.__oer3,
            )
        )
        return state_repr


class ALU:
    def __init__(self) -> None:
        self.__result: str = '0' * DataMemoryConfig.word_hex_num
        self.__zero: bool = False

    @property
    def result(self) -> str:
        return self.__result

    @property
    def zero(self) -> bool:
        return self.__zero

    def calc(self, instruction: str, first_operand: str, second_operand: str) -> None:
        opcode_hex: str = instruction[:2]
        match opcode_hex:
            case Opcode.CHAR.value:
                symbol: str = first_operand[6:]
                symbol_code: ord = ord(str(int(symbol, 16)))
                assert -128 <= symbol_code <= 127, f"input token is out of bound: {symbol_code}"
                symbol_hex = number_to_hex(
                    DataMemoryConfig.word_hex_num, symbol_code
                )
                self.__result = symbol_hex
            case Opcode.INC.value:
                self.__result = number_to_hex(DataMemoryConfig.word_hex_num, int(first_operand, 16) + 1)
            case Opcode.DEC.value:
                self.__result = number_to_hex(DataMemoryConfig.word_hex_num, int(first_operand, 16) - 1)
            case Opcode.ADD.value:
                self.__result = number_to_hex(
                    DataMemoryConfig.word_hex_num, int(first_operand, 16) + int(second_operand, 16)
                )
            case Opcode.SUB.value:
                self.__result = number_to_hex(
                    DataMemoryConfig.word_hex_num, int(first_operand, 16) - int(second_operand, 16)
                )
            case Opcode.MUL.value:
                self.__result = number_to_hex(
                    DataMemoryConfig.word_hex_num, int(first_operand, 16) * int(second_operand, 16)
                )
            case Opcode.DIV.value:
                self.__result = number_to_hex(
                    DataMemoryConfig.word_hex_num, int(first_operand, 16) // int(second_operand, 16)
                )
            case Opcode.SLB.value:
                self.__result = number_to_hex(
                    DataMemoryConfig.word_hex_num, int(first_operand, 16) << 8
                )
            case Opcode.SRB.value:
                self.__result = number_to_hex(
                    DataMemoryConfig.word_hex_num, int(first_operand, 16) >> 8
                )
            case Opcode.MOD.value:
                self.__result = number_to_hex(
                    DataMemoryConfig.word_hex_num, int(first_operand, 16) % int(second_operand, 16)
                )
            case Opcode.AND.value:
                self.__result = number_to_hex(
                    DataMemoryConfig.word_hex_num, int(first_operand, 16) & int(second_operand, 16)
                )
            case Opcode.OR.value:
                self.__result = number_to_hex(
                    DataMemoryConfig.word_hex_num, int(first_operand, 16) | int(second_operand, 16)
                )
            case Opcode.LOAD.value | Opcode.POP.value:
                self.__result = second_operand
            case Opcode.CMP.value:
                self.__result = number_to_hex(
                    DataMemoryConfig.word_hex_num, int(first_operand, 16) - int(second_operand, 16)
                )
                self.__zero = self.__result == '0' * DataMemoryConfig.word_hex_num
            case Opcode.IES.value:
                self.__result = number_to_hex(
                    DataMemoryConfig.word_hex_num, int(first_operand, 16) & int('F0000000', 16)
                )
                self.__zero = self.__result == '0' * DataMemoryConfig.word_hex_num
            case _:
                raise Exception(f"alu not supported instruction {opcode_hex}")

    def __repr__(self) -> str:
        state_repr: str = "ALU - result: {} zero: {}".format(
            self.__result,
            self.zero
        )
        return state_repr


class DataPath:
    def __init__(self, input_buffer: list[str], data_memory: list[str]) -> None:
        assert len(data_memory) == DataMemoryConfig.size, f"incorrect size data memory {len(data_memory)}"
        self.__data_memory: list[str] = data_memory
        self.__input_buffer: list[str] = input_buffer
        self.__output_buffer: list[str] = []
        self.__address_register: str = '000'
        self.__stack_pointer: str = 'fff'
        self.__stack_buffer: str = '000'
        self.__heap_counter: str = '0ff'
        self.__second_operand_buffer: str = '0' * DataMemoryConfig.word_hex_num
        self.__registers: Registers = Registers()
        self.__alu: ALU = ALU()
        self.__oea: bool = False

    @property
    def output_buffer(self) -> list[str]:
        return self.__output_buffer

    @property
    def stack_pointer(self) -> str:
        return self.__stack_pointer

    @property
    def heap_counter(self) -> str:
        return self.__heap_counter

    @property
    def zero(self) -> bool:
        return self.__alu.zero

    def get_value_from_data_memory(self) -> str:
        return self.__data_memory[int(self.__address_register, 16)]

    def signal_latch_stack_pointer(self, sel: str) -> None:
        opcode_hex: str = sel[:2]
        match opcode_hex:
            case Opcode.POP.value | Opcode.RET.value:
                self.__stack_pointer = number_to_hex(
                    DataMemoryConfig.address_hex_num, int(self.__stack_pointer, 16) + 1
                )
            case Opcode.PUSH.value | Opcode.CALL.value:
                self.__stack_pointer = number_to_hex(
                    DataMemoryConfig.address_hex_num, int(self.__stack_pointer, 16) - 1
                )
            case _:
                raise Exception(f"internal error, incorrect selector: {sel}")

    def signal_latch_stack_buffer(self, sel: str) -> None:
        address_code_hex: str = sel[:1]
        n: int = int(sel[1:], 16)
        match address_code_hex:
            case AddressCode.INDIRECT_SP.value:
                self.__stack_buffer = number_to_hex(
                    DataMemoryConfig.address_hex_num, int(self.__stack_pointer, 16) + n
                )
                assert int(self.__stack_buffer, 16) < DataMemoryConfig.size, \
                    f"internal error, indirect address (SP) go out memory data {self.__stack_buffer}"
            case _:
                raise Exception(f"internal error, incorrect selector: {sel}")

    def signal_latch_heap_counter(self) -> None:
        self.__heap_counter = self.__data_memory[DataMemoryConfig.heap_counter_value][5:]
        assert DataMemoryConfig.named_memory_size < int(self.__heap_counter, 16) < DataMemoryConfig.size, \
            f"internal error heap counter value go out permissible range {self.__heap_counter}"

    def signal_latch_address_reg(self, sel: str) -> None:
        if self.__oea:
            self.__address_register = self.__data_memory[int(self.__address_register, 16)][5:]
            self.__oea = False
            return
        opcode_hex: str = sel[:2]
        match opcode_hex:
            case Opcode.PUSH.value | Opcode.POP.value | Opcode.CALL.value | Opcode.RET.value:
                self.__address_register = self.__stack_pointer
                return
        address_code_hex: str = sel[:1]
        arg_address: str = sel[1:]
        match address_code_hex:
            case AddressCode.DIRECT_ABS.value | AddressCode.DIRECT_OFFSET.value:
                self.__address_register = arg_address
            case AddressCode.INDIRECT_SP.value:
                self.__address_register = self.__stack_buffer
            case _:
                raise Exception("internal error")

    def signal_latch_second_op_buf(self) -> None:
        self.__second_operand_buffer = self.__data_memory[int(self.__address_register, 16)]

    def signal_output_enable_addr(self) -> None:
        self.__oea = True

    def signal_output_enable(self) -> None:
        return

    def signal_latch_r0(self, sel_instruction: str) -> None:
        self.__signal_latch(sel_instruction, self.__registers.r0, self.__registers.r0_setter)

    def signal_latch_r1(self, sel_instruction: str) -> None:
        self.__signal_latch(sel_instruction, self.__registers.r1, self.__registers.r1_setter)

    def signal_latch_r2(self, sel_instruction: str) -> None:
        self.__signal_latch(sel_instruction, self.__registers.r2, self.__registers.r2_setter)

    def signal_latch_r3(self, sel_instruction: str) -> None:
        self.__signal_latch(sel_instruction, self.__registers.r3, self.__registers.r3_setter)

    def __signal_latch(self, sel_instruction: str, r_value: str, r_setter: Callable[[str], None]) -> None:
        opcode_hex: str = sel_instruction[:2]
        if opcode_hex == Opcode.READ.value:
            if len(self.__input_buffer) == 0:
                raise EOFError()
            symbol = self.__input_buffer.pop(0)
            symbol_code = ord(symbol)
            assert -128 <= symbol_code <= 127, f"input token is out of bound: {symbol_code}"
            symbol_hex = number_to_hex(2, symbol_code)
            r_setter(symbol_hex + r_value[2:])
            logging.debug("input: %s", repr(symbol))
            return
        r_setter(self.__alu.result)

    def signal_oer0(self) -> None:
        self.__registers.signal_oer0()

    def signal_oer1(self) -> None:
        self.__registers.signal_oer1()

    def signal_oer2(self) -> None:
        self.__registers.signal_oer2()

    def signal_oer3(self) -> None:
        self.__registers.signal_oer3()

    def signal_write_in_mem(self, sel: str, pc_value: str = None) -> None:
        opcode_hex: str = sel[:2]
        if opcode_hex == Opcode.CALL.value:
            self.__data_memory[int(self.__address_register, 16)] = pc_value
            return
        self.__data_memory[int(self.__address_register, 16)] = self.__registers.output_data()

    def signal_calc(self, instruction: str) -> None:
        first_operand: str = self.__registers.output_data()
        second_operand: str = self.__second_operand_buffer
        self.__alu.calc(instruction, first_operand, second_operand)

    def signal_input(self) -> None:
        return

    def signal_output(self) -> None:
        value_output: ord = ord(bytes.fromhex(self.__registers.output_data()[6:]))
        if value_output == 0:
            logging.debug("zero value skipped for output: %s << %s",
                          str("".join(self.output_buffer)), chr(value_output))
            return
        logging.debug("output: %s << %s", str("".join(self.output_buffer)), chr(value_output))
        self.__output_buffer.append(chr(value_output))

    def __repr__(self) -> str:
        state_repr: str = "DataPath - AR: {} SP: {} SB: {} HC: {} SOB: {} ({}) ({})".format(
            self.__address_register,
            self.__stack_pointer,
            self.__stack_buffer,
            self.__heap_counter,
            self.__second_operand_buffer,
            self.__alu,
            self.__registers
        )
        return state_repr


class ControlUnit:
    def __init__(self, instructions: list[str], data_path: DataPath) -> None:
        zero_word_instr_memory: str = '0' * InstrMemoryConfig.word_hex_num
        self.__instr_memory: list[str] = [zero_word_instr_memory] * InstrMemoryConfig.size
        self.__fill_instr_memory(instructions)
        self.__program_counter: str = '000'
        self.__instruction_buffer: str = zero_word_instr_memory
        self.__data_path: DataPath = data_path
        self.__tick: int = 0

    def __fill_instr_memory(self, instructions: list[str]) -> None:
        assert len(instructions) <= len(self.__instr_memory), "program does not fit into memory"
        for i, instruction in enumerate(instructions):
            self.__instr_memory[i] = instruction

    @property
    def tick(self) -> int:
        return self.__tick

    def __tick_inc(self) -> None:
        self.__tick += 1

    def signal_latch_heap_counter(self) -> None:
        self.__data_path.signal_latch_heap_counter()

    def decode_and_execute_instruction(self) -> None:
        instruction: str = self.__instr_memory[int(self.__program_counter, 16)]
        opcode_hex: str = instruction[:2]
        match opcode_hex:
            case Opcode.CALL.value | Opcode.RET.value | Opcode.JMP.value | Opcode.JZ.value | Opcode.JNZ.value:
                self.__execute_control_flow_instruction(instruction)
                return
            case Opcode.NOP.value | Opcode.HALT.value:
                self.__execute_zero_arg_instruction(instruction)
            case Opcode.INC.value | Opcode.DEC.value | Opcode.SLB.value | Opcode.SRB.value | Opcode.CHAR.value:
                self.__execute_one_arg_instruction(instruction)
            case Opcode.ADD.value | Opcode.SUB.value | Opcode.MUL.value | Opcode.DIV.value | (
                    Opcode.AND.value) | Opcode.OR.value | Opcode.MOD.value:
                self.__execute_two_arg_instruction(instruction)
            case Opcode.LOAD.value:
                self.__execute_load_instruction(instruction)
            case Opcode.STORE.value:
                self.__execute_store_instruction(instruction)
            case Opcode.POP.value:
                self.__execute_pop_instruction(instruction)
            case Opcode.PUSH.value:
                self.__execute_push_instruction(instruction)
            case Opcode.READ.value:
                self.__execute_read_instruction(instruction)
            case Opcode.PRINT.value:
                self.__execute_print_instruction(instruction)
            case Opcode.IES.value:
                self.__execute_ies_instruction(instruction)
            case Opcode.CMP.value:
                self.__execute_cmp_instruction(instruction)
        self.__next_instr_word()

    def __execute_control_flow_instruction(self, instruction: str) -> None:
        opcode_hex: str = instruction[:2]
        match opcode_hex:
            case Opcode.CALL.value:
                self.__signal_latch_instruction_buf(instruction)
                self.__tick_inc()
                self.__start_push_operation_with_stack(instruction)
                self.__next_instr_word()
                self.__next_instr_word()
                self.__data_path.signal_write_in_mem(
                    self.__instruction_buffer,
                    pc_value=number_to_hex(
                        DataMemoryConfig.word_hex_num, int(self.__program_counter, 16)
                    )
                )
                self.__jump_to_instruction()
            case Opcode.RET.value:
                self.__signal_latch_instruction_buf(instruction)
                self.__tick_inc()
                self.__start_pop_operation_with_stack(instruction)
                self.__data_path.signal_output_enable()
                self.__tick_inc()
                self.__jump_to_instruction()
            case Opcode.JMP.value:
                self.__save_instr_and_go_next(instruction)
                self.__jump_to_instruction()
            case Opcode.JZ.value:
                self.__save_instr_and_go_next(instruction)
                if self.__data_path.zero:
                    self.__jump_to_instruction()
                else:
                    self.__next_instr_word()
            case Opcode.JNZ.value:
                self.__save_instr_and_go_next(instruction)
                if not self.__data_path.zero:
                    self.__jump_to_instruction()
                else:
                    self.__next_instr_word()
            case _:
                raise Exception("internal error")
    
    @staticmethod
    def __execute_zero_arg_instruction(instruction: str) -> None:
        opcode_hex: str = instruction[:2]
        match opcode_hex:
            case Opcode.HALT.value:
                raise StopIteration()
            case Opcode.NOP.value:
                return
            case _:
                raise Exception("internal error")

    def __execute_one_arg_instruction(self, instruction: str) -> None:
        opcode_hex: str = instruction[:2]
        match opcode_hex:
            case Opcode.INC.value | Opcode.DEC.value | Opcode.SLB.value | Opcode.SRB.value | Opcode.CHAR.value:
                self.__save_instr_and_go_next(instruction)
                arg_word: str = self.__get_current_instruction_word()
                self.__prepare_register_to_output(arg_word)
                self.__calc()
                self.__write_in_register(arg_word)
            case _:
                raise Exception("internal error")

    def __execute_two_arg_instruction(self, instruction: str) -> None:
        opcode_hex: str = instruction[:2]
        match opcode_hex:
            case Opcode.ADD.value | Opcode.SUB.value | Opcode.MUL.value | Opcode.DIV.value | (
                    Opcode.AND.value) | Opcode.OR.value | Opcode.MOD.value:
                self.__save_instr_and_go_next(instruction)
                second_arg_word: str = self.__get_current_instruction_word()
                self.__mem_to_alu(second_arg_word)
                self.__next_instr_word()
                first_arg_word: str = self.__get_current_instruction_word()
                self.__prepare_register_to_output(first_arg_word)
                self.__calc()
                self.__write_in_register(first_arg_word)
            case _:
                raise Exception("internal error")

    def __execute_load_instruction(self, instruction: str) -> None:
        opcode_hex: str = instruction[:2]
        assert opcode_hex == Opcode.LOAD.value, "internal error"
        self.__save_instr_and_go_next(instruction)
        second_arg_word: str = self.__get_current_instruction_word()
        self.__mem_to_alu(second_arg_word)
        self.__next_instr_word()
        first_arg_word: str = self.__get_current_instruction_word()
        self.__calc()
        self.__write_in_register(first_arg_word)

    def __execute_store_instruction(self, instruction: str) -> None:
        opcode_hex: str = instruction[:2]
        assert opcode_hex == Opcode.STORE.value, "internal error"
        self.__save_instr_and_go_next(instruction)
        second_arg_word: str = self.__get_current_instruction_word()
        self.__set_address_for_arg(second_arg_word)
        self.__next_instr_word()
        first_arg_word: str = self.__get_current_instruction_word()
        self.__prepare_register_to_output(first_arg_word)
        self.__data_path.signal_write_in_mem(first_arg_word)
        self.__tick_inc()

    def __execute_pop_instruction(self, instruction: str) -> None:
        opcode_hex: str = instruction[:2]
        assert opcode_hex == Opcode.POP.value, "internal error"
        self.__signal_latch_instruction_buf(instruction)
        self.__tick_inc()
        self.__start_pop_operation_with_stack(instruction)
        self.__data_path.signal_output_enable()
        self.__tick_inc()
        self.__data_path.signal_latch_second_op_buf()
        self.__tick_inc()
        self.__next_instr_word()
        self.__calc()
        arg_word: str = self.__get_current_instruction_word()
        self.__write_in_register(arg_word)

    def __execute_push_instruction(self, instruction: str) -> None:
        opcode_hex: str = instruction[:2]
        assert opcode_hex == Opcode.PUSH.value, "internal error"
        self.__signal_latch_instruction_buf(instruction)
        self.__tick_inc()
        self.__start_push_operation_with_stack(instruction)
        self.__next_instr_word()
        arg_word: str = self.__get_current_instruction_word()
        self.__prepare_register_to_output(arg_word)
        self.__data_path.signal_write_in_mem(arg_word)

    def __execute_read_instruction(self, instruction: str) -> None:
        opcode_hex: str = instruction[:2]
        assert opcode_hex == Opcode.READ.value, "internal error"
        self.__save_instr_and_go_next(instruction)
        arg_word: str = self.__get_current_instruction_word()
        self.__data_path.signal_input()
        self.__tick_inc()
        self.__write_in_register(arg_word)

    def __execute_print_instruction(self, instruction: str) -> None:
        opcode_hex: str = instruction[:2]
        assert opcode_hex == Opcode.PRINT.value, "internal error"
        self.__save_instr_and_go_next(instruction)
        arg_word: str = self.__get_current_instruction_word()
        self.__prepare_register_to_output(arg_word)
        self.__data_path.signal_output()
        self.__tick_inc()

    def __execute_ies_instruction(self, instruction: str) -> None:
        opcode_hex: str = instruction[:2]
        assert opcode_hex == Opcode.IES.value, "internal error"
        self.__save_instr_and_go_next(instruction)
        arg_word: str = self.__get_current_instruction_word()
        self.__prepare_register_to_output(arg_word)
        self.__calc()

    def __execute_cmp_instruction(self, instruction: str) -> None:
        opcode_hex: str = instruction[:2]
        assert opcode_hex == Opcode.CMP.value, "internal error"
        self.__save_instr_and_go_next(instruction)
        second_arg_word: str = self.__get_current_instruction_word()
        self.__mem_to_alu(second_arg_word)
        self.__next_instr_word()
        first_arg_word: str = self.__get_current_instruction_word()
        self.__prepare_register_to_output(first_arg_word)
        self.__calc()

    # Сигналы
    def __signal_latch_program_counter(self, sel_next: bool) -> None:
        if sel_next:
            self.__program_counter = number_to_hex(
                InstrMemoryConfig.address_hex_num, int(self.__program_counter, 16) + 1
            )
            return
        instruction: str = self.__instruction_buffer
        opcode_hex: str = instruction[:2]
        if opcode_hex == Opcode.RET.value:
            jump_to: str = self.__data_path.get_value_from_data_memory()[5:]
        elif opcode_hex == Opcode.CALL.value:
            prev_instruction_word: str = self.__instr_memory[int(self.__program_counter, 16) - 1]
            jump_to: str = prev_instruction_word[1:]
        else:
            current_instruction_word: str = self.__instr_memory[int(self.__program_counter, 16)]
            jump_to: str = current_instruction_word[1:]
        self.__program_counter = jump_to

    def __signal_latch_instruction_buf(self, current_instruction_word: str) -> None:
        self.__instruction_buffer = current_instruction_word

    # Вспомогательные методы
    def __get_current_instruction_word(self) -> str:
        return self.__instr_memory[int(self.__program_counter, 16)]

    def __next_instr_word(self) -> None:
        self.__signal_latch_program_counter(sel_next=True)
        self.__tick_inc()

    def __jump_to_instruction(self) -> None:
        self.__signal_latch_program_counter(sel_next=False)
        self.__tick_inc()

    def __calc(self) -> None:
        self.__data_path.signal_calc(self.__instruction_buffer)
        self.__tick_inc()

    def __save_instr_and_go_next(self, instruction: str) -> None:
        self.__signal_latch_instruction_buf(instruction)
        self.__tick_inc()
        self.__next_instr_word()

    def __start_push_operation_with_stack(self, instruction: str) -> None:
        self.__data_path.signal_latch_stack_pointer(instruction)
        self.__check_stack_pointer()
        self.__tick_inc()
        self.__data_path.signal_latch_address_reg(instruction)
        self.__tick_inc()

    def __start_pop_operation_with_stack(self, instruction: str) -> None:
        self.__data_path.signal_latch_address_reg(instruction)
        self.__tick_inc()
        self.__data_path.signal_latch_stack_pointer(instruction)
        self.__check_stack_pointer()
        self.__tick_inc()

    def __check_stack_pointer(self) -> bool:
        assert self.__data_path.stack_pointer >= self.__data_path.heap_counter, \
            f"stack pointer points to heap {self.__data_path.stack_pointer}"
        return True

    def __mem_to_alu(self, arg_word: str) -> None:
        self.__set_address_for_arg(arg_word)
        self.__data_path.signal_output_enable()
        self.__tick_inc()
        self.__data_path.signal_latch_second_op_buf()
        self.__tick_inc()

    def __set_address_for_arg(self, arg_word: str) -> None:
        address_code_hex: str = arg_word[:1]
        match address_code_hex:
            case AddressCode.DIRECT_ABS.value:
                self.__data_path.signal_latch_address_reg(arg_word)
                self.__tick_inc()
            case AddressCode.DIRECT_OFFSET.value:
                self.__data_path.signal_latch_address_reg(arg_word)
                self.__tick_inc()
                self.__data_path.signal_output_enable_addr()
                self.__tick_inc()
                self.__data_path.signal_latch_address_reg(arg_word)
                self.__tick_inc()
            case AddressCode.INDIRECT_SP.value:
                self.__data_path.signal_latch_stack_buffer(arg_word)
                self.__tick_inc()
                self.__data_path.signal_latch_address_reg(arg_word)
                self.__tick_inc()
            case _:
                raise Exception("internal error")

    def __write_in_register(self, current_instruction_word: str) -> None:
        address_code_hex: str = current_instruction_word[:1]
        assert address_code_hex == AddressCode.DIRECT_REG.value, \
            f"address code doesn't reference to a register {address_code_hex}"
        num_register: str = current_instruction_word[1:]
        instruction: str = self.__instruction_buffer
        match num_register:
            case '000':
                self.__data_path.signal_latch_r0(instruction)
            case '001':
                self.__data_path.signal_latch_r1(instruction)
            case '002':
                self.__data_path.signal_latch_r2(instruction)
            case '003':
                self.__data_path.signal_latch_r3(instruction)
        self.__tick_inc()

    def __prepare_register_to_output(self, current_instruction_word: str) -> None:
        address_code_hex: str = current_instruction_word[:1]
        assert address_code_hex == AddressCode.DIRECT_REG.value, \
            f"address code doesn't reference to a register {address_code_hex}"
        num_register: str = current_instruction_word[1:]
        match num_register:
            case '000':
                self.__data_path.signal_oer0()
            case '001':
                self.__data_path.signal_oer1()
            case '002':
                self.__data_path.signal_oer2()
            case '003':
                self.__data_path.signal_oer3()
        self.__tick_inc()

    def __repr__(self) -> str:
        state_repr: str = "TICK: {} PC: {} IB {} ({})".format(
            self.__tick,
            self.__program_counter,
            self.__instruction_buffer,
            self.__data_path
        )
        return state_repr


def simulation(
        input_tokens: list[str],
        instructions: list[str],
        data_memory: list[str], limit: int) -> tuple[str, int, int]:
    data_path: DataPath = DataPath(input_tokens, data_memory)
    control_unit: ControlUnit = ControlUnit(instructions, data_path)
    instr_counter: int = 0
    control_unit.signal_latch_heap_counter()
    logging.debug("%s | %s", instr_counter, control_unit)
    try:
        while instr_counter < limit:
            try:
                control_unit.decode_and_execute_instruction()
            except EOFError:
                logging.warning("Input buffer is empty!")
            instr_counter += 1
            logging.debug("%s | %s", instr_counter, control_unit)
    except StopIteration:
        pass
    if instr_counter >= limit:
        logging.warning("Limit exceeded!")
    logging.info("output_buffer: %s", repr("".join(data_path.output_buffer)))
    return "".join(data_path.output_buffer), instr_counter, control_unit.tick


def read_input_file(input_file_name: str) -> list[str]:
    with open(input_file_name, 'rt', encoding='utf-8') as source_file:
        source: str = source_file.read()
    return list(source)


def read_input_byte_file(input_file_name: str, word_size: int) -> list[str]:
    result_list: list[str] = []
    with open(input_file_name, 'rb') as source_file:
        byte: bytes = source_file.read(word_size)
        byte_hex: str = byte.hex()
        if byte_hex != '':
            result_list.append(byte.hex())
        while byte:
            byte = source_file.read(word_size)
            byte_hex: str = byte.hex()
            if byte_hex != '':
                result_list.append(byte.hex())
    return result_list


def main(input_file: str, input_instruction_file: str, input_data_file: str) -> None:
    input_tokens: list[str] = read_input_file(input_file)
    instructions: list[str] = read_input_byte_file(input_instruction_file, InstrMemoryConfig.word_size)
    data_memory: list[str] = read_input_byte_file(input_data_file, DataMemoryConfig.word_size)
    output, instr_counter, ticks = simulation(input_tokens, instructions, data_memory, 1000000)
    code_byte: int = len(instructions) * InstrMemoryConfig.word_size
    print("".join(output))
    print("code_byte: ", code_byte, "instr_counter: ", instr_counter, "ticks: ", ticks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    assert len(sys.argv) == 4, "Wrong arguments: machine.py <input_file> <input_instruction_file> <input_data_file>"
    _, input_file_arg, input_instruction_file_arg, input_data_file_arg = sys.argv
    main(input_file_arg, input_instruction_file_arg, input_data_file_arg)
