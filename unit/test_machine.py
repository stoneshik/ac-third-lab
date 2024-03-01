from __future__ import annotations

import pytest
from exceptions import AluNotSupportedInstrError, IncorrectSelectorError, RegisterReadingError
from isa import AddressCode, Opcode, number_to_hex
from machine import ALU, DataPath, Registers
from memory_config import DataMemoryConfig
from translator import Translator


class TestRegisters:
    def test_output_data(self) -> None:
        registers: Registers = Registers()
        registers.r0_setter("00000000")
        registers.r1_setter("00000001")
        registers.r2_setter("00000002")
        registers.r3_setter("00000003")
        assert registers.output_data(AddressCode.DIRECT_REG.value + "000") == "00000000"
        assert registers.output_data(AddressCode.DIRECT_REG.value + "001") == "00000001"
        assert registers.output_data(AddressCode.DIRECT_REG.value + "002") == "00000002"
        assert registers.output_data(AddressCode.DIRECT_REG.value + "003") == "00000003"

    @pytest.mark.xfail(strict=True)
    def test_output_data_wrong_address_code_error(self) -> None:
        registers: Registers = Registers()
        registers.output_data("0000")

    @pytest.mark.xfail(strict=True, raises=RegisterReadingError)
    def test_output_data_register_reading_error(self) -> None:
        registers: Registers = Registers()
        sel: str = AddressCode.DIRECT_REG.value + "0015"
        registers.output_data(sel)


class TestALU:
    def test_calc_char(self) -> None:
        alu: ALU = ALU()
        alu.calc(Opcode.CHAR.value + "00", "12345608", "00000000")
        assert alu.result == "00000038"

    def test_calc_inc(self) -> None:
        alu: ALU = ALU()
        alu.calc(Opcode.INC.value + "00", "12345678", "00000000")
        assert alu.result == "12345679"

    def test_calc_dec(self) -> None:
        alu: ALU = ALU()
        alu.calc(Opcode.DEC.value + "00", "12345678", "00000000")
        assert alu.result == "12345677"

    def test_calc_add(self) -> None:
        alu: ALU = ALU()
        alu.calc(Opcode.ADD.value + "00", "00000005", "00000006")
        assert alu.result == "0000000b"

    def test_calc_sub(self) -> None:
        alu: ALU = ALU()
        alu.calc(Opcode.SUB.value + "00", "00000015", "00000005")
        assert alu.result == "00000010"

    def test_calc_mul(self) -> None:
        alu: ALU = ALU()
        alu.calc(Opcode.MUL.value + "00", "00000010", "00000005")
        assert alu.result == "00000050"

    def test_calc_div(self) -> None:
        alu: ALU = ALU()
        alu.calc(Opcode.DIV.value + "00", "0000000a", "00000003")
        assert alu.result == "00000003"

    def test_calc_slb(self) -> None:
        alu: ALU = ALU()
        alu.calc(Opcode.SLB.value + "00", "12345678", "00000000")
        assert alu.result == "34567800"

    def test_calc_srb(self) -> None:
        alu: ALU = ALU()
        alu.calc(Opcode.SRB.value + "00", "12345678", "00000000")
        assert alu.result == "00123456"

    def test_calc_mod(self) -> None:
        alu: ALU = ALU()
        alu.calc(Opcode.MOD.value + "00", "00000008", "00000005")
        assert alu.result == "00000003"

    def test_calc_and(self) -> None:
        alu: ALU = ALU()
        alu.calc(Opcode.AND.value + "00", "12345678", "0000000a")
        assert alu.result == "00000008"

    def test_calc_or(self) -> None:
        alu: ALU = ALU()
        alu.calc(Opcode.OR.value + "00", "00005678", "00001111")
        assert alu.result == "00005779"

    def test_calc_load_and_pop(self) -> None:
        alu: ALU = ALU()
        alu.calc(Opcode.LOAD.value + "00", "00000000", "12345678")
        assert alu.result == "12345678"
        alu.calc(Opcode.POP.value + "00", "00000000", "12345678")
        assert alu.result == "12345678"

    def test_calc_cmp(self) -> None:
        alu: ALU = ALU()
        alu.calc(Opcode.CMP.value + "00", "00000025", "00000015")
        assert alu.result == "00000010"
        assert not alu.zero
        alu.calc(Opcode.CMP.value + "00", "00000025", "00000025")
        assert alu.result == "00000000"
        assert alu.zero

    def test_calc_ies(self) -> None:
        alu: ALU = ALU()
        alu.calc(Opcode.IES.value + "00", "12345678", "00000000")
        assert alu.result == "12000000"
        assert not alu.zero
        alu.calc(Opcode.IES.value + "00", "00345678", "00000000")
        assert alu.result == "00000000"
        assert alu.zero

    @pytest.mark.xfail(strict=True)
    def test_calc_inc_overflow_error(self) -> None:
        alu: ALU = ALU()
        alu.calc(Opcode.INC.value + "00", "FFFFFFFF", "00000000")

    @pytest.mark.xfail(strict=True)
    def test_calc_add_overflow_error(self) -> None:
        alu: ALU = ALU()
        alu.calc(Opcode.ADD.value + "00", "FFFFFEEE", "0000C000")

    @pytest.mark.xfail(strict=True)
    def test_calc_mul_overflow_error(self) -> None:
        alu: ALU = ALU()
        alu.calc(Opcode.MUL.value + "00", "FFFFFEEE", "0000C000")

    @pytest.mark.xfail(strict=True)
    def test_calc_dec_negative_result_error(self) -> None:
        alu: ALU = ALU()
        alu.calc(Opcode.DEC.value + "00", "00000000", "00000000")

    @pytest.mark.xfail(strict=True)
    def test_calc_sub_negative_result_error(self) -> None:
        alu: ALU = ALU()
        alu.calc(Opcode.SUB.value + "00", "00000005", "00000010")

    @pytest.mark.xfail(strict=True, raises=AluNotSupportedInstrError)
    def test_calc_not_supported_error(self) -> None:
        alu: ALU = ALU()
        alu.calc("0000", "00000000", "00000000")


class TestDataPath:
    @classmethod
    def create_data_path(cls, input_buffer: list[str]) -> DataPath:
        zero_word_data_memory: str = "0" * DataMemoryConfig.word_hex_num
        data_memory: list[str] = [zero_word_data_memory] * DataMemoryConfig.size
        heap_counter: int = DataMemoryConfig.named_memory_size
        data_memory[heap_counter] = number_to_hex(DataMemoryConfig.word_hex_num, 0)
        heap_counter += 1
        data_memory[heap_counter] = number_to_hex(DataMemoryConfig.word_hex_num, 10)
        heap_counter += 1
        data_memory[heap_counter] = number_to_hex(DataMemoryConfig.word_hex_num, DataMemoryConfig.word_size)
        heap_counter += 1
        data_memory[heap_counter] = number_to_hex(DataMemoryConfig.word_hex_num, Translator.LOW_BYTE_FILTER_NUM)
        heap_counter += 1
        data_memory[heap_counter] = number_to_hex(DataMemoryConfig.word_hex_num, Translator.ZERO_ASCII_NUM)
        heap_counter += 1
        data_memory[heap_counter] = number_to_hex(DataMemoryConfig.word_hex_num, Translator.ZERO_LAST_WORD)
        heap_counter += 1
        data_memory[DataMemoryConfig.heap_counter_value] = number_to_hex(DataMemoryConfig.word_hex_num, heap_counter)
        return DataPath(input_buffer, data_memory)

    def test_get_value_from_data_memory(self) -> None:
        data_path: DataPath = self.create_data_path([])
        data_path.__setattr__(
            "_DataPath__address_register",
            number_to_hex(DataMemoryConfig.address_hex_num, DataMemoryConfig.named_memory_size + 1),
        )
        assert int(data_path.get_value_from_data_memory(), 16) == 10

    @pytest.mark.xfail(strict=True)
    def test_get_value_from_data_memory_reading_from_input_port_error(self) -> None:
        data_path: DataPath = self.create_data_path([])
        data_path.__setattr__(
            "_DataPath__address_register", number_to_hex(DataMemoryConfig.address_hex_num, DataMemoryConfig.input_port)
        )
        data_path.get_value_from_data_memory()

    @pytest.mark.xfail(strict=True)
    def test_get_value_from_data_memory_reading_from_output_port_error(self) -> None:
        data_path: DataPath = self.create_data_path([])
        data_path.__setattr__(
            "_DataPath__address_register", number_to_hex(DataMemoryConfig.address_hex_num, DataMemoryConfig.output_port)
        )
        data_path.get_value_from_data_memory()

    @pytest.mark.parametrize(("opcode", "expected_value"), [(Opcode.POP, "fff"), (Opcode.RET, "fff")])
    def test_signal_latch_stack_pointer_inc(self, opcode: Opcode, expected_value: str) -> None:
        data_path: DataPath = self.create_data_path([])
        sel: str = opcode.value + "0" * 2
        data_path.__setattr__("_DataPath__stack_pointer", "ffe")
        data_path.signal_latch_stack_pointer(sel)
        assert data_path.stack_pointer == expected_value

    @pytest.mark.parametrize(("opcode", "expected_value"), [(Opcode.PUSH, "ffe"), (Opcode.CALL, "ffe")])
    def test_signal_latch_stack_pointer_dec(self, opcode: Opcode, expected_value: str) -> None:
        data_path: DataPath = self.create_data_path([])
        sel: str = opcode.value + "0" * 2
        data_path.signal_latch_stack_pointer(sel)
        assert data_path.stack_pointer == expected_value

    @pytest.mark.xfail(strict=True)
    def test_signal_latch_stack_pointer_incorrect_selector_error(self) -> None:
        data_path: DataPath = self.create_data_path([])
        sel: str = Opcode.NOP.value + "0" * 2
        data_path.signal_latch_stack_pointer(sel)

    def test_signal_latch_stack_buffer(self) -> None:
        data_path: DataPath = self.create_data_path([])
        sel: str = AddressCode.INDIRECT_SP.value + "002"
        data_path.__setattr__("_DataPath__stack_pointer", "ffd")
        data_path.signal_latch_stack_buffer(sel)
        assert data_path.__getattribute__("_DataPath__stack_buffer") == "fff"

    @pytest.mark.xfail(strict=True)
    def test_signal_latch_stack_buffer_go_out_memory_data_error(self) -> None:
        data_path: DataPath = self.create_data_path([])
        sel: str = AddressCode.DIRECT_ABS.value + "001"
        data_path.signal_latch_stack_buffer(sel)

    @pytest.mark.xfail(strict=True, raises=IncorrectSelectorError)
    def test_signal_latch_stack_buffer_incorrect_selector_error(self) -> None:
        data_path: DataPath = self.create_data_path([])
        sel: str = AddressCode.DIRECT_ABS.value + "000"
        data_path.signal_latch_stack_buffer(sel)

    def test_signal_latch_heap_counter(self) -> None:
        data_path: DataPath = self.create_data_path([])
        data_path.signal_latch_heap_counter()
        assert int(data_path.heap_counter, 16) == int(
            data_path.__getattribute__("_DataPath__data_memory")[DataMemoryConfig.heap_counter_value], 16
        )

    @pytest.mark.parametrize("heap_counter_value", [1, DataMemoryConfig.size])
    @pytest.mark.xfail(strict=True)
    def test_signal_latch_heap_counter_go_out_range_error(self, heap_counter_value: int) -> None:
        data_path: DataPath = self.create_data_path([])
        data_path.__getattribute__("_DataPath__data_memory")[DataMemoryConfig.heap_counter_value] = number_to_hex(
            DataMemoryConfig.word_hex_num, heap_counter_value
        )
        data_path.signal_latch_heap_counter()

    def test_signal_latch_address_reg_oea(self) -> None:
        data_path: DataPath = self.create_data_path([])
        data_path.__setattr__("_DataPath__oea", True)
        data_path.__setattr__(
            "_DataPath__address_register",
            number_to_hex(DataMemoryConfig.address_hex_num, DataMemoryConfig.named_memory_size + 1),
        )
        data_path.signal_latch_address_reg("0000")
        assert data_path.__getattribute__("_DataPath__address_register") == number_to_hex(
            DataMemoryConfig.address_hex_num, 10
        )
        assert not data_path.__getattribute__("_DataPath__oea")

    @pytest.mark.parametrize("opcode", [Opcode.PUSH, Opcode.POP, Opcode.CALL, Opcode.RET])
    def test_signal_latch_address_reg_stack_pointer_value(self, opcode: Opcode) -> None:
        data_path: DataPath = self.create_data_path([])
        sel: str = opcode.value + "00"
        data_path.signal_latch_address_reg(sel)
        assert data_path.__getattribute__("_DataPath__address_register") == "fff"

    @pytest.mark.parametrize(
        ("address_code", "arg_address"), [(AddressCode.DIRECT_ABS, "001"), (AddressCode.DIRECT_OFFSET, "003")]
    )
    def test_signal_latch_address_reg_stack_arg_address_value(
        self, address_code: AddressCode, arg_address: str
    ) -> None:
        data_path: DataPath = self.create_data_path([])
        sel: str = address_code.value + arg_address
        data_path.signal_latch_address_reg(sel)
        assert data_path.__getattribute__("_DataPath__address_register") == arg_address

    def test_signal_latch_address_reg_stack_buffer_value(self) -> None:
        data_path: DataPath = self.create_data_path([])
        sel: str = AddressCode.INDIRECT_SP.value + "000"
        stack_buffer_value = "ffe"
        data_path.__setattr__("_DataPath__stack_buffer", stack_buffer_value)
        data_path.signal_latch_address_reg(sel)
        assert data_path.__getattribute__("_DataPath__address_register") == stack_buffer_value

    @pytest.mark.xfail(strict=True, raises=IncorrectSelectorError)
    def test_signal_latch_address_reg_incorrect_selector_error(self) -> None:
        data_path: DataPath = self.create_data_path([])
        sel: str = "0000"
        data_path.signal_latch_address_reg(sel)

    def test_signal_latch_second_op_buf(self) -> None:
        data_path: DataPath = self.create_data_path([])
        data_path.__setattr__(
            "_DataPath__address_register",
            number_to_hex(DataMemoryConfig.address_hex_num, DataMemoryConfig.named_memory_size + 1),
        )
        data_path.signal_latch_second_op_buf()
        assert data_path.__getattribute__("_DataPath__second_operand_buffer") == number_to_hex(
            DataMemoryConfig.word_hex_num, 10
        )

    def test_signal_latch_r0(self) -> None:
        data_path: DataPath = self.create_data_path([])
        registers: Registers = data_path.__getattribute__("_DataPath__registers")
        alu: ALU = data_path.__getattribute__("_DataPath__alu")
        alu_result: str = "00000002"
        alu.__setattr__("_ALU__result", alu_result)
        data_path.signal_latch_r0("0000")
        assert registers.r0 == alu_result

    def test_signal_latch_r1(self) -> None:
        data_path: DataPath = self.create_data_path([])
        registers: Registers = data_path.__getattribute__("_DataPath__registers")
        alu: ALU = data_path.__getattribute__("_DataPath__alu")
        alu_result: str = "00000002"
        alu.__setattr__("_ALU__result", alu_result)
        data_path.signal_latch_r1("0000")
        assert registers.r1 == alu_result

    def test_signal_latch_r2(self) -> None:
        data_path: DataPath = self.create_data_path([])
        registers: Registers = data_path.__getattribute__("_DataPath__registers")
        alu: ALU = data_path.__getattribute__("_DataPath__alu")
        alu_result: str = "00000002"
        alu.__setattr__("_ALU__result", alu_result)
        data_path.signal_latch_r2("0000")
        assert registers.r2 == alu_result

    def test_signal_latch_r3(self) -> None:
        data_path: DataPath = self.create_data_path([])
        registers: Registers = data_path.__getattribute__("_DataPath__registers")
        alu: ALU = data_path.__getattribute__("_DataPath__alu")
        alu_result: str = "00000002"
        alu.__setattr__("_ALU__result", alu_result)
        data_path.signal_latch_r3("0000")
        assert registers.r3 == alu_result

    def test_signal_latch_r0_read(self) -> None:
        data_path: DataPath = self.create_data_path(["a", "b"])
        registers: Registers = data_path.__getattribute__("_DataPath__registers")
        sel_instruction: str = Opcode.READ.value + "00"
        data_path.signal_latch_r0(sel_instruction + "00")
        assert registers.r0 == "61000000"

    def test_signal_latch_r1_read(self) -> None:
        data_path: DataPath = self.create_data_path(["a", "b"])
        registers: Registers = data_path.__getattribute__("_DataPath__registers")
        sel_instruction: str = Opcode.READ.value + "00"
        data_path.signal_latch_r1(sel_instruction + "00")
        assert registers.r1 == "61000000"

    def test_signal_latch_r2_read(self) -> None:
        data_path: DataPath = self.create_data_path(["a", "b"])
        registers: Registers = data_path.__getattribute__("_DataPath__registers")
        sel_instruction: str = Opcode.READ.value + "00"
        data_path.signal_latch_r2(sel_instruction + "00")
        assert registers.r2 == "61000000"

    def test_signal_latch_r3_read(self) -> None:
        data_path: DataPath = self.create_data_path(["a", "b"])
        registers: Registers = data_path.__getattribute__("_DataPath__registers")
        sel_instruction: str = Opcode.READ.value + "00"
        data_path.signal_latch_r3(sel_instruction + "00")
        assert registers.r3 == "61000000"

    def test_signal_write_in_mem(self) -> None:
        data_path: DataPath = self.create_data_path([])
        address_pointer: int = DataMemoryConfig.named_memory_size + 1
        data_path.__setattr__(
            "_DataPath__address_register", number_to_hex(DataMemoryConfig.address_hex_num, address_pointer)
        )
        registers: Registers = data_path.__getattribute__("_DataPath__registers")
        r_value: str = "00000005"
        sel: str = AddressCode.DIRECT_REG.value + "000"
        registers.r0_setter(r_value)
        data_path.signal_write_in_mem(sel)
        assert data_path.__getattribute__("_DataPath__data_memory")[address_pointer] == r_value

    def test_signal_write_in_mem_pc_value(self) -> None:
        data_path: DataPath = self.create_data_path([])
        address_pointer: int = DataMemoryConfig.named_memory_size + 1
        data_path.__setattr__(
            "_DataPath__address_register", number_to_hex(DataMemoryConfig.address_hex_num, address_pointer)
        )
        pc_value: str = "0015"
        sel: str = Opcode.CALL.value + "00"
        data_path.signal_write_in_mem(sel, pc_value=pc_value)
        assert data_path.__getattribute__("_DataPath__data_memory")[address_pointer] == pc_value

    @pytest.mark.xfail(strict=True)
    def test_signal_write_in_mem_writing_in_input_port_error(self) -> None:
        data_path: DataPath = self.create_data_path([])
        data_path.__setattr__(
            "_DataPath__address_register", number_to_hex(DataMemoryConfig.address_hex_num, DataMemoryConfig.input_port)
        )
        data_path.get_value_from_data_memory()

    @pytest.mark.xfail(strict=True)
    def test_signal_write_in_mem_writing_in_output_port_error(self) -> None:
        data_path: DataPath = self.create_data_path([])
        data_path.__setattr__(
            "_DataPath__address_register", number_to_hex(DataMemoryConfig.address_hex_num, DataMemoryConfig.output_port)
        )
        data_path.get_value_from_data_memory()

    def test_signal_output(self) -> None:
        data_path: DataPath = self.create_data_path([])
        registers: Registers = data_path.__getattribute__("_DataPath__registers")
        r_value: str = "12345678"
        sel: str = AddressCode.DIRECT_REG.value + "000"
        registers.r0_setter(r_value)
        data_path.signal_output(sel)
        assert data_path.output_buffer[-1] == "x"

    def test_signal_output_zero(self) -> None:
        data_path: DataPath = self.create_data_path([])
        registers: Registers = data_path.__getattribute__("_DataPath__registers")
        r_value: str = "12345600"
        sel: str = AddressCode.DIRECT_REG.value + "000"
        registers.r0_setter(r_value)
        data_path.signal_output(sel)
        assert len(data_path.output_buffer) == 0
