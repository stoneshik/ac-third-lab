from __future__ import annotations

import pytest
from exceptions import AluNotSupportedInstrError, RegisterReadingError
from isa import Opcode
from machine import ALU, Registers


class TestRegisters:
    def test_signal_oer0(self) -> None:
        registers: Registers = Registers()
        registers.signal_oer0()
        assert registers.__getattribute__("_Registers__oer0")
        assert not registers.__getattribute__("_Registers__oer1")
        assert not registers.__getattribute__("_Registers__oer2")
        assert not registers.__getattribute__("_Registers__oer3")

    def test_signal_oer1(self) -> None:
        registers: Registers = Registers()
        registers.signal_oer1()
        assert not registers.__getattribute__("_Registers__oer0")
        assert registers.__getattribute__("_Registers__oer1")
        assert not registers.__getattribute__("_Registers__oer2")
        assert not registers.__getattribute__("_Registers__oer3")

    def test_signal_oer2(self) -> None:
        registers: Registers = Registers()
        registers.signal_oer2()
        assert not registers.__getattribute__("_Registers__oer0")
        assert not registers.__getattribute__("_Registers__oer1")
        assert registers.__getattribute__("_Registers__oer2")
        assert not registers.__getattribute__("_Registers__oer3")

    def test_signal_oer3(self) -> None:
        registers: Registers = Registers()
        registers.signal_oer3()
        assert not registers.__getattribute__("_Registers__oer0")
        assert not registers.__getattribute__("_Registers__oer1")
        assert not registers.__getattribute__("_Registers__oer2")
        assert registers.__getattribute__("_Registers__oer3")

    def test_output_data(self) -> None:
        registers: Registers = Registers()
        registers.r0_setter("00000000")
        registers.r1_setter("00000001")
        registers.r2_setter("00000002")
        registers.r3_setter("00000003")
        registers.signal_oer0()
        assert registers.output_data() == "00000000"
        registers.signal_oer1()
        assert registers.output_data() == "00000001"
        registers.signal_oer2()
        assert registers.output_data() == "00000002"
        registers.signal_oer3()
        assert registers.output_data() == "00000003"

    @pytest.mark.xfail(strict=True, raises=RegisterReadingError)
    def test_output_data_error(self) -> None:
        registers: Registers = Registers()
        registers.__setattr__("_Registers__oer0", False)
        registers.output_data()


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
