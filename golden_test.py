import contextlib
import io
import logging
import os
import tempfile

import machine
import pytest
import translator


@pytest.mark.golden_test("golden/*_aboba.yml")
def test_translator_and_machine(golden, caplog):
    # Установим уровень отладочного вывода на DEBUG
    caplog.set_level(logging.DEBUG)
    # Создаём временную папку для тестирования приложения.
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        # Готовим имена файлов для входных и выходных данных.
        source = os.path.join(tmp_dir_name, "source.aboba")
        input_stream = os.path.join(tmp_dir_name, "input.txt")
        target_instructions = os.path.join(tmp_dir_name, "instr.bin")
        target_data = os.path.join(tmp_dir_name, "data.bin")
        target_mnemonics = os.path.join(tmp_dir_name, "mnemonics.txt")
        # Записываем входные данные в файлы. Данные берутся из теста.
        with open(source, "w", encoding="utf-8") as file:
            file.write(golden["in_source"])
        with open(input_stream, "w", encoding="utf-8") as file:
            file.write(golden["in_stdin"])
        # Запускаем транслятор и собираем весь стандартный вывод в переменную stdout
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            translator.main(source, target_instructions, target_data, target_mnemonics)
            print("============================================================")
            machine.main(input_stream, target_instructions, target_data)
        # Выходные данные также считываем в переменные.
        with open(target_mnemonics, encoding="utf-8") as file:
            mnemonics = file.read()
        # Проверяем, что ожидания соответствуют реальности.
        assert mnemonics == golden.out["out_mnemonics"]
        assert stdout.getvalue() == golden.out["out_stdout"]
        caplog_end_index: int = caplog.text.find("DEBUG   machine:simulation    201 | TICK:")
        if caplog_end_index == -1:
            assert caplog.text == golden.out["out_log"]
        else:
            # Выводим первые сообщения первых 200 инструкций
            assert caplog.text[:caplog_end_index] == golden.out["out_log"]
