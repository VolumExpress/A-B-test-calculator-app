from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_design_flow():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not at.exception

    at.text_input[0].input("Тестовый пилот")
    at.text_area[0].input("Treatment улучшит результат")
    at.button[0].click().run()
    assert not at.exception

    for _ in range(2):
        next_button = next(button for button in at.button if button.label == "Далее →")
        next_button.click().run()
        assert not at.exception

    calculate = next(button for button in at.button if button.label == "Рассчитать дизайн")
    calculate.click().run()
    assert not at.exception
    assert any(message.value.startswith("Расчёт выполнен") for message in at.success)

    next_button = next(button for button in at.button if button.label == "Далее →")
    next_button.click().run()
    assert not at.exception
    assert any(
        item.value == "Шаг 4. Результат проектирования"
        for item in at.subheader
    )


def test_streamlit_advanced_mode_opens():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30).run()
    at.sidebar.radio[0].set_value("Расширенные методы").run()
    assert not at.exception
    assert any(item.value == "Расширенные методы" for item in at.title)
    assert any(item.label == "Выберите метод" for item in at.selectbox)
