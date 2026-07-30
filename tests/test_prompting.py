from eval_tool.prompting import PromptTemplate, load_prompt_text


def test_load_prompt_text_reads_utf8_file(tmp_path):
    path = tmp_path / "prompt.txt"
    path.write_text("请回答：{question}", encoding="utf-8")

    assert load_prompt_text(path) == "请回答：{question}"


def test_prompt_template_renders_missing_fields_as_empty_string():
    prompt = PromptTemplate("题目：{question}\n选项A：{A}\n选项D：{D}")

    rendered = prompt.render({"question": "选哪个？", "A": "燃油泵"})

    assert "题目：选哪个？" in rendered
    assert "选项A：燃油泵" in rendered
    assert "选项D：" in rendered
