"""Minimal Gradio 6.x test"""
import gradio as gr
print(f"Gradio version: {gr.__version__}")

def test_fn(files, choices):
    print(f"test_fn called! files={files}, choices={choices}")
    if not files:
        return []
    return [(f, f"file: {f}") for f in files[:1]]

with gr.Blocks() as demo:
    gr.Markdown("# Test")
    with gr.Row():
        with gr.Column():
            f = gr.File(file_count="multiple", label="Upload")
            c = gr.CheckboxGroup(choices=["A","B","C"], value=["A"], label="Pick")
            btn = gr.Button("Go")
        with gr.Column():
            g = gr.Gallery(label="Result")

    btn.click(fn=test_fn, inputs=[f, c], outputs=[g])

demo.launch(server_name="0.0.0.0", server_port=7861)
