from infer import main
import gradio as gr

def infer_tiktok(image):
    result = main(image)
    return result

iface = gr.Interface(
    fn=infer_tiktok,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="People Segmentation",
    description="Upload an image to see the segmentation result."
)

if __name__ == "__main__":
    iface.launch(share=True)