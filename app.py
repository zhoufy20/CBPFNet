import gradio as gr





# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Machine Learning for Accelerated Peak Force Prediction")
    gr.Markdown("### we introduce CBPFNet (a pretrained Covalent Bond Peak Force Network), "
                "an advanced graph attention network model that simulates the dynamic process of covalent bond cleavage and accurately "
                "predicts stress responses under various mechanical loads. Based on the trained CBPFNet, "
                "we have developed an automated program capable of predicting the strength of covalent bonds in organic molecules "
                "using the optimized structure file, CONTCAR.")

    with gr.Row():
        with gr.Column():
            gr.Image("Figure1.png", label="De novo crystal generation by MatterGPT targeting desired Eg, Ef",
                     width=1000, height=300)
            gr.Markdown(
                "**Enter desired properties to inversely design materials (encoded in SLICES), then decode it into crystal structure.**")
            gr.Markdown("**Allow 1-2 minutes for completion using 2 CPUs.**")

    with gr.Row():
        with gr.Column(scale=2):
            band_gap = gr.Number(label="Band Gap (eV)", value=2.0)
            formation_energy = gr.Number(label="Formation Energy (eV/atom)", value=-1.0)
            generate_button = gr.Button("Generate")

        with gr.Column(scale=3):
            slices_output = gr.Textbox(label="Generated SLICES String")
            cif_output = gr.File(label="Download CIF", file_types=[".cif"])
            structure_image = gr.Image(label="Structure Visualization")
            structure_summary = gr.Textbox(label="Structure Summary", lines=6)
            conversion_status = gr.Textbox(label="Conversion Status")

    # generate_button.click(
    #     generate_and_convert,
    #     inputs=[formation_energy, band_gap],
    #     outputs=[slices_output, cif_output, structure_image, structure_summary, conversion_status]
    # )

demo.launch(share=True)