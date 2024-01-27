@app.route('/generate_image1', methods=['POST'])
# def generate_image1():
#     try:
#         prompt = request.form.get('prompt')
#         print(prompt)
#         img = generate_style_image(prompt)
#         image_path = os.path.join(static_dir, "generated_style_img1.jpg")
#         img.save(image_path)
#         print("image generated and saved to ,",image_path)
#         return redirect(url_for('get_generated_style_image1'))
#     except Exception as e:
#         print(f"Error in generate_image: {str(e)}")


# @app.route('/generated_style_image1')
# def get_generated_style_image1():
#     return send_file(os.path.join(static_dir, "generated_style_img1.jpg"), mimetype='image/jpg')