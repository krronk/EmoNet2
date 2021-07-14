import streamlit as st
from PIL import Image
from clf import predict

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Emotion Detection App")
st.write("This app predicts facial emotions of subjects present in your uploaded image. The model was trained with the [RAF-DB Dataset](http://www.whdeng.cn/RAF/model1.html). ")

st.write('** Code:** [Github repository](https://github.com/krronk/EmoNet2)')


st.write("**Upload your Image**")
file_up = st.file_uploader("Upload your image in JPG format", type="jpg")

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.title('**Class predictions:**')
    labels = predict(file_up)

    # print out the top 5 prediction labels with scores
    for i in labels:
        st.write(i[0], ",   Score: ", i[1])
