import streamlit as st
import tensorflow as tf


one_step_reloaded = tf.saved_model.load('one_step')


st.title("Conversational Text Generator using GRU-RNN")
st.subheader('')
n= st.number_input('Enter the number of charaters you want in the generated text.', format='%u', max_value=10000, value=100)
st.subheader("Enter your text prompt below 👇👇")
text= st.text_input("")
print(text)



if text!="" or text or not n:
    states = None
    next_char = tf.constant([text])
    result = [next_char]

    for n in range(int(n)):
        next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
        result.append(next_char)

    prompt= (tf.strings.join(result)[0].numpy().decode("utf-8"))

    st.subheader('Here is your generated text😁: ', prompt)
    st.write(prompt)

else:
    st.write('')