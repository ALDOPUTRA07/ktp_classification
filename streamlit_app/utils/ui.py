import streamlit as st


def header_description():
    st.title('KTP Classification')
    st.image('static/KTP Classification-UI.png')
    st.write(
        '''This project is an AI tool based on CNN with a transfer learning 
        method to classify a document, including KTP or Non-KTP.
        To get results, you only need to input an image, and the results will come out'''
    )


def dataset_description():
    url = "https://universe.roboflow.com/fauzan-ihza-fajar/ktp-hohxm"

    st.write(
        ''' **Datasets**
            check out this [link](%s)'''
        % url
    )


def result(output):
    st.subheader("Result")

    output_json = output.json()
    output_json['probability'] = round(output_json['probability'], 3)
    output_json['execution_time'] = str(round(output_json['execution_time'], 3)) + ' s'

    return st.json(output_json)
