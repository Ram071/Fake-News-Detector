import pickle
import numpy as np
import streamlit as st


# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


def predict(title, author, text):
    """Predict the probability that the news is fake."""
    input_data = np.array([[title, author, text]], dtype=str)

    prediction = model.predict_proba(input_data)

    # Probability of the fake-news class
    fake_probability = prediction[0][0]

    return float(fake_probability)


def main():
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="📰",
        layout="centered"
    )

    st.title("📰 Fake News Detector")

    st.markdown(
        """
        <div style="
            background-color:#025246;
            padding:15px;
            border-radius:10px;
        ">
            <h2 style="color:white;text-align:center;">
                Fake News Detection using Machine Learning
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.write("Enter the details of the news article below.")

    title = st.text_input(
        "News Title",
        placeholder="Enter the news title"
    )

    author = st.text_input(
        "Author",
        placeholder="Enter the author's name"
    )

    text = st.text_area(
        "News Content",
        placeholder="Enter the complete news article here",
        height=200
    )

    if st.button("🔍 Predict", use_container_width=True):

        if not title or not text:
            st.warning("Please enter both the news title and news content.")
            return

        try:
            probability = predict(title, author, text)

            st.success(
                f"Probability that this news is fake: "
                f"{probability:.2%}"
            )

            if probability > 0.5:
                st.markdown(
                    """
                    <div style="
                        background-color:#F08080;
                        padding:15px;
                        border-radius:10px;
                    ">
                        <h2 style="color:black;text-align:center;">
                            ⚠️ News is likely Fake
                        </h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div style="
                        background-color:#F4D03F;
                        padding:15px;
                        border-radius:10px;
                    ">
                        <h2 style="color:black;text-align:center;">
                            ✅ News is likely Real
                        </h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


if __name__ == "__main__":
    main()
