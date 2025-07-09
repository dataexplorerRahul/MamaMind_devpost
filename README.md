## MamaMind - Gentle Guidance for New Beginnings App

## About: 
MamaMind is an intelligent mental health chatbot designed to provide support and assistance to individuals experiencing perinatal depression. The application offers a range of functionalities tailored to help users understand, manage, and alleviate their mental health challenges through a friendly and supportive interface. Here's a breakdown of its key features and capabilities:

1. **Welcome and Initial Interaction**
Upon launching the application, users are greeted with a warm welcome message that sets a comforting tone for their interaction. The chatbot introduces itself and offers users the option to engage in a mental health assessment through the Edinburgh Postnatal Depression Scale (EPDS) questionnaire.

2. **EPDS Questionnaire**
If users choose to proceed with the EPDS questionnaire, MamaMind guides them through a series of questions designed to evaluate their depression severity. The responses are collected, scored, and analyzed to provide an accurate assessment of the user's mental health status.

3. **Severity-Based Guidance**
Based on the results of the EPDS questionnaire, MamaMind categorizes the severity of the user's depression and offers tailored guidance and resources. This personalized approach ensures that users receive appropriate advice and support based on their specific needs.

4. **Interactive Chat Support**
Users can engage in an open-ended conversation with MamaMind by typing their concerns and questions into the chat input. The chatbot leverages advanced natural language processing techniques to understand the user's input and provide relevant responses. It uses knowledge of cognitive behavioral therapy, meditation techniques, mindfulness practices, and other therapeutic methods to guide users through their feelings and improve their well-being.

5. **Contextual Responses**
MamaMind employs a Retrieval-Augmented Generation (RAG) with query decomposition method to retrieve relevant information from a pre-built knowledge base of 84 PubMed articles, guideline documents recommended by OECD member countries related to perinatal depression and reliable resources such as WHO and NIMH. This ensures that the responses are contextually accurate and informative. The chatbot uses the retrieved context, along with the user's input and severity level, to generate comprehensive and helpful responses.

6. **Friendly Closure**
When users decide to end the conversation, MamaMind ensures a friendly and supportive closure. It responds positively to expressions of gratitude and offers a parting message that encourages users to return whenever they need further assistance.

7. **Continuous Interaction**
MamaMind is designed to facilitate continuous interaction. It asks relevant follow-up questions to maintain the conversation flow and provide ongoing support. The chatbot remains available to answer any new questions or concerns users might have.

**Key Benefits**
**Personalized Support:** Tailored advice and resources based on the user's depression severity.
**Expert Guidance:** Incorporates the knowledge base from the Pubmed research articles on perinatal depression and the guideline documents recommended by member countries of OECD. 
**User-Friendly Interface:** Easy-to-use chat interface.
**Contextual Accuracy:** Utilizes RAG using query decomposition to enhance the responses from the LLM and also ensuring accurate and reliable responses.
**Ongoing Availability:** Always available for continuous support and interaction.
MamaMind aims to create a safe and supportive environment for individuals experiencing perinatal depression, offering them the tools and resources needed to navigate their mental health journey.


**Main Requirements:**
Check the requirements.txt file for more information.

* faiss-cpu
* groq
* huggingface-hub
* langchain
* langchain-community
* langchain-core
* langchain-groq
* langchain-huggingface
* langchain-text-splitters
* python-dotenv
* sentence-transformers
* streamlit
* streamlit-extras
* transformers

**Installation:**

1. Clone this repository.
2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## **Running the App:**

1. Open the project in your preferred IDE (e.g., VS Code).
2. Start the Streamlit app:
    ```bash
    streamlit run welcome.py
    ```
3. Access the app in your web browser: http://localhost:8501

## **License:**

This project is licensed under the MIT License. See the LICENSE file for details.

## **Contact:**

For any questions or feedback, please contact our team : nprinka235@gmail.com, rahulmenon1758@gmail.com, vaishnavim1311@gmail.com and foutse.yuehgoh@gmail.com.

## Link to the streamlit application

The application has been deployed on Streamlit community cloud. 
Link ---> https://devposthackathonmamamind.streamlit.app/

## Conclusion
"MamaMind" is committed to providing effective, empathetic, and accessible mental health support for individuals experiencing perinatal depression. We are excited about the future and the potential to make a meaningful impact on the well-being of our users.

As a ground-breaking tool for maternal mental health, "MamaMind" is an example of how technology can be used to provide empathetic and effective support and ensure that no woman is left alone with perinatal depression. This innovative approach not only enhances the well-being of mothers, but also contributes to healthier family dynamics and better developmental outcomes for children. "MamaMind" is a testament to the potential of AI-driven solutions in transforming healthcare, offering hope and support to those who need it most.


