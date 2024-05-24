import spacy
import json
nlp = spacy.load("en_core_web_lg")

def extract_entities(text):
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

def generate_question(sentence, entity):
    if entity[1] == "PERSON":
        question = f"Who is {entity[0]}?"
    elif entity[1] == "DATE":
        question = f"When did {sentence.replace(entity[0], '____')}?"
    elif entity[1] == "GPE":
        question = f"Where is {entity[0]}?"
    else:
        question = f"What is {entity[0]}?"
    return question

if __name__ == "__main__":
    with open('sentences.txt', 'r') as f:
        sentences = f.readlines()

    qa_pairs = []
    for sentence in sentences:
        sentence = sentence.strip()
        entities = extract_entities(sentence)
        for entity in entities:
            question = generate_question(sentence, entity)
            answer = entity[0]
            qa_pairs.append({"question": question, "answer": answer})

    with open('qa_pairs.json', 'w') as f:
        json.dump(qa_pairs, f, indent=4)

    print("QA pairs generated and saved to qa_pairs.json")
