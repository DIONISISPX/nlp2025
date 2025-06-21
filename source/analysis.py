from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import re

text_1_original = (
    "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. "
    "Hope you too, to enjoy it as my deepest wishes. Thank your message to show our words to the doctor, as his next contract checking, to all of us. "
    "I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago. "
    "I am very appreciated the full support of the professor, for our Springer proceedings publication."
)

text_2_original = (
    "During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing "
    "as it not included the full feedback from reviewer or maybe editor? Anyway, I believe the team, although bit delay and less communication at recent days, "
    "they really tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think. "
    "Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again. "
    "Because I didn't see that part final yet, or maybe I missed, I apologize if so. "
    "Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets."
)

# Αποτελέσματα Παραδοτέου 1Α
sentence_1A_original_1 = "Hope you too, to enjoy it as my deepest wishes."
sentence_1A_original_2 = "Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation."

reconstructed_1A_1 = "I hope you enjoy it too.My best wishes."
reconstructed_1A_2 = "Anyway, I believe the team, although a bit of delay and less communication in recent days, they really tried their best on the paper and in our cooperation."


# Αποτελέσματα Παραδοτέου 1Β
paraphrased_texts_1B = {
    "text_1": {
        "original": text_1_original,
        "Vamsi/T5_Paraphrase_Paws": "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes. I got this message to see the approved message , in fact I received the message from the professor a couple of days ago to show me this . I am very appreciated the full support of the professor for our Springer proceedings publication .",
        "ramsrigouthamg/t5_paraphraser": "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes. I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago. I am very appreciated the full support of the professor, for our Springer proceedings publication.",
        "prithivida/parrot_paraphraser_on_T5": "Today is our dragon boat festival in our Chinese culture to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes. I got this message to see the approved message. In fact I received the message from the professor to show me this, a couple of days ago."
    },
    "text_2": {
        "original": text_2_original,
        "Vamsi/T5_Paraphrase_Paws": "During our final discussion, I told him about the new submission — the one we were waiting for since last autumn , but the updates was confusing as it did not include the full feedback from reviewer or maybe editor ? Anyway, I think the team really tried best for paper and cooperation . We should be grateful, I mean all of us , for the acceptance and efforts until the Springer link finally came last week , I think . Also, kindly remind me if the doctor still plan for the acknowledgments section edit before",
        "ramsrigouthamg/t5_paraphraser": "During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor? Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think. Also, kindly remind me please, if the doctor still plan for",
        "prithivida/parrot_paraphraser_on_T5": "I told him about the new submission — the one we were waiting since last autumn but the updates was confusing as it not included the full feedback from reviewer or maybe editor? Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link finally came last week. Also, please remind me if the doctor still plan for"
    }
}

# Φόρτωση Sentence-BERT Model
model_sbert = SentenceTransformer('all-MiniLM-L6-v2')

# Δημιουργούμε ένα ενιαίο dictionary που περιλαμβάνει και το 1Α και το 1Β
all_texts_for_analysis = {
    "deliverable_1A_sentence_1": {
        "original": sentence_1A_original_1,
        "custom_reconstruction": reconstructed_1A_1
    },
    "deliverable_1A_sentence_2": {
        "original": sentence_1A_original_2,
        "custom_reconstruction": reconstructed_1A_2
    }
}
all_texts_for_analysis.update(paraphrased_texts_1B) # Προσθέτουμε τα κείμενα του 1Β

# Υπολογισμός Cosine Similarity Scores
similarity_results = {}

for text_group_key, versions in all_texts_for_analysis.items():
    original_text = versions["original"]
    original_embedding = model_sbert.encode(original_text, convert_to_tensor=True)
    
    similarity_results[text_group_key] = {}
    
    for model_name, reconstructed_text in versions.items():
        if model_name == "original":
            continue
        
        reconstructed_embedding = model_sbert.encode(reconstructed_text, convert_to_tensor=True)
        
        cosine_score = util.pytorch_cos_sim(original_embedding, reconstructed_embedding).item()
        similarity_results[text_group_key][model_name] = cosine_score

# Σύγκριση Μεθόδων ως προς τα Α, Β του Παραδοτέου 1 (Βασισμένο σε Cosine Similarity)

# Ανάλυση για το Παραδοτέο 1Α
print("\nAnalysis for Deliverable 1A")
print("-" * 50)
for key in ["deliverable_1A_sentence_1", "deliverable_1A_sentence_2"]:
    scores = similarity_results.get(key, {})
    if scores:
        print(f"\n{key.upper()} Original vs Custom Reconstruction: ")
        custom_score = scores.get("custom_reconstruction", "N/A")
        print(f"Cosine Similarity = {custom_score:.4f}")


# Ανάλυση για το Παραδοτέο 1Β
print("\nAnalysis for Deliverable 1B")
print("-" * 50)
for text_key in ["text_1", "text_2"]:
    scores = similarity_results.get(text_key, {})
    if scores:
        print(f"\n{text_key.upper()} vs Pre-trained Models:")
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        print(f"Ranking by Cosine Similarity (Highest First):")
        for i, (model, score) in enumerate(sorted_scores, 1):
            print(f"    {i}. {model}: {score:.4f}")
    
# Οπτικοποίηση Ενσωματώσεων (PCA)
print("\nVisualizing Sentence Embeddings (PCA)...")

# Συγκεντρώνουμε όλα τα κείμενα (πρωτότυπα και ανακατασκευασμένα από 1Α & 1Β) για οπτικοποίηση
all_texts_for_viz = []
labels_for_viz = []
colors_for_viz = []
markers_for_viz = []

# Καθορισμός χρωμάτων και συμβόλων για σαφήνεια
text_group_colors = {
    "text_1": "blue",
    "text_2": "red",
    "deliverable_1A_sentence_1": "green",  
    "deliverable_1A_sentence_2": "purple"
}
model_markers = {
    "original": "o",                    # Κύκλος για τα αρχικά
    "Vamsi/T5_Paraphrase_Paws": "s",    # Τετράγωνο για Vamsi
    "ramsrigouthamg/t5_paraphraser": "^", # Τρίγωνο προς τα πάνω για ramsrigouthamg
    "prithivida/parrot_paraphraser_on_T5": "D", # Ρόμβος για parrot
    "custom_reconstruction": "X"        # X για custom ανακατασκευή
}

for text_group_key, versions in all_texts_for_analysis.items():
    current_color = text_group_colors[text_group_key]
    for model_name, text_content in versions.items():
        all_texts_for_viz.append(text_content)
        
        # Καθαρισμός του ονόματος μοντέλου για τη λεζάντα
        display_model_name = model_name
        if "deliverable_1A" in text_group_key:
             display_model_name = "Original" if model_name == "original" else "Custom Method"
        elif "original" == model_name:
            display_model_name = "Original"
            
        labels_for_viz.append(f"{text_group_key.replace('deliverable_1A_sentence_', '1A Sent ').upper()} - {display_model_name}")
        colors_for_viz.append(current_color)
        markers_for_viz.append(model_markers.get(model_name, "o"))

# Υπολογισμός embeddings για όλα τα κείμενα
all_embeddings = model_sbert.encode(all_texts_for_viz, convert_to_tensor=False)

pca = PCA(n_components=2, random_state=42)
reduced_embeddings_pca = pca.fit_transform(all_embeddings)

plt.figure(figsize=(14, 10))
# Συλλέγουμε μοναδικές ετικέτες για τη λεζάντα
unique_legend_handles = {}
for i, label in enumerate(labels_for_viz):
    # Κάθε συνδυασμός χρώματος-marker πρέπει να εμφανίζεται μία φορά στη λεζάντα
    color_marker_tuple = (colors_for_viz[i], markers_for_viz[i])
    if color_marker_tuple not in unique_legend_handles:
        unique_legend_handles[color_marker_tuple] = label # Αποθηκεύουμε την πρώτη εμφάνιση
    plt.scatter(reduced_embeddings_pca[i, 0], reduced_embeddings_pca[i, 1],
                color=colors_for_viz[i], marker=markers_for_viz[i], s=100)
    
plt.title('PCA of Sentence Embeddings (All Reconstructions)', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.grid(True)

# Δημιουργία προσαρμοσμένης λεζάντας από τις μοναδικές εγγραφές
legend_elements = []
for (color, marker), label in unique_legend_handles.items():
    legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w', label=label,
                                      markerfacecolor=color, markersize=10))

plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
plt.tight_layout()
plt.show()
