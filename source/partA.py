import stanza
import re

# Αρχικοποίηση του Stanza Pipeline
print("Loading Stanza pipeline...")
try:
    
    stanza.download('en', package='default')
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma', verbose=False)
    print("Pipeline loaded.")
except Exception as e:
    print(f"Failed to load Stanza: {e}. Please ensure 'stanza.download(\"en\")' has been run successfully.")
    exit()

# Συνάρτηση ανακατασκευής
def reconstruct_sentence(sentence: str) -> str:
    """
    Η τελική, βελτιωμένη συνάρτηση που λειτουργεί ως ο "αυτόματος μηχανισμός".
    Αναλύει μια πρόταση και εφαρμόζει μια βελτιωμένη σειρά κανόνων.
    """
    reconstructed_text = sentence

    # Κανόνες που εφαρμόζονται με απλή αντικατάσταση κειμένου
    # Κανόνας 2.1: Διόρθωση "bit delay" -> "a bit of delay"
    reconstructed_text = reconstructed_text.replace("bit delay", "a bit of delay")
    # Κανόνας 2.2: Διόρθωση "at recent days" -> "in recent days"
    reconstructed_text = reconstructed_text.replace("at recent days", "in recent days")
    # Κανόνας 2.4: Διόρθωση "tried best" -> "tried their best"
    reconstructed_text = reconstructed_text.replace("tried best", "tried their best")
    # Κανόνας 2.5: Διόρθωση "for paper and cooperation" -> "on the paper and in our cooperation"
    reconstructed_text = reconstructed_text.replace("for paper and cooperation", "on the paper and in our cooperation")

    # Κανόνες που απαιτούν γλωσσική ανάλυση με Stanza
    doc = nlp(reconstructed_text)
    words = [word.text for sent in doc.sentences for word in sent.words]
    lemmas = [word.lemma for sent in doc.sentences for word in sent.words]

    # Κανόνας 1.1: Εισαγωγή υποκειμένου "I"
    if lemmas and lemmas[0].lower() == 'hope' and words[0].lower() == 'hope' and words[0] != 'I':
        words.insert(0, 'I')
        # Μετά την εισαγωγή του 'I', αλλάζουμε το 'Hope' σε 'hope' αν είναι στην αρχή.
        if len(words) > 1 and words[1].lower() == 'hope' and words[1][0].isupper():
            words[1] = words[1].lower()

    # Κανόνας 1.2: Αφαίρεση του περιττού "to"
    try:
        hope_idx = -1
        to_idx = -1
        enjoy_idx = -1
        for i, word in enumerate(words):
            if word.lower() == 'hope' and hope_idx == -1:
                hope_idx = i
            elif hope_idx != -1 and word.lower() == 'to' and to_idx == -1:
                to_idx = i
            elif to_idx != -1 and word.lower() == 'enjoy' and enjoy_idx == -1:
                enjoy_idx = i
                break  # Βρέθηκε το μοτίβο, σταματάμε
        if hope_idx != -1 and to_idx != -1 and enjoy_idx != -1 and to_idx < enjoy_idx:
            words.pop(to_idx)
    except (ValueError, IndexError):
        pass  # Αν το μοτίβο δεν υπάρχει, δεν κάνουμε τίποτα

    # Κανόνας 1.3 & 1.4: Αντικατάσταση αφύσικης φράσης και μετακίνηση του "too"
    current_sentence_str = " ".join(words)
    too_pattern = r'\btoo\s*,\s*'
    wishes_pattern = r'\bas my deepest wishes\b'
    has_too = False
    if re.search(too_pattern, current_sentence_str, re.IGNORECASE):
        has_too = True
        current_sentence_str = re.sub(too_pattern, ' ', current_sentence_str, flags=re.IGNORECASE).strip()
    if re.search(wishes_pattern, current_sentence_str, re.IGNORECASE):
        current_sentence_str = re.sub(wishes_pattern, '', current_sentence_str, flags=re.IGNORECASE).strip()
    words = current_sentence_str.split()  # Ενημερώνουμε τη λίστα λέξεων

    # Ξανασυνθέτουμε την πρόταση 1
    if 'enjoy' in words and 'it' in words:
        try:
            enjoy_it_index = words.index('it')
            if has_too:
                words.insert(enjoy_it_index + 1, 'too')
            # Προσθέτουμε την φράση "My best wishes" ως ξεχωριστή πρόταση
            if 'My' not in words and 'best' not in words and 'wishes' not in words:
                words.append('.')
                words.append('My')
                words.append('best')
                words.append('wishes')
        except ValueError:
            pass  # Δεν βρέθηκε "enjoy it"

    # Κανόνας 2.3: Αφαίρεση πλεονάζουσας αντωνυμίας "they"
    try:
        team_idx = -1
        they_idx = -1
        for i, word in enumerate(words):
            if word.lower() == 'team,':
                team_idx = i
            if word.lower() == 'they' and team_idx != -1 and i > team_idx:
                they_idx = i
                break
        if they_idx != -1 and team_idx != -1 and they_idx > team_idx:
            words.pop(they_idx)
    except (ValueError, IndexError):
        pass

    # Ενώνουμε τις λέξεις και κάνουμε έναν τελικό καθαρισμό
    final_sentence = " ".join(words)
    final_sentence = final_sentence.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!").replace("  ", " ").strip()

    # Επιπλέον καθαρισμός: Διόρθωση διπλών διαστημάτων, κεφαλαίων στην αρχή πρότασης και τελεία στο τέλος
    final_sentence = re.sub(r'\s+', ' ', final_sentence).strip()
    sentences = re.split(r'([.?!])\s*', final_sentence)
    cleaned_sentences = []
    for i in range(0, len(sentences), 2):
        s = sentences[i].strip()
        if not s:
            continue
        if s and s[0].islower():
            s = s[0].upper() + s[1:]
        cleaned_sentences.append(s)
        if i + 1 < len(sentences):
            cleaned_sentences.append(sentences[i + 1])
    final_sentence = "".join(cleaned_sentences).strip()
    # Τελικός έλεγχος για τελεία στο τέλος αν λείπει
    if final_sentence and not final_sentence.endswith(('.', '?', '!')):
        final_sentence += '.'
    # Έλεγχος για διπλή τελεία
    final_sentence = final_sentence.replace("..", ".")
    return final_sentence

if __name__ == "__main__":
    sentence1 = "Hope you too, to enjoy it as my deepest wishes."
    sentence2 = "Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation."

    print(f"Αρχική 1: {sentence1}")
    reconstructed1 = reconstruct_sentence(sentence1)
    print(f"Ανακατασκευασμένη 1: {reconstructed1}")
    print("-" * 50)

    print(f"Αρχική 2: {sentence2}")
    reconstructed2 = reconstruct_sentence(sentence2)
    print(f"Ανακατασκευασμένη 2: {reconstructed2}")
    print("-" * 50) 